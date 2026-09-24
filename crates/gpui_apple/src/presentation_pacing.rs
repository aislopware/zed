//! Keeps a window's frames from queueing in front of the display.
//!
//! A `CAMetalLayer` shows its drawables in order, each for at least one refresh. A window that
//! draws once per vsync therefore keeps whatever queue it once built: two frames submitted
//! within one refresh (a late tick followed by a punctual one, a slow first frame, a
//! compositor hiccup) put a frame between every later frame and the glass, and every later
//! frame pays one refresh more for good. On a 75 Hz display that took continuous animation
//! from about 18 ms submit-to-glass to 31 ms.
//!
//! The pacer holds back one vsync tick when frames are arriving a refresh later than the
//! pipeline needs, which drains the queue by one frame, and it never begins two frames within half a
//! refresh, which stops a queue from forming out of tick jitter.
//!
//! What the pipeline needs, the floor, is the lowest submit-to-glass seen recently. A frame
//! reaches the glass at the first refresh its compositor deadline allows, so an unqueued
//! frame's latency lies within one refresh above the floor and a queued frame's latency is at
//! least a refresh above it. The floor only ever overestimates, which errs toward keeping a
//! queue rather than toward dropping frames. When a hold gains nothing, the floor was wrong
//! (the display or compositor path changed) and the pacer takes the new latency as its floor.

use std::time::{Duration, Instant};

use gpui::PresentedFrame;

/// Reports per floor epoch. The floor is the minimum over the current and previous epoch, so
/// it follows a slower pipeline within two epochs.
const FLOOR_EPOCH: u32 = 240;

/// Holds in a row that gain nothing before the pacer takes the latency as its new floor.
/// Fewer would let a busy machine, which can queue a frame again right after a hold, switch
/// pacing off for a while.
const FAILED_HOLDS_BEFORE_NEW_FLOOR: u32 = 3;

/// Reports to wait after the first failed hold before trying again, doubling per failure.
const COOLDOWN_REPORTS: u32 = 8;

#[derive(Debug)]
pub(crate) struct PresentationPacer {
    refresh: Duration,
    floor: EpochMin,
    /// The tick the last drawing tick approved, taken by the frame it drew.
    approved_tick: Option<Instant>,
    /// When the last frame began: its tick, or its submission when no tick drew it.
    last_frame_began: Option<Instant>,
    /// The last report was of a queued frame.
    queued: bool,
    last_latency: Duration,
    hold: Option<Hold>,
    failed_holds: u32,
    /// Reports left before another hold may follow a failed one.
    cooldown: u32,
}

#[derive(Debug)]
struct Hold {
    at: Instant,
    latency_before: Duration,
    judged: bool,
}

impl PresentationPacer {
    pub(crate) fn new(refresh: Duration) -> Self {
        Self {
            refresh,
            floor: EpochMin::starting_at(prior_floor(refresh)),
            approved_tick: None,
            last_frame_began: None,
            queued: false,
            last_latency: Duration::ZERO,
            hold: None,
            failed_holds: 0,
            cooldown: 0,
        }
    }

    /// The display's shortest refresh interval.
    pub(crate) fn set_refresh(&mut self, refresh: Duration) {
        if refresh != self.refresh {
            self.refresh = refresh;
            self.floor = EpochMin::starting_at(prior_floor(refresh));
        }
    }

    pub(crate) fn submitted(&mut self, at: Instant) {
        self.last_frame_began = Some(self.approved_tick.take().unwrap_or(at));
    }

    pub(crate) fn presented(&mut self, frame: PresentedFrame) {
        let Some(presented_at) = frame.presented_at else {
            return;
        };
        let latency = presented_at.saturating_duration_since(frame.submitted_at);
        if let Some(hold) = &mut self.hold {
            // Frames submitted before the hold were already queued; they say nothing of it.
            if frame.submitted_at < hold.at {
                return;
            }
            if !hold.judged {
                hold.judged = true;
                if latency + self.refresh / 2 > hold.latency_before {
                    self.failed_holds += 1;
                    if self.failed_holds >= FAILED_HOLDS_BEFORE_NEW_FLOOR {
                        self.failed_holds = 0;
                        self.floor.restart(latency);
                    } else {
                        self.cooldown = COOLDOWN_REPORTS << self.failed_holds;
                    }
                    self.queued = false;
                    return;
                }
                self.failed_holds = 0;
            }
        }
        self.cooldown = self.cooldown.saturating_sub(1);
        self.floor.observe(latency);
        let Some(floor) = self.floor.get() else {
            return;
        };
        self.queued = latency >= floor + self.refresh + self.refresh / 4;
        if self.queued {
            self.last_latency = latency;
        }
    }

    /// Whether no frame began within the last refresh.
    pub(crate) fn idle_for_a_refresh(&self, now: Instant) -> bool {
        self.last_frame_began
            .is_none_or(|last| now.saturating_duration_since(last) >= self.refresh)
    }

    /// Whether a vsync tick at `now` should draw. A tick this refuses leaves the window's
    /// demand for the next one.
    pub(crate) fn should_draw(&mut self, now: Instant) -> bool {
        if self.queued && self.cooldown == 0 {
            self.queued = false;
            self.hold = Some(Hold {
                at: now,
                latency_before: self.last_latency,
                judged: false,
            });
            return false;
        }
        let draw = self
            .last_frame_began
            .is_none_or(|last| now.saturating_duration_since(last) >= self.refresh / 2);
        if draw {
            self.approved_tick = Some(now);
        }
        draw
    }
}

/// The floor assumed before any frame shows the real one: a frame counts as queued once it
/// takes two refreshes. A window's first frames can queue before one of them ever reaches
/// the glass unqueued, and a floor learned only from them would never see the queue.
fn prior_floor(refresh: Duration) -> Duration {
    refresh * 3 / 4
}

/// The minimum over the current and the previous epoch of [`FLOOR_EPOCH`] samples.
#[derive(Debug, Default)]
struct EpochMin {
    current: Option<Duration>,
    previous: Option<Duration>,
    samples: u32,
}

impl EpochMin {
    fn observe(&mut self, sample: Duration) {
        self.current = Some(self.current.map_or(sample, |current| current.min(sample)));
        self.samples += 1;
        if self.samples >= FLOOR_EPOCH {
            self.previous = self.current.take();
            self.samples = 0;
        }
    }

    fn starting_at(sample: Duration) -> Self {
        let mut min = Self::default();
        min.observe(sample);
        min
    }

    fn restart(&mut self, sample: Duration) {
        *self = Self::starting_at(sample);
    }

    fn get(&self) -> Option<Duration> {
        match (self.current, self.previous) {
            (Some(current), Some(previous)) => Some(current.min(previous)),
            (current, previous) => current.or(previous),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REFRESH: Duration = Duration::from_micros(13_333);
    const FLOOR: Duration = Duration::from_millis(13);

    fn ms(value: f64) -> Duration {
        Duration::from_secs_f64(value / 1e3)
    }

    /// Where the compositor's deadline sits after a tick.
    const PHASE: Duration = Duration::from_millis(5);

    /// A display that shows a frame at the first refresh at least `floor` after its
    /// submission, and never sooner than a refresh after the frame ahead of it. A tick
    /// arrives every refresh.
    struct Display {
        pacer: PresentationPacer,
        start: Instant,
        /// Submission and presentation of frames not yet reported.
        in_flight: Vec<(Instant, Instant)>,
        last_frame: (Instant, Instant),
        holds: usize,
    }

    impl Display {
        /// A display whose first frame, submitted right at a deadline, reaches the glass
        /// unqueued `floor` later, and reports it when `report_first` holds.
        fn new(floor: Duration, report_first: bool) -> Self {
            let start = Instant::now();
            let mut pacer = PresentationPacer::new(REFRESH);
            let first = (start + PHASE, start + PHASE + floor);
            pacer.submitted(first.0);
            if report_first {
                pacer.presented(PresentedFrame {
                    submitted_at: first.0,
                    presented_at: Some(first.1),
                });
            }
            Self {
                pacer,
                start,
                in_flight: Vec::new(),
                last_frame: first,
                holds: 0,
            }
        }

        fn at(&self, tick: u32) -> Instant {
            self.start + REFRESH * (tick + 10)
        }

        fn submit(&mut self, submitted_at: Instant, floor: Duration) {
            self.pacer.submitted(submitted_at);
            let mut on_glass = self.last_frame.1;
            while on_glass < submitted_at + floor || on_glass < self.last_frame.1 + REFRESH {
                on_glass += REFRESH;
            }
            self.last_frame = (submitted_at, on_glass);
            self.in_flight.push(self.last_frame);
        }

        /// Reports every frame on the glass by the tick, then offers it to the pacer.
        fn tick(&mut self, tick: u32, floor: Duration, build: Duration) {
            let now = self.at(tick);
            let (shown, waiting): (Vec<_>, Vec<_>) = self
                .in_flight
                .drain(..)
                .partition(|(_, on_glass)| *on_glass <= now);
            self.in_flight = waiting;
            for (submitted_at, presented_at) in shown {
                self.pacer.presented(PresentedFrame {
                    submitted_at,
                    presented_at: Some(presented_at),
                });
            }
            if self.pacer.should_draw(now) {
                self.submit(now + build, floor);
            } else {
                self.holds += 1;
            }
        }

        fn latency(&self) -> Duration {
            self.last_frame.1 - self.last_frame.0
        }
    }

    #[test]
    fn an_unqueued_animation_is_never_held() {
        let mut display = Display::new(FLOOR, true);
        for tick in 0..600 {
            display.tick(tick, FLOOR, ms(0.3));
        }
        assert_eq!(display.holds, 0);
        assert_eq!(display.latency(), FLOOR + PHASE - ms(0.3));
    }

    #[test]
    fn a_queued_frame_is_drained_with_one_hold() {
        let mut display = Display::new(FLOOR, true);
        for tick in 0..100 {
            display.tick(tick, FLOOR, ms(0.3));
        }
        let unqueued = display.latency();
        // A second frame within the refresh, as a late tick followed by a punctual one
        // makes, bypassing the pacer.
        display.submit(display.at(99) + ms(1.), FLOOR);
        for tick in 100..400 {
            display.tick(tick, FLOOR, ms(0.3));
        }
        assert_eq!(display.holds, 1);
        assert_eq!(display.latency(), unqueued);
    }

    #[test]
    fn a_queue_from_the_first_frames_is_drained() {
        let mut display = Display::new(FLOOR, false);
        display.submit(display.at(0) - ms(2.), FLOOR);
        for tick in 0..300 {
            display.tick(tick, FLOOR, ms(0.3));
        }
        assert_eq!(display.holds, 1);
        assert_eq!(display.latency(), FLOOR + PHASE - ms(0.3));
    }

    #[test]
    fn a_frame_that_misses_its_deadline_is_not_queued() {
        // Built for 9 ms after the tick, each frame makes the refresh after the one an idle
        // frame makes; nothing is ahead of it, so holding a tick would only drop a frame.
        let mut display = Display::new(FLOOR, true);
        for tick in 0..600 {
            display.tick(tick, FLOOR, ms(9.));
        }
        assert_eq!(display.holds, 0);
    }

    #[test]
    fn a_slower_pipeline_costs_a_few_holds_then_becomes_the_floor() {
        let mut display = Display::new(FLOOR, true);
        let slower = FLOOR + REFRESH + ms(2.);
        for tick in 0..600 {
            display.tick(tick, slower, ms(0.3));
        }
        assert_eq!(display.holds, FAILED_HOLDS_BEFORE_NEW_FLOOR as usize);
        for tick in 600..1200 {
            display.tick(tick, slower, ms(0.3));
        }
        assert_eq!(display.holds, FAILED_HOLDS_BEFORE_NEW_FLOOR as usize);
    }

    #[test]
    fn a_queue_that_returns_right_after_a_hold_is_drained_later() {
        let mut display = Display::new(FLOOR, true);
        for tick in 0..100 {
            display.tick(tick, FLOOR, ms(0.3));
        }
        let unqueued = display.latency();
        display.submit(display.at(99) + ms(1.), FLOOR);
        let mut requeued = false;
        for tick in 100..400 {
            display.tick(tick, FLOOR, ms(0.3));
            if display.holds == 1 && !requeued {
                requeued = true;
                display.submit(display.at(tick) + ms(1.), FLOOR);
            }
        }
        assert_eq!(display.holds, 2);
        assert_eq!(display.latency(), unqueued);
    }

    #[test]
    fn two_frames_never_go_within_half_a_refresh() {
        let mut pacer = PresentationPacer::new(REFRESH);
        let now = Instant::now();
        assert!(pacer.should_draw(now));
        pacer.submitted(now);
        assert!(!pacer.should_draw(now + ms(3.)));
        assert!(pacer.should_draw(now + ms(7.)));
    }

    #[test]
    fn dropped_frames_teach_nothing() {
        let mut pacer = PresentationPacer::new(REFRESH);
        let now = Instant::now();
        pacer.presented(PresentedFrame {
            submitted_at: now,
            presented_at: None,
        });
        assert_eq!(pacer.floor.get(), Some(prior_floor(REFRESH)));
    }
}
