//! When an iOS window draws: the display link's schedule, as data.
//!
//! A window's `CADisplayLink` runs only while GPUI wants frames. Each frame that leaves no
//! demand behind pauses it, and GPUI's frame waker resumes it as soon as a view is notified
//! or an animation frame is requested. An idle window therefore costs no 120 Hz ticks.
//!
//! Resuming a paused link alone would add latency. The first tick after idle arrives at
//! the next vsync, and the frame drawn there reaches the glass a refresh later still. So a
//! wake from idle also draws at once, on the next main-queue turn, and the link paces
//! whatever follows. The immediate draw happens only when the previous frame is at least
//! one refresh old, which keeps a source that notifies faster than the display from
//! drawing faster than it too. It also happens only while the window is visible, because
//! iOS refuses GPU work from the background.
//!
//! The state machine is pure and compiled on the host too, so its tests run there.

use std::time::{Duration, Instant};

/// What the window must do after a wake.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Wake {
    /// The link is running or a frame is in progress; that frame or the next tick serves the
    /// demand.
    Nothing,
    /// Unpause the link; its next tick draws.
    Resume,
    /// Unpause the link and queue an immediate frame on the main queue.
    ResumeAndDrawNow,
}

/// Where a frame came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FrameSource {
    /// A `CADisplayLink` tick.
    DisplayLink,
    /// The immediate frame a [`Wake::ResumeAndDrawNow`] queued.
    Immediate,
}

#[derive(Debug)]
pub(crate) struct FramePacer {
    /// Whether the link is paused, as last requested.
    paused: bool,
    in_frame: bool,
    /// A wake arrived since the current frame began.
    demand: bool,
    immediate_pending: bool,
    last_frame_at: Option<Instant>,
    refresh_interval: Duration,
}

impl FramePacer {
    /// A pacer for a link that starts paused, on a display refreshing every
    /// `refresh_interval`.
    pub(crate) fn new(refresh_interval: Duration) -> Self {
        Self {
            paused: true,
            in_frame: false,
            demand: false,
            immediate_pending: false,
            last_frame_at: None,
            refresh_interval,
        }
    }

    /// GPUI wants a frame.
    pub(crate) fn wake(&mut self, now: Instant, visible: bool) -> Wake {
        self.demand = true;
        if self.in_frame || !self.paused {
            return Wake::Nothing;
        }
        self.paused = false;
        let idle_for_a_refresh = self
            .last_frame_at
            .is_none_or(|last| now.saturating_duration_since(last) >= self.refresh_interval);
        if visible && idle_for_a_refresh && !self.immediate_pending {
            self.immediate_pending = true;
            Wake::ResumeAndDrawNow
        } else {
            Wake::Resume
        }
    }

    /// A frame is about to run. Returns `false` for an immediate frame a display link tick
    /// already served, which must not run.
    pub(crate) fn begin_frame(&mut self, source: FrameSource, now: Instant) -> bool {
        if source == FrameSource::Immediate && !self.immediate_pending {
            return false;
        }
        self.immediate_pending = false;
        self.in_frame = true;
        self.demand = false;
        self.last_frame_at = Some(now);
        true
    }

    /// The frame finished. Returns whether the link must pause: nothing asked for another
    /// frame while this one ran.
    pub(crate) fn end_frame(&mut self) -> bool {
        self.in_frame = false;
        if self.demand || self.paused {
            return false;
        }
        self.paused = true;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REFRESH: Duration = Duration::from_micros(8_333);

    fn frame(pacer: &mut FramePacer, source: FrameSource, at: Instant, wake: bool) -> bool {
        assert!(pacer.begin_frame(source, at));
        if wake {
            assert_eq!(pacer.wake(at, true), Wake::Nothing);
        }
        pacer.end_frame()
    }

    #[test]
    fn a_wake_from_idle_draws_at_once_and_an_idle_frame_pauses() {
        let start = Instant::now();
        let mut pacer = FramePacer::new(REFRESH);
        assert_eq!(pacer.wake(start, true), Wake::ResumeAndDrawNow);
        assert!(
            frame(&mut pacer, FrameSource::Immediate, start, false),
            "a frame that leaves no demand pauses the link"
        );

        let later = start + REFRESH * 3;
        assert_eq!(pacer.wake(later, true), Wake::ResumeAndDrawNow);
    }

    #[test]
    fn an_animation_keeps_the_link_running_until_it_stops() {
        let start = Instant::now();
        let mut pacer = FramePacer::new(REFRESH);
        pacer.wake(start, true);
        assert!(!frame(&mut pacer, FrameSource::Immediate, start, true));
        for tick in 1..4 {
            let at = start + REFRESH * tick;
            assert!(
                !frame(&mut pacer, FrameSource::DisplayLink, at, true),
                "a frame that asks for the next keeps the link running"
            );
            assert_eq!(
                pacer.wake(at, true),
                Wake::Nothing,
                "a running link needs no resume"
            );
        }
        assert!(frame(
            &mut pacer,
            FrameSource::DisplayLink,
            start + REFRESH * 4,
            false
        ));
    }

    #[test]
    fn a_wake_within_a_refresh_of_the_last_frame_waits_for_the_link() {
        let start = Instant::now();
        let mut pacer = FramePacer::new(REFRESH);
        pacer.wake(start, true);
        assert!(frame(&mut pacer, FrameSource::Immediate, start, false));

        assert_eq!(
            pacer.wake(start + REFRESH / 2, true),
            Wake::Resume,
            "drawing again now would outrun the display"
        );
    }

    #[test]
    fn a_hidden_window_never_draws_immediately() {
        let mut pacer = FramePacer::new(REFRESH);
        assert_eq!(pacer.wake(Instant::now(), false), Wake::Resume);
    }

    #[test]
    fn an_immediate_frame_a_tick_already_served_does_not_run() {
        let start = Instant::now();
        let mut pacer = FramePacer::new(REFRESH);
        assert_eq!(pacer.wake(start, true), Wake::ResumeAndDrawNow);
        assert!(frame(
            &mut pacer,
            FrameSource::DisplayLink,
            start + REFRESH / 4,
            false
        ));
        assert!(
            !pacer.begin_frame(FrameSource::Immediate, start + REFRESH / 2),
            "the queued immediate frame is stale once a tick drew"
        );
    }

    #[test]
    fn repeated_wakes_queue_one_immediate_frame() {
        let start = Instant::now();
        let mut pacer = FramePacer::new(REFRESH);
        assert_eq!(pacer.wake(start, true), Wake::ResumeAndDrawNow);
        assert_eq!(pacer.wake(start, true), Wake::Nothing);
    }
}
