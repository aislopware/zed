//! Measures how long frames take to reach the display, using `Window::on_frame_presented`.
//!
//! `cargo run -p gpui --example frame_latency --profile release-fast -- <mode>` opens one
//! window, prints one line of numbers and quits:
//!
//! - `continuous`: a view that redraws every refresh, like a running animation.
//! - `heavy`: the same with thousands of laid-out cells (`FRAME_LATENCY_CELLS`, 4000 by
//!   default), so each frame costs most of a refresh.
//! - `idle`: a single frame after a quarter second of idle, like a keystroke in a quiet
//!   terminal. `wake_to_glass` is from the notify to the glass.
//! - `echo`: after a quarter second of idle, a notify (the keystroke, drawn at once like a
//!   local-echo guess) and a second one `FRAME_LATENCY_ECHO_US` later (3000 by default: the
//!   remote echo arriving). `wake_to_glass` is the keystroke's frame, `echo_to_glass` is from the
//!   keystroke to the glass of the frame holding the second notify.
//!
//! `submit_to_glass` is from the frame's submission to the GPU to the instant it was shown,
//! and `build` from the start of the view's render to that submission.
//! `missed` counts refreshes that showed no new frame while the view wanted one each refresh,
//! and `dropped` counts frames the system discarded unshown.

#![cfg_attr(target_family = "wasm", no_main)]

use std::{
    cell::RefCell,
    collections::VecDeque,
    rc::Rc,
    time::{Duration, Instant},
};

use gpui::{
    App, Bounds, Context, PresentedFrame, Subscription, Window, WindowBounds, WindowOptions, div,
    prelude::*, px, rgb, size,
};
use gpui_platform::application;

const WARMUP_FRAMES: usize = 60;
const CONTINUOUS_FRAMES: usize = 900;
const IDLE_SAMPLES: usize = 40;
const IDLE_GAP: Duration = Duration::from_millis(250);

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Continuous,
    Heavy,
    Idle,
    Echo,
}

#[derive(Default)]
struct Stats {
    seen: usize,
    submit_to_glass: Vec<f64>,
    wake_to_glass: Vec<f64>,
    build: Vec<f64>,
    render_started: VecDeque<Instant>,
    presented_at: Vec<Instant>,
    dropped: usize,
    wake: Option<Instant>,
    /// `echo`: the keystroke the pending second notify belongs to, and that notify.
    echo: Option<(Instant, Instant)>,
    echo_to_glass: Vec<f64>,
}

struct FrameLatency {
    mode: Mode,
    cells: usize,
    tick: u64,
    stats: Rc<RefCell<Stats>>,
    _presented: Subscription,
}

impl FrameLatency {
    fn new(mode: Mode, cells: usize, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let stats = Rc::new(RefCell::new(Stats::default()));
        let presented = window.on_frame_presented({
            let stats = stats.clone();
            move |frame, _, cx| {
                if record(&mut stats.borrow_mut(), mode, frame) {
                    report(mode, &stats.borrow());
                    cx.quit();
                }
            }
        });
        if matches!(mode, Mode::Idle | Mode::Echo) {
            let echo_after = (mode == Mode::Echo).then(|| {
                Duration::from_micros(
                    std::env::var("FRAME_LATENCY_ECHO_US")
                        .ok()
                        .and_then(|us| us.parse().ok())
                        .unwrap_or(3000),
                )
            });
            // Wakes come from another thread at a phase that walks across the refresh, as
            // keystrokes do, rather than from a main-thread timer that may coalesce with vsync.
            let (wakes, mut woken) = futures::channel::mpsc::unbounded();
            std::thread::spawn(move || {
                for step in 0u32.. {
                    std::thread::sleep(
                        IDLE_GAP + Duration::from_micros(u64::from(step % 16) * 830),
                    );
                    let key = Instant::now();
                    if wakes.unbounded_send((key, None)).is_err() {
                        break;
                    }
                    if let Some(after) = echo_after {
                        std::thread::sleep(after);
                        if wakes.unbounded_send((Instant::now(), Some(key))).is_err() {
                            break;
                        }
                    }
                }
            });
            let stats = stats.clone();
            cx.spawn(async move |this, cx| {
                while let Some((wake, key)) = futures::StreamExt::next(&mut woken).await {
                    match key {
                        None => stats.borrow_mut().wake = Some(wake),
                        Some(key) => stats.borrow_mut().echo = Some((key, wake)),
                    }
                    if this.update(cx, |_, cx| cx.notify()).is_err() {
                        break;
                    }
                }
            })
            .detach();
        }
        Self {
            mode,
            cells,
            tick: 0,
            stats,
            _presented: presented,
        }
    }
}

/// Returns whether the run has all its samples.
fn record(stats: &mut Stats, mode: Mode, frame: PresentedFrame) -> bool {
    stats.seen += 1;
    let mut render_started = None;
    while let Some(started) = stats
        .render_started
        .pop_front_if(|started| *started <= frame.submitted_at)
    {
        render_started = Some(started);
    }
    let warmup = if matches!(mode, Mode::Idle | Mode::Echo) {
        2
    } else {
        WARMUP_FRAMES
    };
    if stats.seen <= warmup {
        return false;
    }
    let Some(presented_at) = frame.presented_at else {
        stats.dropped += 1;
        return false;
    };
    let ms = |from: Instant| presented_at.saturating_duration_since(from).as_secs_f64() * 1e3;
    match mode {
        Mode::Continuous | Mode::Heavy => {
            stats.submit_to_glass.push(ms(frame.submitted_at));
            stats.presented_at.push(presented_at);
            if let Some(started) = render_started {
                stats
                    .build
                    .push((frame.submitted_at - started).as_secs_f64() * 1e3);
            }
            stats.submit_to_glass.len() >= CONTINUOUS_FRAMES
        }
        Mode::Idle => {
            if let Some(wake) = stats.wake.take_if(|wake| *wake <= frame.submitted_at) {
                stats.submit_to_glass.push(ms(frame.submitted_at));
                stats.wake_to_glass.push(ms(wake));
            }
            stats.submit_to_glass.len() >= IDLE_SAMPLES
        }
        Mode::Echo => {
            if let Some(wake) = stats.wake.take_if(|wake| *wake <= frame.submitted_at) {
                stats.wake_to_glass.push(ms(wake));
            }
            if let Some((key, _)) = stats.echo.take_if(|(_, echo)| *echo <= frame.submitted_at) {
                stats.submit_to_glass.push(ms(frame.submitted_at));
                stats.echo_to_glass.push(ms(key));
            }
            stats.echo_to_glass.len() >= IDLE_SAMPLES
        }
    }
}

fn summary(label: &str, samples: &[f64]) -> String {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let at = |q: f64| sorted[((sorted.len() - 1) as f64 * q).round() as usize];
    format!(
        "{label}_ms min={:.1} p50={:.1} p99={:.1} max={:.1}",
        at(0.0),
        at(0.5),
        at(0.99),
        at(1.0)
    )
}

fn report(mode: Mode, stats: &Stats) {
    let name = match mode {
        Mode::Continuous => "continuous",
        Mode::Heavy => "heavy",
        Mode::Idle => "idle",
        Mode::Echo => "echo",
    };
    let mut line = format!(
        "mode={name} frames={} dropped={}",
        stats.submit_to_glass.len(),
        stats.dropped
    );
    if matches!(mode, Mode::Continuous | Mode::Heavy) {
        let mut gaps: Vec<f64> = stats
            .presented_at
            .windows(2)
            .map(|pair| (pair[1] - pair[0]).as_secs_f64() * 1e3)
            .collect();
        gaps.sort_by(f64::total_cmp);
        let refresh = gaps[gaps.len() / 2];
        let missed: u64 = gaps
            .iter()
            .map(|gap| ((gap / refresh).round() as u64).saturating_sub(1))
            .sum();
        line += &format!(" refresh_ms={refresh:.2} missed={missed} ");
        line += &summary("build", &stats.build);
    }
    line += " ";
    line += &summary("submit_to_glass", &stats.submit_to_glass);
    if matches!(mode, Mode::Idle | Mode::Echo) {
        line += " ";
        line += &summary("wake_to_glass", &stats.wake_to_glass);
    }
    if mode == Mode::Echo {
        line += " ";
        line += &summary("echo_to_glass", &stats.echo_to_glass);
    }
    println!("{line}");
}

impl Render for FrameLatency {
    fn render(&mut self, window: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        self.tick += 1;
        self.stats
            .borrow_mut()
            .render_started
            .push_back(Instant::now());
        if matches!(self.mode, Mode::Continuous | Mode::Heavy) {
            window.request_animation_frame();
        }
        let offset = px((self.tick % 400) as f32);
        let tick = self.tick as usize;
        div()
            .size_full()
            .bg(rgb(0x202020))
            .child(
                div()
                    .absolute()
                    .top(px(8.))
                    .left(offset)
                    .size(px(40.))
                    .bg(rgb(0xff8000)),
            )
            .child(
                div()
                    .flex()
                    .flex_wrap()
                    .pt(px(56.))
                    .children((0..self.cells).map(|cell| {
                        let shade = ((cell + tick) % 256) as u32;
                        div()
                            .size(px(6.))
                            .m(px(1.))
                            .rounded(px(2.))
                            .border_1()
                            .border_color(rgb(0x404040))
                            .bg(rgb(shade << 16 | (255 - shade) << 8 | 0x40))
                    })),
            )
    }
}

fn main() {
    let mode = match std::env::args().nth(1).as_deref() {
        None | Some("continuous") => Mode::Continuous,
        Some("heavy") => Mode::Heavy,
        Some("idle") => Mode::Idle,
        Some("echo") => Mode::Echo,
        Some(other) => {
            eprintln!("unknown mode {other}; expected continuous, heavy, idle or echo");
            std::process::exit(2);
        }
    };
    let cells = match mode {
        Mode::Heavy => std::env::var("FRAME_LATENCY_CELLS")
            .ok()
            .and_then(|cells| cells.parse().ok())
            .unwrap_or(4000),
        Mode::Continuous | Mode::Idle | Mode::Echo => 0,
    };
    application().run(move |cx: &mut App| {
        let bounds = Bounds::centered(None, size(px(900.), px(700.)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                // Full rate even when another app keeps focus, as a run from a terminal does.
                inactive_frame_interval: None,
                ..Default::default()
            },
            |window, cx| cx.new(|cx| FrameLatency::new(mode, cells, window, cx)),
        )
        .expect("open the probe window");
        cx.activate(true);
    });
}
