//! Native iOS platform support for GPUI.
//!
//! This crate connects GPUI to UIKit, CoreText, and Metal through the shared
//! Apple renderer. It owns iOS application lifecycle, windowing, native
//! text input, safe-area and keyboard insets, and raw touch delivery.

#[cfg(target_os = "ios")]
pub mod ios;

pub mod described;
pub mod hardware_keyboard;

#[cfg(target_os = "ios")]
pub use ios::{IosPlatform, current_platform};

/// Why a described input could not be delivered.
#[cfg(all(target_os = "ios", feature = "test-support"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InjectError {
    /// Called off the main thread; UIKit delivers on it and so does this.
    NotMainThread,
    /// No window is open.
    NoWindow,
}

#[cfg(all(target_os = "ios", feature = "test-support"))]
impl std::fmt::Display for InjectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::NotMainThread => "inject: not on the main thread",
            Self::NoWindow => "inject: no window",
        })
    }
}

#[cfg(all(target_os = "ios", feature = "test-support"))]
impl std::error::Error for InjectError {}

/// Delivers a described input to the window as if UIKit had: the same code the view's
/// `pressesBegan:`, `touchesBegan:`, pinch target, `insertText:` and `deleteBackward` run
/// once they have unpacked their objects. Main thread only, with no GPUI window leased (the
/// delivery re-enters GPUI's dispatch the way a real callback does). Test builds only.
#[cfg(all(target_os = "ios", feature = "test-support"))]
pub fn inject(input: described::DescribedInput) -> Result<(), InjectError> {
    ios::ffi::inject(input)
}

/// The foreground color used by the iOS status bar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StatusBarContentStyle {
    /// Light status-bar content for dark backgrounds.
    Light,
    /// Dark status-bar content for light backgrounds.
    #[default]
    Dark,
}
