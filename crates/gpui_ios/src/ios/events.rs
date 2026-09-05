//! Conversion from UIKit events to GPUI input events.

use gpui::{Pixels, Point, TouchId, px};
use objc2::msg_send;
use objc2::runtime::AnyObject;

use super::cg_types::ObjcCGPoint;
use crate::described::UiTouchPhase;

/// Returns the touch position in window coordinates.
pub fn touch_location_in_view(touch: *mut AnyObject, view: *mut AnyObject) -> Point<Pixels> {
    unsafe {
        let location: ObjcCGPoint = msg_send![touch, locationInView: view];
        Point::new(px(location.x as f32), px(location.y as f32))
    }
}

/// Returns the current UIKit touch phase.
pub fn touch_phase(touch: *mut AnyObject) -> UiTouchPhase {
    unsafe {
        let phase: i64 = msg_send![touch, phase];
        UiTouchPhase::from_raw(phase)
    }
}

/// Returns an identifier stable for the lifetime of this UIKit touch.
pub fn touch_id(touch: *mut AnyObject) -> TouchId {
    TouchId(touch as usize as u64)
}

/// Returns normalized pressure when UIKit reports a meaningful force range.
pub fn touch_force(touch: *mut AnyObject) -> Option<f32> {
    unsafe {
        let maximum_force: f64 = msg_send![touch, maximumPossibleForce];
        if maximum_force <= 0.0 {
            return None;
        }

        let force: f64 = msg_send![touch, force];
        Some((force / maximum_force).clamp(0.0, 1.0) as f32)
    }
}

/// Where UIKit predicts `touch` will be about a frame from now, in window coordinates. UIKit
/// only offers predictions while a touch moves; `None` otherwise.
pub fn predicted_touch_location(
    touch: *mut AnyObject,
    event: *mut AnyObject,
    view: *mut AnyObject,
) -> Option<Point<Pixels>> {
    if event.is_null() {
        return None;
    }
    unsafe {
        let predicted: *mut AnyObject = msg_send![event, predictedTouchesForTouch: touch];
        if predicted.is_null() {
            return None;
        }
        let last: *mut AnyObject = msg_send![predicted, lastObject];
        if last.is_null() {
            return None;
        }
        let location: ObjcCGPoint = msg_send![last, locationInView: view];
        Some(Point::new(px(location.x as f32), px(location.y as f32)))
    }
}
