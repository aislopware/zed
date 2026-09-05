//! Input at the UIKit boundary, as data.
//!
//! UIKit hands the metal view `UIPress`es, `UITouch`es, a `UIPinchGestureRecognizer` and the
//! text system's `insertText:` / `deleteBackward`. None of those objects can be made from
//! outside UIKit (`UIPress`, `UITouch` and `UIKey` have no public initialiser), so a test
//! that wants to prove the UIKit delivery cannot post them. What it can do is *describe* one:
//! the primitive values the window reads out of the object (a HID usage and modifier flags,
//! a touch id, phase and point, a recognizer state and scale, a string). The window's UIKit
//! callbacks unpack the object into exactly these values and hand them to one delivery
//! function per kind; [`DescribedInput`] enters the same delivery function, so everything
//! after the unpacking runs the path a finger or a key takes.
//!
//! The mappings in here are pure and compiled on every target, so their tests run on the
//! host: raw `UITouchPhase` and `UIGestureRecognizerState` values to GPUI phases, and the
//! US layout stand-in that fills in what `UIKey.characters` would say for a usage.

use gpui::TouchPhase;

use crate::hardware_keyboard::{ALPHA_SHIFT, ALTERNATE, COMMAND, CONTROL, SHIFT, UiKey};

/// Which end of a hardware key press UIKit reported (`pressesBegan:` / `pressesEnded:` /
/// `pressesCancelled:`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PressPhase {
    /// `pressesBegan:`.
    Began,
    /// `pressesEnded:`.
    Ended,
    /// `pressesCancelled:`: released without a key-up, treated as an end.
    Cancelled,
}

/// One hardware key press as the window reads it from a `UIPress.key`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DescribedPress {
    /// `UIKey.keyCode`: a USB HID usage from the keyboard page.
    pub usage: u32,
    /// `UIKey.modifierFlags` (`UIKeyModifierFlags`).
    pub flags: u32,
    /// Which callback.
    pub phase: PressPhase,
}

/// `UITouchPhase`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i64)]
pub enum UiTouchPhase {
    /// `UITouchPhaseBegan`.
    Began = 0,
    /// `UITouchPhaseMoved`.
    Moved = 1,
    /// `UITouchPhaseStationary`.
    Stationary = 2,
    /// `UITouchPhaseEnded`.
    Ended = 3,
    /// `UITouchPhaseCancelled`.
    Cancelled = 4,
}

impl UiTouchPhase {
    /// The phase a raw `UITouchPhase` names; anything else (the regionEntered/Moved/Exited
    /// values of a hover) counts as cancelled, so a stray touch never sticks.
    pub const fn from_raw(value: i64) -> Self {
        match value {
            0 => Self::Began,
            1 => Self::Moved,
            2 => Self::Stationary,
            3 => Self::Ended,
            _ => Self::Cancelled,
        }
    }
}

impl From<UiTouchPhase> for TouchPhase {
    fn from(phase: UiTouchPhase) -> Self {
        match phase {
            UiTouchPhase::Began => TouchPhase::Started,
            UiTouchPhase::Moved | UiTouchPhase::Stationary => TouchPhase::Moved,
            UiTouchPhase::Ended => TouchPhase::Ended,
            UiTouchPhase::Cancelled => TouchPhase::Cancelled,
        }
    }
}

/// One touch of a `touchesBegan:` / `touchesMoved:` / `touchesEnded:` / `touchesCancelled:`
/// set, as the window reads it from the `UITouch`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DescribedTouch {
    /// Identifies the finger across its phases (UIKit keeps one `UITouch` object per finger;
    /// the window uses its address).
    pub id: u64,
    /// `UITouch.phase`.
    pub phase: UiTouchPhase,
    /// `locationInView:` the metal view, x in points.
    pub x: f32,
    /// `locationInView:` the metal view, y in points.
    pub y: f32,
}

/// `UIGestureRecognizerState`, the four values a continuous recognizer reports to its target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i64)]
pub enum GestureState {
    /// `UIGestureRecognizerStateBegan`.
    Began = 1,
    /// `UIGestureRecognizerStateChanged`.
    Changed = 2,
    /// `UIGestureRecognizerStateEnded`.
    Ended = 3,
    /// `UIGestureRecognizerStateCancelled`.
    Cancelled = 4,
}

impl GestureState {
    /// The state a raw value names, `None` for possible / failed (nothing to deliver).
    pub const fn from_raw(value: i64) -> Option<Self> {
        Some(match value {
            1 => Self::Began,
            2 => Self::Changed,
            3 => Self::Ended,
            4 => Self::Cancelled,
            _ => return None,
        })
    }

    /// The GPUI phase of a gesture in this state.
    pub const fn phase(self) -> TouchPhase {
        match self {
            Self::Began => TouchPhase::Started,
            Self::Changed => TouchPhase::Moved,
            Self::Ended => TouchPhase::Ended,
            Self::Cancelled => TouchPhase::Cancelled,
        }
    }
}

/// One report of the view's `UIPinchGestureRecognizer`, as the window reads it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DescribedPinch {
    /// `recognizer.state`.
    pub state: GestureState,
    /// `recognizer.scale` since the previous report (the window resets it to 1 after each).
    pub scale: f32,
    /// `locationInView:` the metal view, x in points.
    pub x: f32,
    /// `locationInView:` the metal view, y in points.
    pub y: f32,
}

/// Something UIKit could deliver to the window, described rather than posted.
#[derive(Debug, Clone, PartialEq)]
pub enum DescribedInput {
    /// A hardware key press (`pressesBegan:` and friends on the metal view).
    Press(DescribedPress),
    /// One `touches…:withEvent:` set: every touch in it shares the callback.
    Touches(Vec<DescribedTouch>),
    /// The pinch recognizer firing.
    Pinch(DescribedPinch),
    /// The text system's `insertText:` on the text input view.
    InsertText(String),
    /// The text system's `deleteBackward` on the text input view.
    DeleteBackward,
}

/// What a US keyboard types for a usage: `(unshifted, shifted)`, `None` for a key that types
/// nothing (arrows, modifiers, function keys).
const fn us_pair(usage: u32) -> Option<(char, char)> {
    Some(match usage {
        0x04..=0x1D => {
            let lower = (b'a' + (usage - 0x04) as u8) as char;
            (lower, lower.to_ascii_uppercase())
        }
        0x1E => ('1', '!'),
        0x1F => ('2', '@'),
        0x20 => ('3', '#'),
        0x21 => ('4', '$'),
        0x22 => ('5', '%'),
        0x23 => ('6', '^'),
        0x24 => ('7', '&'),
        0x25 => ('8', '*'),
        0x26 => ('9', '('),
        0x27 => ('0', ')'),
        0x28 | 0x58 => ('\r', '\r'),
        0x29 => ('\u{1b}', '\u{1b}'),
        0x2A => ('\u{8}', '\u{8}'),
        0x2B => ('\t', '\t'),
        0x2C => (' ', ' '),
        0x2D => ('-', '_'),
        0x2E => ('=', '+'),
        0x2F => ('[', '{'),
        0x30 => (']', '}'),
        0x31 => ('\\', '|'),
        0x33 => (';', ':'),
        0x34 => ('\'', '"'),
        0x35 => ('`', '~'),
        0x36 => (',', '<'),
        0x37 => ('.', '>'),
        0x38 => ('/', '?'),
        _ => return None,
    })
}

/// The `UIKey` a US keyboard would hand the window for `usage` under `flags`: what UIKit's
/// layout engine fills into `characters` and `charactersIgnoringModifiers`.
///
/// `charactersIgnoringModifiers` ignores everything but Shift (so `!` for shift-1, `A` for
/// shift-a); `characters` also applies Control (the C0 code of a letter, as UIKit reports it)
/// and is empty under Command, the way a Mac reports a chord. Option is left as the plain
/// character: the dead-key and symbol layer of a real layout is not modelled here.
pub fn us_key(usage: u32, flags: u32) -> UiKey {
    let shift = flags & SHIFT != 0;
    let caps = flags & ALPHA_SHIFT != 0;
    let (ignoring, characters) = match us_pair(usage) {
        Some((plain, shifted)) => {
            let letter = plain.is_ascii_lowercase();
            let ignoring = if shift || (caps && letter) {
                shifted
            } else {
                plain
            };
            let characters = if flags & COMMAND != 0 {
                String::new()
            } else if flags & CONTROL != 0 && letter {
                char::from((plain as u8) & 0x1f).to_string()
            } else {
                ignoring.to_string()
            };
            (ignoring.to_string(), characters)
        }
        None => (String::new(), String::new()),
    };
    let _ = ALTERNATE;
    UiKey {
        hid: usage,
        characters,
        characters_ignoring_modifiers: ignoring,
        flags,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware_keyboard::{is_plain_text, keystroke};

    #[test]
    fn touch_phases_map_like_uikit_documents_them() {
        assert_eq!(
            TouchPhase::from(UiTouchPhase::from_raw(0)),
            TouchPhase::Started
        );
        assert_eq!(
            TouchPhase::from(UiTouchPhase::from_raw(1)),
            TouchPhase::Moved
        );
        assert_eq!(
            TouchPhase::from(UiTouchPhase::from_raw(2)),
            TouchPhase::Moved
        );
        assert_eq!(
            TouchPhase::from(UiTouchPhase::from_raw(3)),
            TouchPhase::Ended
        );
        assert_eq!(
            TouchPhase::from(UiTouchPhase::from_raw(4)),
            TouchPhase::Cancelled
        );
        // Hover region phases (5..=7) and garbage never leave a finger down.
        assert_eq!(
            TouchPhase::from(UiTouchPhase::from_raw(6)),
            TouchPhase::Cancelled
        );
        assert_eq!(
            TouchPhase::from(UiTouchPhase::from_raw(-1)),
            TouchPhase::Cancelled
        );
    }

    #[test]
    fn gesture_states_deliver_only_while_active() {
        assert_eq!(GestureState::from_raw(0), None, "possible");
        assert_eq!(GestureState::from_raw(5), None, "failed");
        assert_eq!(
            GestureState::from_raw(1).map(GestureState::phase),
            Some(TouchPhase::Started)
        );
        assert_eq!(
            GestureState::from_raw(2).map(GestureState::phase),
            Some(TouchPhase::Moved)
        );
        assert_eq!(
            GestureState::from_raw(3).map(GestureState::phase),
            Some(TouchPhase::Ended)
        );
        assert_eq!(
            GestureState::from_raw(4).map(GestureState::phase),
            Some(TouchPhase::Cancelled)
        );
    }

    #[test]
    fn the_us_stand_in_types_what_a_keyboard_would() {
        let l = us_key(0x0F, 0);
        assert_eq!(
            (
                l.characters.as_str(),
                l.characters_ignoring_modifiers.as_str()
            ),
            ("l", "l")
        );
        let shift_l = us_key(0x0F, SHIFT);
        assert_eq!(shift_l.characters_ignoring_modifiers, "L");
        let caps_l = us_key(0x0F, ALPHA_SHIFT);
        assert_eq!(caps_l.characters, "L", "caps lock shifts letters");
        let caps_1 = us_key(0x1E, ALPHA_SHIFT);
        assert_eq!(caps_1.characters, "1", "but not digits");
        let bang = us_key(0x1E, SHIFT);
        assert_eq!(
            (
                bang.characters.as_str(),
                bang.characters_ignoring_modifiers.as_str()
            ),
            ("!", "!")
        );
        let ctrl_c = us_key(0x06, CONTROL);
        assert_eq!(ctrl_c.characters, "\u{3}");
        assert_eq!(ctrl_c.characters_ignoring_modifiers, "c");
        let cmd_shift_l = us_key(0x0F, COMMAND | SHIFT);
        assert_eq!(cmd_shift_l.characters, "", "a ⌘ chord types nothing");
        assert_eq!(cmd_shift_l.characters_ignoring_modifiers, "L");
        let up = us_key(0x52, 0);
        assert_eq!(
            (
                up.characters.as_str(),
                up.characters_ignoring_modifiers.as_str()
            ),
            ("", "")
        );
        assert_eq!(us_key(0x28, 0).characters, "\r");
    }

    /// The stand-in feeds the real mapping: the keystrokes it yields are the ones the Mac
    /// backend builds for the same chords, so a keymap test on the simulator means something.
    #[test]
    fn described_presses_become_the_keystrokes_the_mac_would_build() {
        let k = keystroke(&us_key(0x0F, COMMAND | SHIFT)).unwrap();
        assert_eq!(k.key, "l");
        assert!(k.modifiers.platform && k.modifiers.shift && !k.modifiers.control);
        assert_eq!(k.key_char, None);
        let up = keystroke(&us_key(0x52, 0)).unwrap();
        assert_eq!((up.key.as_str(), up.key_char), ("up", None));
        let a = keystroke(&us_key(0x04, 0)).unwrap();
        assert_eq!((a.key.as_str(), a.key_char.as_deref()), ("a", Some("a")));
        assert!(is_plain_text(&a));
        let bang = keystroke(&us_key(0x1E, SHIFT)).unwrap();
        assert_eq!(bang.key, "!");
        assert!(!bang.modifiers.shift);
        let ctrl_c = keystroke(&us_key(0x06, CONTROL)).unwrap();
        assert_eq!((ctrl_c.key.as_str(), ctrl_c.key_char), ("c", None));
        assert!(ctrl_c.modifiers.control);
        assert!(
            keystroke(&us_key(0xE3, COMMAND)).is_none(),
            "a modifier key itself"
        );
    }
}
