//! Hardware keyboard presses (`UIPress` with a `UIKey`) turned into GPUI keystrokes.
//!
//! UIKit hands a physical key to the responder chain as `pressesBegan:` / `pressesEnded:`
//! before the text system sees it. Everything that is not plain text (modifier chords,
//! arrows, escape, function keys, enter, tab, backspace) is turned into a `KeyDown` /
//! `KeyUp` here, exactly the way `gpui_macos` shapes an `NSEvent`, so keymaps written for the
//! Mac work unchanged on an iPad with a keyboard. Plain characters are left to the text
//! system (`insertText:` on the first responder) while a text input is active, so IME and
//! marked text keep working; when nothing is editing they become keystrokes too.
//!
//! UIKit does not auto-repeat presses, so the window runs its own repeat timer.

use gpui::{Capslock, Keystroke, Modifiers};

/// `UIKeyModifierFlags`.
pub const ALPHA_SHIFT: u32 = 1 << 16;
pub const SHIFT: u32 = 1 << 17;
pub const CONTROL: u32 = 1 << 18;
pub const ALTERNATE: u32 = 1 << 19;
pub const COMMAND: u32 = 1 << 20;

/// What UIKit tells us about one key press (`UIKey`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UiKey {
    /// `UIKey.keyCode`, a USB HID usage from the keyboard page.
    pub hid: u32,
    /// `UIKey.characters`: what the key types with every modifier applied.
    pub characters: String,
    /// `UIKey.charactersIgnoringModifiers`: the same ignoring everything but Shift.
    pub characters_ignoring_modifiers: String,
    /// `UIKey.modifierFlags`.
    pub flags: u32,
}

/// The keyboard-page usages of the eight modifier keys (left/right ctrl, shift, alt, cmd).
pub const fn is_modifier_key(hid: u32) -> bool {
    matches!(hid, 0xE0..=0xE7)
}

/// The modifier state a flag word describes.
pub const fn modifiers(flags: u32) -> (Modifiers, Capslock) {
    (
        Modifiers {
            control: flags & CONTROL != 0,
            alt: flags & ALTERNATE != 0,
            shift: flags & SHIFT != 0,
            platform: flags & COMMAND != 0,
            function: false,
        },
        Capslock {
            on: flags & ALPHA_SHIFT != 0,
        },
    )
}

/// A key GPUI names rather than spells, by HID usage.
fn named_key(hid: u32) -> Option<&'static str> {
    Some(match hid {
        0x28 | 0x58 => "enter",
        0x29 => "escape",
        0x2A => "backspace",
        0x2B => "tab",
        0x2C => "space",
        0x39 => "capslock",
        0x3A => "f1",
        0x3B => "f2",
        0x3C => "f3",
        0x3D => "f4",
        0x3E => "f5",
        0x3F => "f6",
        0x40 => "f7",
        0x41 => "f8",
        0x42 => "f9",
        0x43 => "f10",
        0x44 => "f11",
        0x45 => "f12",
        0x49 => "insert",
        0x4A => "home",
        0x4B => "pageup",
        0x4C => "delete",
        0x4D => "end",
        0x4E => "pagedown",
        0x4F => "right",
        0x50 => "left",
        0x51 => "down",
        0x52 => "up",
        _ => return None,
    })
}

/// A key on the keyboard page that has an ASCII name even when the layout spells it
/// differently (Thai, Armenian…): the Mac behaves the same through `chars_for_modified_key`.
fn ascii_fallback(hid: u32) -> Option<String> {
    let ch = match hid {
        0x04..=0x1D => char::from(b'a' + u8::try_from(hid - 0x04).ok()?),
        0x1E..=0x26 => char::from(b'1' + u8::try_from(hid - 0x1E).ok()?),
        0x27 => '0',
        0x2D => '-',
        0x2E => '=',
        0x2F => '[',
        0x30 => ']',
        0x31 => '\\',
        0x33 => ';',
        0x34 => '\'',
        0x35 => '`',
        0x36 => ',',
        0x37 => '.',
        0x38 => '/',
        _ => return None,
    };
    Some(ch.to_string())
}

/// True for text the key would type: one character that is not a control code.
fn is_text(s: &str) -> bool {
    let mut chars = s.chars();
    match (chars.next(), chars.next()) {
        (Some(c), None) => !c.is_control(),
        _ => false,
    }
}

/// The keystroke a press produces, or `None` for a modifier key or a key nothing maps.
pub fn keystroke(key: &UiKey) -> Option<Keystroke> {
    if is_modifier_key(key.hid) {
        return None;
    }
    let (mut modifiers, _) = modifiers(key.flags);
    let plain = !modifiers.control && !modifiers.platform;

    if let Some(name) = named_key(key.hid) {
        let key_char = match name {
            "enter" if plain => Some("\n".to_string()),
            "tab" if plain => Some("\t".to_string()),
            "space" if plain => Some(" ".to_string()),
            _ => None,
        };
        return Some(Keystroke {
            modifiers,
            key: name.to_string(),
            key_char,
        });
    }

    let ignoring = &key.characters_ignoring_modifiers;
    let spelled = if is_text(ignoring) {
        ignoring.clone()
    } else {
        ascii_fallback(key.hid)?
    };
    // Same rule as the Mac: a shifted letter stays "shift-a"; a shifted symbol is the
    // symbol itself ("!" rather than "shift-1") and the modifier is dropped.
    // UIKit applies Shift to `charactersIgnoringModifiers` ("A", "!"); keymaps name the key
    // unshifted, so a letter becomes "shift-a" while a symbol is itself and drops the modifier.
    let mut spelled = spelled;
    if modifiers.shift {
        let lowered: String = spelled.chars().flat_map(char::to_lowercase).collect();
        if lowered == spelled {
            modifiers.shift = false;
        } else {
            spelled = lowered;
        }
    }
    let key_char = (plain && is_text(&key.characters)).then(|| key.characters.clone());
    Some(Keystroke {
        modifiers,
        key: spelled,
        key_char,
    })
}

/// Whether the text system should type this keystroke instead of us dispatching it: a
/// character with no chord modifier, which `insertText:` delivers through the input handler.
pub fn is_plain_text(keystroke: &Keystroke) -> bool {
    let m = &keystroke.modifiers;
    !m.control
        && !m.platform
        && !m.alt
        && keystroke
            .key_char
            .as_deref()
            .is_some_and(|c| !matches!(keystroke.key.as_str(), "enter" | "tab") && is_text(c))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(hid: u32, chars: &str, ignoring: &str, flags: u32) -> UiKey {
        UiKey {
            hid,
            characters: chars.to_string(),
            characters_ignoring_modifiers: ignoring.to_string(),
            flags,
        }
    }

    #[test]
    fn letters_keep_shift_symbols_drop_it() {
        let a = keystroke(&key(0x04, "A", "A", SHIFT)).unwrap();
        assert_eq!(a.key, "a");
        assert!(a.modifiers.shift);
        assert_eq!(a.key_char.as_deref(), Some("A"));
        let bang = keystroke(&key(0x1E, "!", "!", SHIFT)).unwrap();
        assert_eq!(bang.key, "!");
        assert!(!bang.modifiers.shift);
    }

    #[test]
    fn chords_have_no_key_char() {
        let c = keystroke(&key(0x06, "", "c", CONTROL)).unwrap();
        assert_eq!(c.key, "c");
        assert!(c.modifiers.control);
        assert_eq!(c.key_char, None);
        assert!(!is_plain_text(&c));
        let s = keystroke(&key(0x16, "ß", "s", ALTERNATE)).unwrap();
        assert_eq!(s.key, "s");
        assert_eq!(s.key_char.as_deref(), Some("ß"));
        assert!(!is_plain_text(&s));
    }

    #[test]
    fn named_keys_and_modifiers() {
        let up = keystroke(&key(0x52, "", "", COMMAND)).unwrap();
        assert_eq!(up.key, "up");
        assert!(up.modifiers.platform);
        let enter = keystroke(&key(0x28, "\r", "\r", 0)).unwrap();
        assert_eq!(
            (enter.key.as_str(), enter.key_char.as_deref()),
            ("enter", Some("\n"))
        );
        assert!(!is_plain_text(&enter));
        assert!(keystroke(&key(0xE0, "", "", CONTROL)).is_none());
        assert!(is_plain_text(&keystroke(&key(0x04, "a", "a", 0)).unwrap()));
        assert!(is_plain_text(&keystroke(&key(0x2C, " ", " ", 0)).unwrap()));
    }
}
