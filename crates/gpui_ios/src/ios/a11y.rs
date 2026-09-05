//! VoiceOver: the accesskit tree GPUI builds, mirrored as `UIAccessibilityElement`s on the
//! Metal view.
//!
//! UIKit has no accesskit adapter, so this is the bridge. `a11y_init` keeps GPUI's callbacks;
//! the first time UIKit asks the view for its accessibility elements (VoiceOver exploring the
//! screen), or when VoiceOver is already running at init, the activation callback turns the
//! tree on, and every `a11y_tree_update` rebuilds a flat, reading-order list of elements: one
//! per node with a label, a value or a click action (containers are walked, not exposed), its
//! frame in the view's coordinate space (points), traits from the role, the focused node
//! announced through `UIAccessibilityLayoutChangedNotification` whenever the list or the focus
//! changes. `accessibilityActivate` on an element dispatches accesskit's `Click` to GPUI.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use accesskit::{Action, ActionRequest, Node, NodeId, Role, TreeUpdate};
use gpui::A11yCallbacks;
use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use objc2::{DefinedClass, MainThreadMarker, MainThreadOnly, define_class, msg_send};
use objc2_core_foundation::{CGPoint, CGRect, CGSize};
use objc2_foundation::{NSArray, NSString};
use objc2_ui_kit::{
    UIAccessibilityElement, UIAccessibilityIsVoiceOverRunning,
    UIAccessibilityLayoutChangedNotification, UIAccessibilityPostNotification,
    UIAccessibilityTraitButton, UIAccessibilityTraitHeader, UIAccessibilityTraitImage,
    UIAccessibilityTraitKeyboardKey, UIAccessibilityTraitLink, UIAccessibilityTraitNone,
    UIAccessibilityTraitNotEnabled, UIAccessibilityTraitSearchField, UIAccessibilityTraitSelected,
    UIAccessibilityTraitStaticText, UIAccessibilityTraitUpdatesFrequently, UIAccessibilityTraits,
};
use parking_lot::Mutex;

/// The screen reader's way back into GPUI: shared by every element.
type ActionSink = Arc<Mutex<Box<dyn Fn(ActionRequest) + Send + 'static>>>;

/// What one element remembers: its accesskit node and where to send its activation.
pub(crate) struct ElementIvars {
    node: NodeId,
    action: ActionSink,
}

define_class!(
    /// A `UIAccessibilityElement` that forwards `accessibilityActivate` to GPUI as a click.
    #[unsafe(super(UIAccessibilityElement))]
    #[thread_kind = MainThreadOnly]
    #[name = "GPUIAccessibilityElement"]
    #[ivars = ElementIvars]
    pub(crate) struct Element;

    impl Element {
        #[unsafe(method(accessibilityActivate))]
        fn accessibility_activate(&self) -> bool {
            let ivars = self.ivars();
            (ivars.action.lock())(ActionRequest {
                action: Action::Click,
                target_tree: accesskit::TreeId::ROOT,
                target_node: ivars.node,
                data: None,
            });
            true
        }
    }
);

impl Element {
    fn new(mtm: MainThreadMarker, container: &AnyObject, ivars: ElementIvars) -> Retained<Self> {
        let this = Self::alloc(mtm).set_ivars(ivars);
        // SAFETY: `initWithAccessibilityContainer:` is UIAccessibilityElement's designated
        // initializer; `container` is the live Metal view that owns the element list.
        unsafe { msg_send![super(this), initWithAccessibilityContainer: container] }
    }
}

/// One window's bridge.
pub(crate) struct A11yBridge {
    /// The Metal view, the accessibility container (owned by the window).
    view: *mut AnyObject,
    activation: Box<dyn Fn() -> Option<TreeUpdate> + Send + 'static>,
    action: ActionSink,
    active: Cell<bool>,
    /// The current elements, in reading order, as handed to the view.
    elements: RefCell<Option<Retained<NSArray<Element>>>>,
    /// Digest of the last list (ids, labels, values, frames, traits), to post a layout change
    /// only when something the screen reader can perceive moved.
    digest: Cell<u64>,
    focus: Cell<Option<NodeId>>,
}

impl A11yBridge {
    pub(crate) fn new(view: *mut AnyObject, callbacks: A11yCallbacks) -> Self {
        let bridge = Self {
            view,
            activation: callbacks.activation,
            action: Arc::new(Mutex::new(callbacks.action)),
            active: Cell::new(false),
            elements: RefCell::new(None),
            digest: Cell::new(0),
            focus: Cell::new(None),
        };
        if UIAccessibilityIsVoiceOverRunning() {
            bridge.activate();
        }
        bridge
    }

    /// Turn GPUI's tree on; idempotent. GPUI redraws and the next frame arrives through
    /// [`Self::update`].
    pub(crate) fn activate(&self) {
        if !self.active.replace(true) {
            let _initial_tree = (self.activation)();
        }
    }

    /// The element list for the view's `accessibilityElements`, activating the tree on the
    /// first request (VoiceOver exploring the screen).
    pub(crate) fn elements(&self) -> Option<Retained<NSArray<Element>>> {
        self.activate();
        self.elements.borrow().clone()
    }

    /// A new frame's tree: rebuild the elements and tell VoiceOver when they changed.
    pub(crate) fn update(&self, update: &TreeUpdate, scale_factor: f32) {
        let Some(mtm) = MainThreadMarker::new() else {
            log::warn!("a11y tree update off the main thread; ignored");
            return;
        };
        let nodes: HashMap<NodeId, &Node> = update.nodes.iter().map(|(id, n)| (*id, n)).collect();
        let root = update.tree.as_ref().map_or(NodeId(0), |t| t.root);
        let mut order = Vec::new();
        walk(root, &nodes, &mut order);

        // SAFETY: the view outlives the bridge (the window owns both) and is a UIView.
        let container: &AnyObject = unsafe { &*self.view };
        let scale = f64::from(scale_factor.max(0.01));
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        let mut elements: Vec<Retained<Element>> = Vec::with_capacity(order.len());
        let mut focused: Option<usize> = None;
        for id in order {
            let Some(node) = nodes.get(&id) else { continue };
            let traits = traits_for(node);
            let element = Element::new(
                mtm,
                container,
                ElementIvars {
                    node: id,
                    action: Arc::clone(&self.action),
                },
            );
            element.setIsAccessibilityElement(true);
            let label = node.label().map(str::to_owned);
            let value = node.value().map(str::to_owned);
            if let Some(label) = &label {
                element.setAccessibilityLabel(Some(&NSString::from_str(label)));
            }
            if let Some(value) = &value {
                element.setAccessibilityValue(Some(&NSString::from_str(value)));
            }
            element.setAccessibilityTraits(traits);
            let frame = node.bounds().map(|r| CGRect {
                origin: CGPoint {
                    x: r.x0 / scale,
                    y: r.y0 / scale,
                },
                size: CGSize {
                    width: (r.x1 - r.x0) / scale,
                    height: (r.y1 - r.y0) / scale,
                },
            });
            if let Some(frame) = frame {
                element.setAccessibilityFrameInContainerSpace(frame);
                (frame.origin.x as i64, frame.origin.y as i64).hash(&mut hasher);
                (frame.size.width as i64, frame.size.height as i64).hash(&mut hasher);
            }
            id.0.hash(&mut hasher);
            label.hash(&mut hasher);
            value.hash(&mut hasher);
            traits.hash(&mut hasher);
            if id == update.focus {
                focused = Some(elements.len());
            }
            elements.push(element);
        }
        let digest = hasher.finish();
        let list_changed = digest != self.digest.replace(digest);
        let focus_changed = self.focus.replace(Some(update.focus)) != Some(update.focus);
        if !list_changed && !focus_changed {
            return;
        }

        let array = NSArray::from_retained_slice(&elements);
        // SAFETY: `setAccessibilityElements:` is UIAccessibilityContainer's informal-protocol
        // setter on NSObject; the receiver is the live Metal view and the array outlives the
        // call (UIKit retains it).
        let _: () = unsafe { msg_send![container, setAccessibilityElements: &*array] };
        *self.elements.borrow_mut() = Some(array);

        let argument = focus_changed
            .then(|| focused.and_then(|ix| elements.get(ix)))
            .flatten()
            .map(|element| {
                let object: &AnyObject = element;
                object
            });
        // SAFETY: `UIAccessibilityLayoutChangedNotification` takes an element to move the
        // screen reader's focus to, or nil to re-read the screen in place; called on the main
        // thread with an element the view's array retains.
        unsafe {
            UIAccessibilityPostNotification(UIAccessibilityLayoutChangedNotification, argument)
        };
    }
}

/// Depth-first reading order of the nodes a screen reader should visit: a node with a label,
/// a value or a click action is one stop; containers only lend their order.
fn walk(id: NodeId, nodes: &HashMap<NodeId, &Node>, out: &mut Vec<NodeId>) {
    let Some(node) = nodes.get(&id) else { return };
    if exposed(node) {
        out.push(id);
    }
    for child in node.children() {
        walk(*child, nodes, out);
    }
}

fn exposed(node: &Node) -> bool {
    !matches!(node.role(), Role::Window | Role::GenericContainer)
        && (node.label().is_some() || node.value().is_some() || node.supports_action(Action::Click))
}

/// UIKit traits for an accesskit role and state.
fn traits_for(node: &Node) -> UIAccessibilityTraits {
    // SAFETY: the `UIAccessibilityTrait*` statics are UIKit constants initialised at load.
    let (button, link, header, image, key, search, static_text, selected, disabled, none, live) = unsafe {
        (
            UIAccessibilityTraitButton,
            UIAccessibilityTraitLink,
            UIAccessibilityTraitHeader,
            UIAccessibilityTraitImage,
            UIAccessibilityTraitKeyboardKey,
            UIAccessibilityTraitSearchField,
            UIAccessibilityTraitStaticText,
            UIAccessibilityTraitSelected,
            UIAccessibilityTraitNotEnabled,
            UIAccessibilityTraitNone,
            UIAccessibilityTraitUpdatesFrequently,
        )
    };
    let mut traits = match node.role() {
        Role::Button | Role::DefaultButton | Role::Tab | Role::MenuItem | Role::ListBoxOption => {
            button
        }
        Role::Link => link,
        Role::Heading => header,
        Role::Image => image,
        Role::SearchInput => search,
        Role::TextInput | Role::MultilineTextInput => none,
        Role::Terminal | Role::Log | Role::Timer | Role::Status => static_text | live,
        Role::Label | Role::Paragraph => static_text,
        _ if node.supports_action(Action::Click) => button,
        _ => none,
    };
    if node.role_description().is_some_and(|d| d == "key") {
        traits |= key;
    }
    if node.is_selected() == Some(true) {
        traits |= selected;
    }
    if node.is_disabled() {
        traits |= disabled;
    }
    traits
}
