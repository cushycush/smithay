//! Touch-related types for smithay's input abstraction

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use tracing::{info_span, instrument};
#[cfg(feature = "wayland_frontend")]
use wayland_server::Weak;

use crate::backend::input::TouchSlot;
use crate::utils::{IsAlive, Logical, Point, Serial};

pub use grab::{DefaultGrab, GrabStartData, TouchDownGrab, TouchGrab};

use super::{GrabStatus, Seat, SeatHandler};

mod grab;
#[cfg(test)]
mod tests;

crate::utils::ids::id_gen!(frame_marker);

/// A marker to identify a given touch frame.
///
/// This marker is sent to [`TouchTarget`] during `frame` and `cancel`, and can be returned via the
/// [`TouchTarget::last_frame`] function. They are used internally by [`TouchHandle`] to avoid
/// sending `frame` and `cancel` event more than once.
///
/// FrameMarker are used instead of comparing [`TouchTarget`] as different [`TouchTarget`] could
/// refer to the same underlying object (e.g. multiple surfaces belonging to the same wayland
/// `Client`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameMarker(usize);

/// An handle to a touch handler
///
/// It can be cloned and all clones manipulate the same internal state.
///
/// This handle gives you access to an interface to send touch events to your
/// clients.
///
/// When sending events using this handle, they will be intercepted by a touch
/// grab if any is active. See the [`TouchGrab`] trait for details.
pub struct TouchHandle<D: SeatHandler> {
    pub(crate) inner: Arc<Mutex<TouchInternal<D>>>,
    #[cfg(feature = "wayland_frontend")]
    pub(crate) known_instances: Arc<Mutex<Vec<Weak<wayland_server::protocol::wl_touch::WlTouch>>>>,
    pub(crate) span: tracing::Span,
}

#[cfg(not(feature = "wayland_frontend"))]
impl<D: SeatHandler> fmt::Debug for TouchHandle<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TouchHandle").field("inner", &self.inner).finish()
    }
}

#[cfg(feature = "wayland_frontend")]
impl<D: SeatHandler> fmt::Debug for TouchHandle<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TouchHandle")
            .field("inner", &self.inner)
            .field("known_instances", &self.known_instances)
            .finish()
    }
}

impl<D: SeatHandler> Clone for TouchHandle<D> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            #[cfg(feature = "wayland_frontend")]
            known_instances: self.known_instances.clone(),
            span: self.span.clone(),
        }
    }
}

impl<D: SeatHandler> std::hash::Hash for TouchHandle<D> {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.inner).hash(state)
    }
}

impl<D: SeatHandler> std::cmp::PartialEq for TouchHandle<D> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl<D: SeatHandler> std::cmp::Eq for TouchHandle<D> {}

pub(crate) struct TouchInternal<D: SeatHandler> {
    focus: HashMap<TouchSlot, TouchSlotState<D>>,
    pending_frame: Option<FrameMarker>,
    default_grab: Box<dyn Fn() -> Box<dyn TouchGrab<D>> + Send + 'static>,
    grab: GrabStatus<dyn TouchGrab<D>>,
}

impl<D: SeatHandler> Drop for TouchInternal<D> {
    fn drop(&mut self) {
        if let Some(marker) = self.pending_frame.take() {
            frame_marker::remove(marker.0);
        }
    }
}

struct TouchSlotState<D: SeatHandler> {
    focus: Option<(<D as SeatHandler>::TouchFocus, Point<f64, Logical>)>,
    delivered: Option<(<D as SeatHandler>::TouchFocus, Point<f64, Logical>)>,
    frame_pending: Option<<D as SeatHandler>::TouchFocus>,
    pending: FrameMarker,
    current: Option<FrameMarker>,
}

impl<D: SeatHandler> fmt::Debug for TouchSlotState<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TouchSlotState")
            .field("focus", &self.focus)
            .field("delivered", &self.delivered)
            .field("frame_pending", &self.frame_pending)
            .field("pending", &self.pending)
            .field("current", &self.current)
            .finish()
    }
}

// image_callback does not implement debug, so we have to impl Debug manually
impl<D: SeatHandler> fmt::Debug for TouchInternal<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TouchInternal")
            .field("focus", &self.focus)
            .field("grab", &self.grab)
            .finish()
    }
}

/// Pointer motion event
#[derive(Debug, Clone)]
pub struct DownEvent {
    /// Slot of this event
    pub slot: TouchSlot,
    /// Location of the touch in compositor space
    pub location: Point<f64, Logical>,
    /// Serial of the event
    pub serial: Serial,
    /// Timestamp of the event, with millisecond granularity
    pub time: u32,
}

/// Pointer motion event
#[derive(Debug, Clone)]
pub struct UpEvent {
    /// Slot of this event
    pub slot: TouchSlot,
    /// Serial of the event
    pub serial: Serial,
    /// Timestamp of the event, with millisecond granularity
    pub time: u32,
}

/// Pointer motion event
#[derive(Debug, Clone)]
pub struct MotionEvent {
    /// Slot of this event
    pub slot: TouchSlot,
    /// Location of the touch in compositor space
    pub location: Point<f64, Logical>,
    /// Timestamp of the event, with millisecond granularity
    pub time: u32,
}

/// Pointer motion event
#[derive(Debug, Clone, Copy)]
pub struct ShapeEvent {
    /// Slot of this event
    pub slot: TouchSlot,
    /// Length of the major axis in surface-local coordinates
    pub major: f64,
    /// Length of the minor axis in surface-local coordinates
    pub minor: f64,
}

/// Pointer motion event
#[derive(Debug, Clone, Copy)]
pub struct OrientationEvent {
    /// Slot of this event
    pub slot: TouchSlot,
    /// Angle between major axis and positive surface y-axis in degrees
    pub orientation: f64,
}

/// Trait representing object that can receive touch interactions
pub trait TouchTarget<D>: IsAlive + fmt::Debug + Send
where
    D: SeatHandler,
{
    /// A new touch point has appeared on the target.
    ///
    /// This touch point is assigned a unique ID. Future events from this touch point reference this ID.
    /// The ID ceases to be valid after a touch up event and may be reused in the future.
    fn down(&self, seat: &Seat<D>, data: &mut D, event: &DownEvent);

    /// The touch point has disappeared.
    ///
    /// No further events will be sent for this touch point and the touch point's ID
    /// is released and may be reused in a future touch down event.
    fn up(&self, seat: &Seat<D>, data: &mut D, event: &UpEvent);

    /// A touch point has changed coordinates.
    fn motion(&self, seat: &Seat<D>, data: &mut D, event: &MotionEvent);

    /// Indicates the end of a set of events that logically belong together.
    ///
    /// The `marker` parameter is used to avoid re-sending the cancel event if the [`TouchTarget`]
    /// has multiple active touch slot, or if multiple [`TouchTarget`] belong to the same underlying
    /// target (e.g. a single client). See [`TouchTarget::last_frame`].
    fn frame(&self, seat: &Seat<D>, data: &mut D, marker: FrameMarker);

    /// Touch session cancelled.
    ///
    /// Touch cancellation applies to all touch points currently active on this target.
    /// The client is responsible for finalizing the touch points, future touch points on
    /// this target may reuse the touch point ID.
    ///
    /// The `marker` parameter is used to avoid re-sending the cancel event if the [`TouchTarget`]
    /// has multiple active touch slot, or if multiple [`TouchTarget`] belong to the same underlying
    /// target (e.g. a single client). See [`TouchTarget::last_frame`].
    fn cancel(&self, seat: &Seat<D>, data: &mut D, marker: FrameMarker);

    /// Sent when a touch point has changed its shape.
    ///
    /// A touch point shape is approximated by an ellipse through the major and minor axis length.
    /// The major axis length describes the longer diameter of the ellipse, while the minor axis
    /// length describes the shorter diameter. Major and minor are orthogonal and both are specified
    /// in surface-local coordinates. The center of the ellipse is always at the touch point location
    /// as reported by [`TouchTarget::down`] or [`TouchTarget::motion`].
    fn shape(&self, seat: &Seat<D>, data: &mut D, event: &ShapeEvent);

    /// Sent when a touch point has changed its orientation.
    ///
    /// The orientation describes the clockwise angle of a touch point's major axis to the positive surface
    /// y-axis and is normalized to the -180 to +180 degree range. The granularity of orientation depends
    /// on the touch device, some devices only support binary rotation values between 0 and 90 degrees.
    fn orientation(&self, seat: &Seat<D>, data: &mut D, event: &OrientationEvent);

    /// Returns last know [`FrameMarker`].
    ///
    /// If this function returns `Some(marker)`, [`TouchHandle`] will use that value to avoid
    /// sending the `frame` or `cancel` event more than once.
    fn last_frame(&self, seat: &Seat<D>, data: &mut D) -> Option<FrameMarker>;
}

impl<D: SeatHandler + 'static> TouchHandle<D> {
    pub(crate) fn new<F>(default_grab: F) -> TouchHandle<D>
    where
        F: Fn() -> Box<dyn TouchGrab<D>> + Send + 'static,
    {
        TouchHandle {
            inner: Arc::new(Mutex::new(TouchInternal::new(default_grab))),
            #[cfg(feature = "wayland_frontend")]
            known_instances: Arc::new(Mutex::new(Vec::new())),
            span: info_span!("input_touch"),
        }
    }

    /// Change the current grab on this touch to the provided grab
    ///
    /// Overwrites any current grab.
    #[instrument(level = "debug", parent = &self.span, skip(self, data, grab))]
    pub fn set_grab<G: TouchGrab<D> + 'static>(&self, data: &mut D, grab: G, serial: Serial) {
        let seat = self.get_seat(data);
        self.inner.lock().unwrap().set_grab(data, &seat, serial, grab);
    }

    /// Remove any current grab on this touch, resetting it to the default behavior
    #[instrument(level = "debug", parent = &self.span, skip(self, data))]
    pub fn unset_grab(&self, data: &mut D) {
        let seat = self.get_seat(data);
        self.inner.lock().unwrap().unset_grab(data, &seat);
    }

    /// Check if this touch is currently grabbed with this serial
    pub fn has_grab(&self, serial: Serial) -> bool {
        let guard = self.inner.lock().unwrap();
        match guard.grab {
            GrabStatus::Active(s, _) => s == serial,
            _ => false,
        }
    }

    /// Check if this touch is currently being grabbed
    pub fn is_grabbed(&self) -> bool {
        let guard = self.inner.lock().unwrap();
        !matches!(guard.grab, GrabStatus::None)
    }

    /// Returns the start data for the grab, if any.
    pub fn grab_start_data(&self) -> Option<GrabStartData<D>> {
        let guard = self.inner.lock().unwrap();
        match &guard.grab {
            GrabStatus::Active(_, g) => Some(g.start_data().clone()),
            _ => None,
        }
    }

    /// Calls `f` with the active grab, if any.
    pub fn with_grab<T>(&self, f: impl FnOnce(Serial, &dyn TouchGrab<D>) -> T) -> Option<T> {
        let guard = self.inner.lock().unwrap();
        if let GrabStatus::Active(s, g) = &guard.grab {
            Some(f(*s, &**g))
        } else {
            None
        }
    }

    /// Notify that a new touch point appeared
    ///
    /// You provide the location of the touch, in the form of:
    ///
    /// - The coordinates of the touch in the global compositor space
    /// - The surface on top of which the touch point is, and the coordinates of its
    ///   origin in the global compositor space (or `None` of the touch is not
    ///   on top of a client surface).
    pub fn down(
        &self,
        data: &mut D,
        focus: Option<(<D as SeatHandler>::TouchFocus, Point<f64, Logical>)>,
        event: &DownEvent,
    ) {
        let mut inner = self.inner.lock().unwrap();
        let seat = self.get_seat(data);
        inner.with_grab(data, &seat, |data, handle, grab| {
            grab.down(data, handle, focus, event);
        });
    }

    /// Notify that a touch point disappeared
    pub fn up(&self, data: &mut D, event: &UpEvent) {
        let mut inner = self.inner.lock().unwrap();
        let seat = self.get_seat(data);
        inner.with_grab(data, &seat, |data, handle, grab| {
            grab.up(data, handle, event);
        });
    }

    /// Notify that a touch point has changed coordinates.
    ///
    /// You provide the location of the touch, in the form of:
    ///
    /// - The coordinates of the touch in the global compositor space
    /// - The surface on top of which the touch point is, and the coordinates of its
    ///   origin in the global compositor space (or `None` of the touch is not
    ///   on top of a client surface).
    ///
    /// **Note** that this will **not** update the focus of the touch point, the focus
    /// is only set on [`TouchHandle::down`]. The focus provided to this function
    /// can be used to find DnD targets during touch motion.
    pub fn motion(
        &self,
        data: &mut D,
        focus: Option<(<D as SeatHandler>::TouchFocus, Point<f64, Logical>)>,
        event: &MotionEvent,
    ) {
        let mut inner = self.inner.lock().unwrap();
        let seat = self.get_seat(data);
        inner.with_grab(data, &seat, |data, handle, grab| {
            grab.motion(data, handle, focus, event);
        });
    }

    /// Notify about the end of a set of events that logically belong together.
    ///
    /// This needs to be called after one or move calls to [`TouchHandle::down`] or [`TouchHandle::motion`]
    pub fn frame(&self, data: &mut D) {
        let mut inner = self.inner.lock().unwrap();
        let seat = self.get_seat(data);
        inner.with_grab(data, &seat, |data, handle, grab| {
            grab.frame(data, handle);
        });
    }

    /// Notify that the touch session has been cancelled.
    ///
    /// Use in case you decide the touch stream is a global gesture.
    /// This will remove all current focus targets, and no further events will be sent
    /// until a new touch point appears.
    pub fn cancel(&self, data: &mut D) {
        let mut inner = self.inner.lock().unwrap();
        let seat = self.get_seat(data);
        inner.with_grab(data, &seat, |data, handle, grab| {
            grab.cancel(data, handle);
        });
    }

    /// Tear the whole touch stream down, unconditionally.
    ///
    /// Unlike [`TouchHandle::cancel`], this does not depend on a pending frame: it cancels every
    /// slot that still holds a focus, discharges any frame owed to a slot that saw `up` without a
    /// following `frame`, drops all slot state, and finally unsets any active grab (running its
    /// `unset` hook).
    ///
    /// [`TouchHandle::cancel`] is the right call when the compositor decides an *ongoing* touch
    /// sequence is really a global gesture: it rides the pending frame and is a no-op once the
    /// frame has been delivered. This is the right call when the touch stream ends out of band and
    /// no `up` or `cancel` will ever arrive for the live points, e.g. the touch device is unplugged,
    /// the session is paused or locked, or an input-emulation client disconnects. In those cases
    /// there is usually no pending frame, so `cancel` would silently do nothing and leave the
    /// stored per-slot focus in place, delivering later motion to a client that should no longer
    /// be receiving it.
    ///
    /// One `cancel` is sent per underlying target, deduplicated through [`TouchTarget::last_frame`].
    ///
    /// As with [`TouchHandle::unset_grab`], the internal lock is held while the grab's `unset` hook
    /// runs, so that hook must not call back into this handle.
    pub fn cancel_all(&self, data: &mut D) {
        let mut inner = self.inner.lock().unwrap();
        let seat = self.get_seat(data);
        inner.cancel_all(data, &seat);
    }

    /// Forget the last event actually delivered for a touch slot.
    ///
    /// Compositor-side grabs can consume an `up` or a reused-slot `down` without reaching
    /// [`TouchInternal`]. Call this when the compositor drops the slot from its own live-slot
    /// bookkeeping so a later event cannot be compared against the previous finger's delivery.
    pub fn forget_slot_delivery(&self, slot: TouchSlot) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(state) = inner.focus.get_mut(&slot) {
            state.delivered = None;
        }
    }

    /// Whether one or more attempted touch events still owe a frame.
    pub fn has_pending_frame(&self) -> bool {
        self.inner.lock().unwrap().pending_frame.is_some()
    }

    /// Notify that a touch point has changed its shape.
    pub fn shape(&self, data: &mut D, event: &ShapeEvent) {
        let mut inner = self.inner.lock().unwrap();
        let seat = self.get_seat(data);
        inner.with_grab(data, &seat, |data, handle, grab| {
            grab.shape(data, handle, event);
        });
    }

    /// Notify that a touch point has changed its orientation.
    pub fn orientation(&self, data: &mut D, event: &OrientationEvent) {
        let mut inner = self.inner.lock().unwrap();
        let seat = self.get_seat(data);
        inner.with_grab(data, &seat, |data, handle, grab| {
            grab.orientation(data, handle, event);
        });
    }

    /// Recompute the origin each live touch slot has its delivered coordinates pinned to.
    ///
    /// [`TouchInternal::motion`] ignores the focus it is handed and delivers
    /// `location - loc` using the origin stored for that slot at [`Self::down`]. If a
    /// compositor's surface origins depend on where the event happened (a zoomed or
    /// scrolled view), an origin baked at the down is wrong once the view moves or the
    /// finger travels, so it has to be recomputed at the slot's current location.
    ///
    /// `f` is handed each live slot, its pinned target, the currently stored origin, and
    /// the target plus surface-local point of the last event this slot actually delivered.
    /// It returns the target's new origin at that slot's location, or [`None`] to leave the
    /// stored origin alone. Slot-specific by construction: two fingers on the same target
    /// under a zoomed view resolve different origins.
    ///
    /// Call this immediately before dispatching. It takes the inner mutex briefly and
    /// dispatches nothing.
    pub fn with_slot_origins<F>(&self, mut f: F)
    where
        F: FnMut(
            TouchSlot,
            &<D as SeatHandler>::TouchFocus,
            Point<f64, Logical>,
            Option<&(<D as SeatHandler>::TouchFocus, Point<f64, Logical>)>,
        ) -> Option<Point<f64, Logical>>,
    {
        let mut inner = self.inner.lock().unwrap();
        for (slot, state) in inner.focus.iter_mut() {
            let TouchSlotState { focus, delivered, .. } = state;
            if let Some((target, origin)) = focus.as_mut() {
                if let Some(new_origin) = f(*slot, target, *origin, delivered.as_ref()) {
                    *origin = new_origin;
                }
            }
        }
    }

    /// Recompute the origin the active grab seeds *new* touch points from.
    ///
    /// [`TouchDownGrab`] hands `start_data.focus` to every later down instead of the focus
    /// resolved for that point, and [`TouchInternal::down`] stores it as that slot's origin.
    /// So a second finger is delivered against the first finger's origin, which under a
    /// zoomed or scrolled view is wrong by the distance between them. Recompute it at the
    /// new point's location before dispatching the down; [`Self::with_slot_origins`] cannot
    /// reach this, because the slot does not exist yet.
    ///
    /// `f` is handed the pinned target and returns its origin at the down's location, or
    /// [`None`] to leave the stored origin alone. Takes the inner mutex briefly and
    /// dispatches nothing.
    pub fn with_grab_origin<F>(&self, f: F)
    where
        F: FnOnce(&<D as SeatHandler>::TouchFocus) -> Option<Point<f64, Logical>>,
    {
        let mut inner = self.inner.lock().unwrap();
        if let GrabStatus::Active(_, handler) = &mut inner.grab {
            let start_data = handler.start_data_mut();
            if let Some((target, origin)) = start_data.focus.as_mut() {
                if let Some(new_origin) = f(target) {
                    *origin = new_origin;
                }
            }
        }
    }

    fn get_seat(&self, data: &mut D) -> Seat<D> {
        let seat_state = data.seat_state();
        seat_state
            .seats
            .iter()
            .find(|seat| seat.get_touch().map(|h| &h == self).unwrap_or(false))
            .cloned()
            .unwrap()
    }
}

/// This inner handle is accessed from inside a pointer grab logic, and directly
/// sends event to the client
pub struct TouchInnerHandle<'a, D: SeatHandler> {
    inner: &'a mut TouchInternal<D>,
    seat: &'a Seat<D>,
}

impl<D: SeatHandler> fmt::Debug for TouchInnerHandle<'_, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TouchInnerHandle")
            .field("inner", &self.inner)
            .field("seat", &self.seat.arc.name)
            .finish()
    }
}

impl<D: SeatHandler + 'static> TouchInnerHandle<'_, D> {
    /// Change the current grab on this pointer to the provided grab
    ///
    /// Overwrites any current grab.
    pub fn set_grab<G: TouchGrab<D> + 'static>(
        &mut self,
        handler: &mut dyn TouchGrab<D>,
        data: &mut D,
        serial: Serial,
        grab: G,
    ) {
        handler.unset(data);
        self.inner.set_grab(data, self.seat, serial, grab);
    }

    /// Remove any current grab on this pointer, resetting it to the default behavior
    ///
    /// This will also restore the focus of the underlying pointer if restore_focus
    /// is [`true`]
    pub fn unset_grab(&mut self, handler: &mut dyn TouchGrab<D>, data: &mut D) {
        handler.unset(data);
        self.inner.unset_grab(data, self.seat);
    }

    /// Notify that a new touch point appeared
    ///
    /// You provide the location of the touch, in the form of:
    ///
    /// - The coordinates of the touch in the global compositor space
    /// - The surface on top of which the touch point is, and the coordinates of its
    ///   origin in the global compositor space (or `None` of the touch is not
    ///   on top of a client surface).
    pub fn down(
        &mut self,
        data: &mut D,
        focus: Option<(<D as SeatHandler>::TouchFocus, Point<f64, Logical>)>,
        event: &DownEvent,
    ) {
        self.inner.down(data, self.seat, focus, event)
    }

    /// Notify that a touch point disappeared
    pub fn up(&mut self, data: &mut D, event: &UpEvent) {
        self.inner.up(data, self.seat, event)
    }

    /// Notify that a touch point has changed coordinates.
    ///
    /// You provide the location of the touch, in the form of:
    ///
    /// - The coordinates of the touch in the global compositor space
    /// - The surface on top of which the touch point is, and the coordinates of its
    ///   origin in the global compositor space (or `None` of the touch is not
    ///   on top of a client surface).
    ///
    /// **Note** that this will **not** update the focus of the touch point, the focus
    /// is only set on [`TouchHandle::down`]. The focus provided to this function
    /// can be used to find DnD targets during touch motion.
    pub fn motion(
        &mut self,
        data: &mut D,
        focus: Option<(<D as SeatHandler>::TouchFocus, Point<f64, Logical>)>,
        event: &MotionEvent,
    ) {
        self.inner.motion(data, self.seat, focus, event)
    }

    /// Notify about the end of a set of events that logically belong together.
    ///
    /// This needs to be called after one or move calls to [`TouchHandle::down`] or [`TouchHandle::motion`]
    pub fn frame(&mut self, data: &mut D) {
        self.inner.frame(data, self.seat)
    }

    /// Notify that a touch point has changed its shape.
    pub fn shape(&mut self, data: &mut D, event: &ShapeEvent) {
        self.inner.shape(data, self.seat, event)
    }

    /// Notify that a touch point has changed its orientation.
    pub fn orientation(&mut self, data: &mut D, event: &OrientationEvent) {
        self.inner.orientation(data, self.seat, event)
    }

    /// Notify that the touch session has been cancelled.
    ///
    /// Use in case you decide the touch stream is a global gesture.
    /// This will remove all current focus targets, and no further events will be sent
    /// until a new touch point appears.
    pub fn cancel(&mut self, data: &mut D) {
        self.inner.cancel(data, self.seat)
    }
}

impl<D: SeatHandler + 'static> TouchInternal<D> {
    fn new<F>(default_grab: F) -> Self
    where
        F: Fn() -> Box<dyn TouchGrab<D>> + Send + 'static,
    {
        Self {
            focus: Default::default(),
            pending_frame: None,
            default_grab: Box::new(default_grab),
            grab: GrabStatus::None,
        }
    }

    fn set_grab<G: TouchGrab<D> + 'static>(
        &mut self,
        data: &mut D,
        _seat: &Seat<D>,
        serial: Serial,
        grab: G,
    ) {
        if let GrabStatus::Active(_, handler) = &mut self.grab {
            handler.unset(data);
        }
        self.grab = GrabStatus::Active(serial, Box::new(grab));
    }

    fn unset_grab(&mut self, data: &mut D, _seat: &Seat<D>) {
        if let GrabStatus::Active(_, handler) = &mut self.grab {
            handler.unset(data);
        }
        self.grab = GrabStatus::None;
    }

    fn down(
        &mut self,
        data: &mut D,
        seat: &Seat<D>,
        focus: Option<(<D as SeatHandler>::TouchFocus, Point<f64, Logical>)>,
        event: &DownEvent,
    ) {
        let marker = self.frame_marker();
        self.focus
            .entry(event.slot)
            .and_modify(|state| {
                state.pending = marker;
                state.frame_pending = None;
                state.focus.clone_from(&focus);
                state.delivered = None;
            })
            .or_insert_with(|| TouchSlotState {
                focus,
                delivered: None,
                frame_pending: None,
                pending: marker,
                current: None,
            });

        let state = self.focus.get_mut(&event.slot).unwrap();
        if let Some((focus, loc)) = state.focus.as_ref() {
            let mut new_event = event.clone();
            new_event.location -= *loc;
            let delivered = (focus.clone(), new_event.location);
            focus.down(seat, data, &new_event);
            state.delivered = Some(delivered);
        }
    }

    fn up(&mut self, data: &mut D, seat: &Seat<D>, event: &UpEvent) {
        let marker = self.frame_marker();
        let Some(state) = self.focus.get_mut(&event.slot) else {
            return;
        };
        state.pending = marker;
        if let Some((focus, _)) = state.focus.take() {
            state.delivered = None;
            focus.up(seat, data, event);

            // Keep the focus around to be able to send a frame event after up, but move
            // it out of the current focus to prevent sending other events.
            state.frame_pending = Some(focus);
        }
    }

    fn motion(
        &mut self,
        data: &mut D,
        seat: &Seat<D>,
        _focus: Option<(<D as SeatHandler>::TouchFocus, Point<f64, Logical>)>,
        event: &MotionEvent,
    ) {
        let marker = self.frame_marker();
        let Some(state) = self.focus.get_mut(&event.slot) else {
            return;
        };
        state.pending = marker;
        if let Some((focus, loc)) = state.focus.as_ref() {
            let mut new_event = event.clone();
            new_event.location -= *loc;
            let delivered = (focus.clone(), new_event.location);
            focus.motion(seat, data, &new_event);
            state.delivered = Some(delivered);
        }
    }

    fn frame(&mut self, data: &mut D, seat: &Seat<D>) {
        let Some(marker) = self.pending_frame.take() else {
            tracing::warn!("frame called without prior events");
            return;
        };

        for state in self.focus.values_mut() {
            if state.current.map(|c| c == state.pending).unwrap_or(false) {
                continue;
            }
            state.current = Some(marker);

            // Send the frame event for any stored focus in the up handler
            if let Some(focus) = state.frame_pending.take() {
                if focus.last_frame(seat, data) != Some(marker) {
                    focus.frame(seat, data, marker);
                }
            }

            if let Some((focus, _)) = state.focus.as_ref() {
                if focus.last_frame(seat, data) != Some(marker) {
                    focus.frame(seat, data, marker);
                }
            }
        }

        frame_marker::remove(marker.0);
    }

    fn cancel(&mut self, data: &mut D, seat: &Seat<D>) {
        let Some(marker) = self.pending_frame.take() else {
            tracing::warn!("cancel called without prior events");
            return;
        };

        for state in self.focus.values_mut() {
            if state.current.map(|c| c == state.pending).unwrap_or(false) {
                continue;
            }

            state.current = Some(marker);

            if let Some((focus, _)) = state.focus.take() {
                state.delivered = None;
                if focus.last_frame(seat, data) != Some(marker) {
                    focus.cancel(seat, data, marker);
                }
            }
        }

        frame_marker::remove(marker.0);
    }

    fn cancel_all(&mut self, data: &mut D, seat: &Seat<D>) {
        // Reuse the pending frame if there is one, otherwise take a fresh marker: the whole point
        // of this path is that it works after a frame has already been delivered.
        let marker = self.frame_marker();

        // Cancel every slot that still holds a live focus. `cancel` terminates all of the target's
        // active points at once, so `last_frame` keeps it to one per underlying target.
        for state in self.focus.values_mut() {
            if let Some((focus, _)) = state.focus.take() {
                state.delivered = None;
                if focus.last_frame(seat, data) != Some(marker) {
                    focus.cancel(seat, data, marker);
                }
            }
        }

        // A slot that saw `up` without a following `frame` is still owed one. Discharge it, unless
        // the target already took the cancel above, which ends the sequence anyway. This runs as a
        // second pass so that a target with both an up'd slot and a live slot cannot have its
        // cancel suppressed by a frame that the (unordered) slot map happened to reach first.
        for state in self.focus.values_mut() {
            if let Some(focus) = state.frame_pending.take() {
                if focus.last_frame(seat, data) != Some(marker) {
                    focus.frame(seat, data, marker);
                }
            }
        }

        self.focus.clear();
        if let Some(marker) = self.pending_frame.take() {
            frame_marker::remove(marker.0);
        }

        // Last, so the grab's own cleanup still sees a coherent (now empty) touch state.
        self.unset_grab(data, seat);
    }

    fn shape(&mut self, data: &mut D, seat: &Seat<D>, event: &ShapeEvent) {
        let marker = self.frame_marker();

        let Some(state) = self.focus.get_mut(&event.slot) else {
            return;
        };

        state.pending = marker;
        if let Some((focus, _)) = state.focus.as_ref() {
            focus.shape(seat, data, event);
        }
    }

    fn orientation(&mut self, data: &mut D, seat: &Seat<D>, event: &OrientationEvent) {
        let marker = self.frame_marker();

        let Some(state) = self.focus.get_mut(&event.slot) else {
            return;
        };
        state.pending = marker;
        if let Some((focus, _)) = state.focus.as_ref() {
            focus.orientation(seat, data, event);
        }
    }

    fn with_grab<F>(&mut self, data: &mut D, seat: &Seat<D>, f: F)
    where
        F: FnOnce(&mut D, &mut TouchInnerHandle<'_, D>, &mut dyn TouchGrab<D>),
    {
        let mut grab = std::mem::replace(&mut self.grab, GrabStatus::Borrowed);
        match grab {
            GrabStatus::Borrowed => panic!("Accessed a touch grab from within a touch grab access."),
            GrabStatus::Active(_, ref mut handler) => {
                // If this grab is associated with a surface that is no longer alive, discard it
                if let Some((ref focus, _)) = handler.start_data().focus {
                    if !focus.alive() {
                        handler.unset(data);
                        self.grab = GrabStatus::None;
                        let mut default_grab = (self.default_grab)();
                        f(
                            data,
                            &mut TouchInnerHandle { inner: self, seat },
                            &mut *default_grab,
                        );
                        return;
                    }
                }
                f(data, &mut TouchInnerHandle { inner: self, seat }, &mut **handler);
            }
            GrabStatus::None => {
                let mut default_grab = (self.default_grab)();
                f(
                    data,
                    &mut TouchInnerHandle { inner: self, seat },
                    &mut *default_grab,
                );
            }
        }

        if let GrabStatus::Borrowed = self.grab {
            // the grab has not been ended nor replaced, put it back in place
            self.grab = grab;
        }
    }

    fn frame_marker(&mut self) -> FrameMarker {
        if self.pending_frame.is_none() {
            self.pending_frame = Some(FrameMarker(frame_marker::next()));
        };

        self.pending_frame.unwrap()
    }
}
