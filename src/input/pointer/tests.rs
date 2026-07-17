//! Tests for the origin a pointer grab has its delivered coordinates pinned to.
//!
//! [`ClickGrab`] ignores the focus handed to `motion` and delivers `location - origin`
//! against the origin baked in `start_data.focus` when the button went down. That is fine
//! when a target's origin is the same everywhere, and wrong as soon as it is not (a
//! compositor with a zoomed or scrolled view resolves a different origin per location).

use std::sync::{Arc, Mutex};

use crate::{
    input::{
        pointer::{
            AxisFrame, ButtonEvent, GestureHoldBeginEvent, GestureHoldEndEvent, GesturePinchBeginEvent,
            GesturePinchEndEvent, GesturePinchUpdateEvent, GestureSwipeBeginEvent, GestureSwipeEndEvent,
            GestureSwipeUpdateEvent, MotionEvent, PointerTarget, RelativeMotionEvent,
        },
        Seat, SeatHandler, SeatState,
    },
    utils::{IsAlive, Logical, Point, Serial, SERIAL_COUNTER},
};

/// What the target actually received, in ITS OWN coordinates. Assert on the delivery, never
/// on the stored origin: the stored origin is not what the client sees.
#[derive(Debug, Clone, Default)]
struct Recorder(Arc<Mutex<Vec<Point<f64, Logical>>>>);

impl Recorder {
    fn take(&self) -> Vec<Point<f64, Logical>> {
        std::mem::take(&mut *self.0.lock().unwrap())
    }
}

#[derive(Debug, Clone)]
struct Target {
    name: &'static str,
    recorder: Recorder,
}

impl PartialEq for Target {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl IsAlive for Target {
    fn alive(&self) -> bool {
        true
    }
}

impl PointerTarget<State> for Target {
    fn enter(&self, _: &Seat<State>, _: &mut State, event: &MotionEvent) {
        self.recorder.0.lock().unwrap().push(event.location);
    }
    fn motion(&self, _: &Seat<State>, _: &mut State, event: &MotionEvent) {
        self.recorder.0.lock().unwrap().push(event.location);
    }
    fn relative_motion(&self, _: &Seat<State>, _: &mut State, _: &RelativeMotionEvent) {}
    fn button(&self, _: &Seat<State>, _: &mut State, _: &ButtonEvent) {}
    fn axis(&self, _: &Seat<State>, _: &mut State, _: AxisFrame) {}
    fn frame(&self, _: &Seat<State>, _: &mut State) {}
    fn gesture_swipe_begin(&self, _: &Seat<State>, _: &mut State, _: &GestureSwipeBeginEvent) {}
    fn gesture_swipe_update(&self, _: &Seat<State>, _: &mut State, _: &GestureSwipeUpdateEvent) {}
    fn gesture_swipe_end(&self, _: &Seat<State>, _: &mut State, _: &GestureSwipeEndEvent) {}
    fn gesture_pinch_begin(&self, _: &Seat<State>, _: &mut State, _: &GesturePinchBeginEvent) {}
    fn gesture_pinch_update(&self, _: &Seat<State>, _: &mut State, _: &GesturePinchUpdateEvent) {}
    fn gesture_pinch_end(&self, _: &Seat<State>, _: &mut State, _: &GesturePinchEndEvent) {}
    fn gesture_hold_begin(&self, _: &Seat<State>, _: &mut State, _: &GestureHoldBeginEvent) {}
    fn gesture_hold_end(&self, _: &Seat<State>, _: &mut State, _: &GestureHoldEndEvent) {}
    fn leave(&self, _: &Seat<State>, _: &mut State, _: Serial, _: u32) {}
}

impl crate::input::keyboard::KeyboardTarget<State> for Target {
    fn enter(&self, _: &Seat<State>, _: &mut State, _: Vec<crate::input::keyboard::KeysymHandle<'_>>, _: Serial) {}
    fn leave(&self, _: &Seat<State>, _: &mut State, _: Serial) {}
    fn key(
        &self,
        _: &Seat<State>,
        _: &mut State,
        _: crate::input::keyboard::KeysymHandle<'_>,
        _: crate::backend::input::KeyState,
        _: Serial,
        _: u32,
    ) {
    }
    fn modifiers(&self, _: &Seat<State>, _: &mut State, _: crate::input::keyboard::ModifiersState, _: Serial) {}
}

impl crate::input::touch::TouchTarget<State> for Target {
    fn down(&self, _: &Seat<State>, _: &mut State, _: &crate::input::touch::DownEvent) {}
    fn up(&self, _: &Seat<State>, _: &mut State, _: &crate::input::touch::UpEvent) {}
    fn motion(&self, _: &Seat<State>, _: &mut State, _: &crate::input::touch::MotionEvent) {}
    fn frame(&self, _: &Seat<State>, _: &mut State, _: crate::input::touch::FrameMarker) {}
    fn cancel(&self, _: &Seat<State>, _: &mut State, _: crate::input::touch::FrameMarker) {}
    fn shape(&self, _: &Seat<State>, _: &mut State, _: &crate::input::touch::ShapeEvent) {}
    fn orientation(&self, _: &Seat<State>, _: &mut State, _: &crate::input::touch::OrientationEvent) {}
    fn last_frame(&self, _: &Seat<State>, _: &mut State) -> Option<crate::input::touch::FrameMarker> {
        None
    }
}

struct State {
    seat_state: SeatState<State>,
}

impl SeatHandler for State {
    type KeyboardFocus = Target;
    type PointerFocus = Target;
    type TouchFocus = Target;

    fn seat_state(&mut self) -> &mut SeatState<State> {
        &mut self.seat_state
    }
}

/// A stand-in for a view whose origins depend on where you look: origin at `location` is
/// `base + location * k`. Any `k != 0` means an origin baked at one location is wrong at
/// another, which is the class of bug under test.
fn origin_at(base: Point<f64, Logical>, location: Point<f64, Logical>, k: f64) -> Point<f64, Logical> {
    (base.x + location.x * k, base.y + location.y * k).into()
}

fn motion_event(location: Point<f64, Logical>) -> MotionEvent {
    MotionEvent {
        location,
        serial: SERIAL_COUNTER.next_serial(),
        time: 0,
    }
}

/// Press the left button, which makes `DefaultGrab` install a `ClickGrab` pinning the
/// currently focused target and its origin.
fn press_button(pointer: &crate::input::pointer::PointerHandle<State>, state: &mut State) {
    pointer.button(
        state,
        &ButtonEvent {
            serial: SERIAL_COUNTER.next_serial(),
            time: 0,
            button: 0x110,
            state: crate::backend::input::ButtonState::Pressed,
        },
    );
}

fn setup() -> (State, Seat<State>, Recorder, Target) {
    let mut seat_state = SeatState::new();
    let seat = seat_state.new_seat("test");
    let state = State { seat_state };
    let recorder = Recorder::default();
    let target = Target {
        name: "target",
        recorder: recorder.clone(),
    };
    (state, seat, recorder, target)
}

/// A click-hold delivers against the origin baked at the press. Refreshing it at the
/// location about to be delivered is what keeps the coordinates live. Fails without
/// `with_grab_origin`: the delivery comes out scaled by the view instead of 1:1.
#[test]
fn click_grab_delivers_against_the_origin_refreshed_at_the_delivered_location() {
    let (mut state, mut seat, recorder, target) = setup();
    let pointer = seat.add_pointer();

    let base: Point<f64, Logical> = (0.0, 0.0).into();
    let k = 0.5;

    let press: Point<f64, Logical> = (100.0, 100.0).into();
    let press_origin = origin_at(base, press, k);
    pointer.motion(&mut state, Some((target.clone(), press_origin)), &motion_event(press));
    // Drive the real path rather than constructing a ClickGrab: DefaultGrab installs one on
    // any press, pinning whatever focus is current, which is exactly the production shape.
    press_button(&pointer, &mut state);
    let _ = recorder.take();

    // The pointer moves while held. The grab ignores the focus argument, so only a refresh
    // of its pinned origin can keep the delivery honest.
    let moved: Point<f64, Logical> = (300.0, 240.0).into();
    let moved_origin = origin_at(base, moved, k);
    assert_ne!(press_origin, moved_origin, "the test is vacuous if the origins match");

    pointer.with_grab_origin(|t| {
        assert_eq!(t, &target);
        Some(moved_origin)
    });
    pointer.motion(&mut state, Some((target.clone(), moved_origin)), &motion_event(moved));

    assert_eq!(
        recorder.take(),
        vec![moved - moved_origin],
        "the held pointer is delivered against the origin resolved at the location it is \
         actually being delivered at, not the one baked at the press"
    );
}

/// Returning `None` leaves the stored origin alone, so a compositor that cannot resolve the
/// pinned target keeps the last good origin rather than garbage.
#[test]
fn returning_none_preserves_the_stored_origin() {
    let (mut state, mut seat, recorder, target) = setup();
    let pointer = seat.add_pointer();

    let press: Point<f64, Logical> = (100.0, 100.0).into();
    let origin: Point<f64, Logical> = (40.0, 40.0).into();
    pointer.motion(&mut state, Some((target.clone(), origin)), &motion_event(press));
    press_button(&pointer, &mut state);
    let _ = recorder.take();

    let mut saw_target = false;
    pointer.with_grab_origin(|_| {
        saw_target = true;
        None
    });
    assert!(saw_target, "the callback still sees the pinned target");

    let moved: Point<f64, Logical> = (160.0, 150.0).into();
    pointer.motion(&mut state, Some((target.clone(), origin)), &motion_event(moved));

    assert_eq!(
        recorder.take(),
        vec![moved - origin],
        "the origin is untouched when the callback declines to resolve one"
    );
}

/// `DefaultGrab`'s `start_data_mut` is `unreachable!()`, so the hook must not reach for it
/// when no grab is active.
#[test]
fn grab_origin_refresh_is_a_noop_without_a_grab() {
    let (mut state, mut seat, _recorder, target) = setup();
    let pointer = seat.add_pointer();

    let mut called = false;
    pointer.with_grab_origin(|_| {
        called = true;
        None
    });
    assert!(!called, "no grab means no pinned origin, so the callback never runs");

    // And the handle is still usable afterwards.
    pointer.motion(
        &mut state,
        Some((target, (0.0, 0.0).into())),
        &motion_event((10.0, 10.0).into()),
    );
}
