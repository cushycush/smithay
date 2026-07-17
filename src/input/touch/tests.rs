//! Tests for the origin a touch slot has its delivered coordinates pinned to.
//!
//! `TouchInternal` delivers `event.location - origin`, where the origin is whatever was
//! stored for that slot at `down`. Two paths seed it, and both bake it at one location:
//! `TouchInternal::down` stores what it is handed, and `TouchDownGrab::down` hands it
//! `start_data.focus` rather than the focus resolved for the new point. That is fine when
//! a target's origin is the same everywhere, and wrong as soon as it is not (a compositor
//! with a zoomed or scrolled view resolves a different origin per location).
//!
//! These pin both refresh hooks against real delivery, not against the stored state.

use std::sync::{Arc, Mutex};

use crate::{
    backend::input::TouchSlot,
    input::{
        touch::{
            DownEvent, FrameMarker, GrabStartData, MotionEvent, OrientationEvent, ShapeEvent, TouchGrab,
            TouchInnerHandle, TouchTarget, UpEvent,
        },
        Seat, SeatHandler, SeatState,
    },
    utils::{IsAlive, Logical, Point, Serial, SERIAL_COUNTER},
};

/// What a target actually received, in ITS OWN coordinates. The whole point: assert on the
/// delivery, never on the stored origin, or the test proves nothing about what the client sees.
#[derive(Debug, Clone, PartialEq)]
enum Delivered {
    Down { slot: TouchSlot, location: Point<f64, Logical> },
    Motion { slot: TouchSlot, location: Point<f64, Logical> },
}

#[derive(Debug, Clone, Default)]
struct Recorder(Arc<Mutex<Vec<Delivered>>>);

impl Recorder {
    fn take(&self) -> Vec<Delivered> {
        std::mem::take(&mut *self.0.lock().unwrap())
    }
}

/// A touch target that records what it was handed. `name` gives it identity so two distinct
/// targets compare unequal.
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

impl TouchTarget<State> for Target {
    fn down(&self, _seat: &Seat<State>, _data: &mut State, event: &DownEvent) {
        self.recorder.0.lock().unwrap().push(Delivered::Down {
            slot: event.slot,
            location: event.location,
        });
    }

    fn up(&self, _seat: &Seat<State>, _data: &mut State, _event: &UpEvent) {}

    fn motion(&self, _seat: &Seat<State>, _data: &mut State, event: &MotionEvent) {
        self.recorder.0.lock().unwrap().push(Delivered::Motion {
            slot: event.slot,
            location: event.location,
        });
    }

    fn frame(&self, _seat: &Seat<State>, _data: &mut State, _marker: FrameMarker) {}
    fn cancel(&self, _seat: &Seat<State>, _data: &mut State, _marker: FrameMarker) {}
    fn shape(&self, _seat: &Seat<State>, _data: &mut State, _event: &ShapeEvent) {}
    fn orientation(&self, _seat: &Seat<State>, _data: &mut State, _event: &OrientationEvent) {}

    fn last_frame(&self, _seat: &Seat<State>, _data: &mut State) -> Option<FrameMarker> {
        None
    }
}

/// Keyboard and pointer focus are unused here, but `SeatHandler` demands the types.
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

impl crate::input::pointer::PointerTarget<State> for Target {
    fn enter(&self, _: &Seat<State>, _: &mut State, _: &crate::input::pointer::MotionEvent) {}
    fn motion(&self, _: &Seat<State>, _: &mut State, _: &crate::input::pointer::MotionEvent) {}
    fn relative_motion(&self, _: &Seat<State>, _: &mut State, _: &crate::input::pointer::RelativeMotionEvent) {}
    fn button(&self, _: &Seat<State>, _: &mut State, _: &crate::input::pointer::ButtonEvent) {}
    fn axis(&self, _: &Seat<State>, _: &mut State, _: crate::input::pointer::AxisFrame) {}
    fn frame(&self, _: &Seat<State>, _: &mut State) {}
    fn gesture_swipe_begin(&self, _: &Seat<State>, _: &mut State, _: &crate::input::pointer::GestureSwipeBeginEvent) {}
    fn gesture_swipe_update(&self, _: &Seat<State>, _: &mut State, _: &crate::input::pointer::GestureSwipeUpdateEvent) {
    }
    fn gesture_swipe_end(&self, _: &Seat<State>, _: &mut State, _: &crate::input::pointer::GestureSwipeEndEvent) {}
    fn gesture_pinch_begin(&self, _: &Seat<State>, _: &mut State, _: &crate::input::pointer::GesturePinchBeginEvent) {}
    fn gesture_pinch_update(&self, _: &Seat<State>, _: &mut State, _: &crate::input::pointer::GesturePinchUpdateEvent) {
    }
    fn gesture_pinch_end(&self, _: &Seat<State>, _: &mut State, _: &crate::input::pointer::GesturePinchEndEvent) {}
    fn gesture_hold_begin(&self, _: &Seat<State>, _: &mut State, _: &crate::input::pointer::GestureHoldBeginEvent) {}
    fn gesture_hold_end(&self, _: &Seat<State>, _: &mut State, _: &crate::input::pointer::GestureHoldEndEvent) {}
    fn leave(&self, _: &Seat<State>, _: &mut State, _: Serial, _: u32) {}
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

/// A grab that pins every touch point to the focus it started with, exactly as
/// `TouchDownGrab` does. Reimplemented here rather than reused so the test states the
/// behavior under test instead of inheriting it.
struct PinningGrab {
    start_data: GrabStartData<State>,
}

impl TouchGrab<State> for PinningGrab {
    fn down(
        &mut self,
        data: &mut State,
        handle: &mut TouchInnerHandle<'_, State>,
        _focus: Option<(Target, Point<f64, Logical>)>,
        event: &DownEvent,
    ) {
        // The bug's origin: the resolved focus is dropped, start_data's is used instead.
        handle.down(data, self.start_data.focus.clone(), event);
    }

    fn up(&mut self, data: &mut State, handle: &mut TouchInnerHandle<'_, State>, event: &UpEvent) {
        handle.up(data, event);
    }

    fn motion(
        &mut self,
        data: &mut State,
        handle: &mut TouchInnerHandle<'_, State>,
        _focus: Option<(Target, Point<f64, Logical>)>,
        event: &MotionEvent,
    ) {
        handle.motion(data, self.start_data.focus.clone(), event);
    }

    fn frame(&mut self, data: &mut State, handle: &mut TouchInnerHandle<'_, State>) {
        handle.frame(data);
    }

    fn cancel(&mut self, data: &mut State, handle: &mut TouchInnerHandle<'_, State>) {
        handle.cancel(data);
    }

    fn shape(&mut self, data: &mut State, handle: &mut TouchInnerHandle<'_, State>, event: &ShapeEvent) {
        handle.shape(data, event);
    }

    fn orientation(&mut self, data: &mut State, handle: &mut TouchInnerHandle<'_, State>, event: &OrientationEvent) {
        handle.orientation(data, event);
    }

    fn start_data(&self) -> &GrabStartData<State> {
        &self.start_data
    }

    fn start_data_mut(&mut self) -> &mut GrabStartData<State> {
        &mut self.start_data
    }

    fn unset(&mut self, _data: &mut State) {}
}

fn down_event(slot: u32, location: Point<f64, Logical>) -> DownEvent {
    DownEvent {
        slot: Some(slot).into(),
        location,
        serial: SERIAL_COUNTER.next_serial(),
        time: 0,
    }
}

fn motion_event(slot: u32, location: Point<f64, Logical>) -> MotionEvent {
    MotionEvent {
        slot: Some(slot).into(),
        location,
        time: 0,
    }
}

/// A stand-in for a view whose origins depend on where you look: a target's origin at
/// `location` is `base + location * k`. Any `k != 0` means an origin baked at one location
/// is wrong at another, which is the entire class of bug under test (Drift's real case is
/// a zoomed camera, where the error is `delta * (1 - 1/zoom)`).
fn origin_at(base: Point<f64, Logical>, location: Point<f64, Logical>, k: f64) -> Point<f64, Logical> {
    (base.x + location.x * k, base.y + location.y * k).into()
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

/// The blocker this whole fork commit exists for: a second finger landing during a grab is
/// seeded from `start_data.focus`, so without a refresh at the new point's location it is
/// delivered against the FIRST finger's origin. Fails without `with_grab_origin`.
#[test]
fn second_touch_down_uses_its_own_origin_not_the_first_fingers() {
    let (mut state, mut seat, recorder, target) = setup();
    let touch = seat.add_touch();

    let base: Point<f64, Logical> = (0.0, 0.0).into();
    let k = 0.5;

    // Finger 1 lands at (100, 100). Its origin is resolved there.
    let first_location: Point<f64, Logical> = (100.0, 100.0).into();
    let first_origin = origin_at(base, first_location, k);
    touch.set_grab(
        &mut state,
        PinningGrab {
            start_data: GrabStartData {
                focus: Some((target.clone(), first_origin)),
                slot: Some(0).into(),
                location: first_location,
            },
        },
        SERIAL_COUNTER.next_serial(),
    );
    touch.down(&mut state, Some((target.clone(), first_origin)), &down_event(0, first_location));
    touch.frame(&mut state);
    assert_eq!(
        recorder.take(),
        vec![Delivered::Down {
            slot: Some(0).into(),
            location: first_location - first_origin,
        }],
        "finger 1 is delivered against its own origin"
    );

    // Finger 2 lands somewhere else. Refresh the grab's pinned origin at ITS location first.
    let second_location: Point<f64, Logical> = (300.0, 200.0).into();
    let second_origin = origin_at(base, second_location, k);
    assert_ne!(first_origin, second_origin, "the test is vacuous if the origins match");

    touch.with_grab_origin(|t| {
        assert_eq!(t, &target);
        Some(second_origin)
    });
    touch.down(&mut state, Some((target.clone(), second_origin)), &down_event(1, second_location));
    touch.frame(&mut state);

    assert_eq!(
        recorder.take(),
        vec![Delivered::Down {
            slot: Some(1).into(),
            location: second_location - second_origin,
        }],
        "finger 2 is delivered against the origin resolved at ITS OWN location, \
         not the one baked when finger 1 pressed"
    );
}

/// Two fingers on the same target resolve different origins, so the refresh has to be
/// per-slot. A single shared origin would make one of these wrong.
#[test]
fn slot_origins_refresh_independently_per_slot() {
    let (mut state, mut seat, recorder, target) = setup();
    let touch = seat.add_touch();

    let base: Point<f64, Logical> = (10.0, 10.0).into();
    let k = 0.5;

    let a_down: Point<f64, Logical> = (100.0, 100.0).into();
    let b_down: Point<f64, Logical> = (400.0, 300.0).into();
    touch.down(&mut state, Some((target.clone(), origin_at(base, a_down, k))), &down_event(0, a_down));
    touch.down(&mut state, Some((target.clone(), origin_at(base, b_down, k))), &down_event(1, b_down));
    touch.frame(&mut state);
    let _ = recorder.take();

    // Both fingers move. Each slot's origin must be recomputed at that slot's new location.
    let a_moved: Point<f64, Logical> = (150.0, 120.0).into();
    let b_moved: Point<f64, Logical> = (380.0, 340.0).into();
    let locations = [(TouchSlot::from(Some(0)), a_moved), (TouchSlot::from(Some(1)), b_moved)];

    touch.with_slot_origins(|slot, t| {
        assert_eq!(t, &target);
        let location = locations.iter().find(|(s, _)| *s == slot).map(|(_, l)| *l).unwrap();
        Some(origin_at(base, location, k))
    });

    touch.motion(&mut state, Some((target.clone(), origin_at(base, a_moved, k))), &motion_event(0, a_moved));
    touch.motion(&mut state, Some((target.clone(), origin_at(base, b_moved, k))), &motion_event(1, b_moved));
    touch.frame(&mut state);

    let a_origin = origin_at(base, a_moved, k);
    let b_origin = origin_at(base, b_moved, k);
    assert_ne!(a_origin, b_origin, "the test is vacuous if both slots share an origin");
    assert_eq!(
        recorder.take(),
        vec![
            Delivered::Motion {
                slot: Some(0).into(),
                location: a_moved - a_origin,
            },
            Delivered::Motion {
                slot: Some(1).into(),
                location: b_moved - b_origin,
            },
        ],
        "each slot is delivered against the origin resolved at its own location"
    );
}

/// Returning `None` leaves the stored origin alone, so a compositor that cannot resolve a
/// target keeps delivering against the last good origin rather than against garbage.
#[test]
fn returning_none_preserves_the_stored_origin() {
    let (mut state, mut seat, recorder, target) = setup();
    let touch = seat.add_touch();

    let origin: Point<f64, Logical> = (25.0, 25.0).into();
    let location: Point<f64, Logical> = (100.0, 100.0).into();
    touch.down(&mut state, Some((target.clone(), origin)), &down_event(0, location));
    touch.frame(&mut state);
    let _ = recorder.take();

    let mut saw_target = false;
    touch.with_slot_origins(|_, _| {
        saw_target = true;
        None
    });
    assert!(saw_target, "the callback still sees the live slot");

    let moved: Point<f64, Logical> = (140.0, 130.0).into();
    touch.motion(&mut state, Some((target.clone(), origin)), &motion_event(0, moved));
    touch.frame(&mut state);

    assert_eq!(
        recorder.take(),
        vec![Delivered::Motion {
            slot: Some(0).into(),
            location: moved - origin,
        }],
        "the origin is untouched when the callback declines to resolve one"
    );
}

/// With no grab there is no pinned origin to refresh, and the hook must not panic reaching
/// for one (`DefaultGrab`'s `start_data_mut` is `unreachable!()`).
#[test]
fn grab_origin_refresh_is_a_noop_without_a_grab() {
    let (mut state, mut seat, _recorder, target) = setup();
    let touch = seat.add_touch();

    let mut called = false;
    touch.with_grab_origin(|_| {
        called = true;
        None
    });
    assert!(!called, "no grab means no pinned origin, so the callback never runs");

    // And the handle is still usable afterwards.
    let location: Point<f64, Logical> = (10.0, 10.0).into();
    touch.down(&mut state, Some((target, (0.0, 0.0).into())), &down_event(0, location));
    touch.frame(&mut state);
}
