//! Tests for the EI backend.
//!
//! These drive a real `reis` client against [`EiInput`] over a `socketpair`, so the EI
//! handshake, the device lifecycle and the request conversion are all exercised end to end
//! rather than mocked.
//!
//! Both ends are pumped from the test thread. reis puts its sockets in non-blocking mode, so
//! neither side can block the other, and every write lands in the peer's buffer before the
//! call returns. That makes a plain alternation of client and server pumps converge without
//! any sleeping or any second thread.

use std::{collections::HashMap, io, os::unix::net::UnixStream, time::Duration};

use calloop::EventLoop;
use reis::{
    PendingRequestResult, ei, eis,
    handshake::EiHandshaker,
    request::{self, DeviceCapability},
};

use super::{EiInput, EiInputConnection, EiInputEvent, EiInputSeat, EiRegion, EiSpecialEvent};
use crate::backend::input::InputEvent;

const CLIENT_NAME: &str = "smithay-libei-test";

// A pump round is client-read-then-server-dispatch. Rounds are cheap and side-effect free once
// both ends are idle, so this only bounds a runaway; the assertions decide the outcome.
const MAX_PUMP_ROUNDS: usize = 64;

/// What the server callback saw, flattened into something comparable.
///
/// Fields sampled at delivery time rather than read back later are the point of several of
/// these tests: `Device::remove` drains the interfaces `has_capability` reads, so a consumer
/// told about a removal afterwards can no longer tell what it lost.
///
/// Payloads a test does not match on are still carried, because the assertions dump the whole
/// recording through `Debug` when they fail and that is what makes a failure readable.
#[allow(dead_code)]
#[derive(Debug)]
enum Rec {
    Connected,
    Disconnected,
    DeviceAdded {
        device: request::Device,
    },
    DeviceRemoved {
        device: request::Device,
        had_touch: bool,
    },
    TouchDown {
        device: request::Device,
        touch_id: u32,
        x: f32,
        y: f32,
        classified_touch: bool,
    },
    TouchFrame {
        device: request::Device,
    },
    PointerMotionAbsolute {
        device: request::Device,
        x: f32,
        y: f32,
    },
    StopEmulating {
        device: request::Device,
    },
    TouchscreenReleased {
        device: request::Device,
    },
    Other,
}

fn record(event: EiInputEvent, seat: Option<&EiInputSeat>) -> Rec {
    // Whether the seat's synchronous accessor agrees this is the touch device, sampled now
    // because the answer changes as devices are re-created and closed.
    let classifies =
        |device: &request::Device| seat.and_then(EiInputSeat::touch_device).as_ref() == Some(device);

    match event {
        EiInputEvent::Connected => Rec::Connected,
        EiInputEvent::Disconnected => Rec::Disconnected,
        EiInputEvent::Event(event) => match event {
            InputEvent::DeviceAdded { device } => Rec::DeviceAdded { device },
            InputEvent::DeviceRemoved { device } => Rec::DeviceRemoved {
                had_touch: device.has_capability(DeviceCapability::Touch),
                device,
            },
            InputEvent::TouchDown { event } => Rec::TouchDown {
                classified_touch: classifies(&event.device),
                device: event.device,
                touch_id: event.touch_id,
                x: event.x,
                y: event.y,
            },
            InputEvent::TouchFrame { event } => Rec::TouchFrame { device: event.device },
            InputEvent::PointerMotionAbsolute { event } => Rec::PointerMotionAbsolute {
                device: event.device,
                x: event.dx_absolute,
                y: event.dy_absolute,
            },
            InputEvent::Special(EiSpecialEvent::StopEmulating(event)) => {
                Rec::StopEmulating { device: event.device }
            }
            InputEvent::Special(EiSpecialEvent::TouchscreenReleased(event)) => {
                Rec::TouchscreenReleased { device: event.device }
            }
            _ => Rec::Other,
        },
    }
}

/// One tag per record, for asserting the exact shape of a short sequence.
///
/// Where a test cares about ordering rather than payloads, comparing tags says so directly and
/// fails with the whole sequence rather than a bare `false`.
fn tags(recs: &[Rec]) -> Vec<&'static str> {
    recs.iter()
        .map(|rec| match rec {
            Rec::Connected => "connected",
            Rec::Disconnected => "disconnected",
            Rec::DeviceAdded { .. } => "device-added",
            Rec::DeviceRemoved { .. } => "device-removed",
            Rec::TouchDown { .. } => "touch-down",
            Rec::TouchFrame { .. } => "touch-frame",
            Rec::PointerMotionAbsolute { .. } => "pointer-motion-absolute",
            Rec::StopEmulating { .. } => "stop-emulating",
            Rec::TouchscreenReleased { .. } => "touchscreen-released",
            Rec::Other => "other",
        })
        .collect()
}

#[derive(Default)]
struct ServerState {
    connection: Option<EiInputConnection>,
    seat: Option<EiInputSeat>,
    events: Vec<Rec>,
}

/// One EI protocol event for a device, kept in arrival order.
///
/// The order is the assertion for regions: the protocol requires every region to arrive
/// before the device's `done`.
#[derive(Debug, PartialEq)]
enum DeviceLog {
    Region {
        offset_x: u32,
        offset_y: u32,
        width: u32,
        height: u32,
        scale: f32,
    },
    Interface(String),
    Done,
    Destroyed,
    Resumed,
    Paused,
}

#[derive(Default)]
struct ClientDevice {
    name: Option<String>,
    interfaces: HashMap<String, reis::Object>,
    log: Vec<DeviceLog>,
    destroyed: bool,
}

#[derive(Default)]
struct ClientSeat {
    capabilities: HashMap<String, u64>,
}

enum Phase {
    Handshake(EiHandshaker<'static>),
    Connected,
}

/// A minimal EI client that binds everything a seat offers and records what it is told.
struct TestClient {
    context: ei::Context,
    phase: Phase,
    seats: HashMap<ei::Seat, ClientSeat>,
    devices: HashMap<ei::Device, ClientDevice>,
    // Creation order, so a test can talk about "the second touch device" after a re-add.
    device_order: Vec<ei::Device>,
    touchscreens_destroyed: usize,
    last_serial: u32,
    sequence: u32,
    events_seen: usize,
    eof: bool,
}

impl TestClient {
    fn new(context: ei::Context) -> Self {
        Self {
            context,
            phase: Phase::Handshake(EiHandshaker::new(CLIENT_NAME, ei::handshake::ContextType::Sender)),
            seats: HashMap::new(),
            devices: HashMap::new(),
            device_order: Vec::new(),
            touchscreens_destroyed: 0,
            last_serial: 0,
            sequence: 0,
            events_seen: 0,
            eof: false,
        }
    }

    fn pump(&mut self) {
        if self.eof {
            return;
        }
        match self.context.read() {
            Ok(_) => {}
            // The server dropped the connection. Nothing more will arrive, but what already
            // parsed is still worth handling below.
            Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => self.eof = true,
            Err(err) => panic!("client read failed: {err}"),
        }

        while let Some(result) = self.context.pending_event() {
            let event = match result {
                PendingRequestResult::Request(event) => event,
                PendingRequestResult::ParseError(err) => panic!("client parse error: {err:?}"),
                PendingRequestResult::InvalidObject(id) => panic!("client saw invalid object {id}"),
            };
            self.events_seen += 1;
            self.handle_event(event);
        }

        let _ = self.context.flush();
    }

    fn handle_event(&mut self, event: ei::Event) {
        if let Phase::Handshake(handshaker) = &mut self.phase {
            match handshaker.handle_event(event) {
                Ok(Some(resp)) => {
                    self.last_serial = resp.serial;
                    self.phase = Phase::Connected;
                }
                Ok(None) => {}
                Err(err) => panic!("client handshake failed: {err}"),
            }
            return;
        }

        match event {
            ei::Event::Connection(_connection, event) => match event {
                ei::connection::Event::Seat { seat } => {
                    self.seats.insert(seat, ClientSeat::default());
                }
                ei::connection::Event::Ping { ping } => ping.done(0),
                _ => {}
            },
            ei::Event::Seat(seat, event) => match event {
                ei::seat::Event::Capability { mask, interface } => {
                    if let Some(data) = self.seats.get_mut(&seat) {
                        data.capabilities.insert(interface, mask);
                    }
                }
                ei::seat::Event::Done => {
                    // Bind everything offered. The server only creates devices it has been
                    // told to add, so the test stays in control of what exists.
                    if let Some(data) = self.seats.get(&seat) {
                        let mask = data.capabilities.values().fold(0, |acc, mask| acc | mask);
                        seat.bind(mask);
                    }
                }
                ei::seat::Event::Device { device } => {
                    self.devices.insert(device.clone(), ClientDevice::default());
                    self.device_order.push(device);
                }
                _ => {}
            },
            ei::Event::Device(device, event) => {
                let Some(data) = self.devices.get_mut(&device) else {
                    return;
                };
                match event {
                    ei::device::Event::Name { name } => data.name = Some(name),
                    ei::device::Event::Region {
                        offset_x,
                        offset_y,
                        width,
                        hight,
                        scale,
                    } => data.log.push(DeviceLog::Region {
                        offset_x,
                        offset_y,
                        width,
                        height: hight,
                        scale,
                    }),
                    ei::device::Event::Interface { object } => {
                        let name = object.interface().to_owned();
                        data.log.push(DeviceLog::Interface(name.clone()));
                        data.interfaces.insert(name, object);
                    }
                    ei::device::Event::Done => data.log.push(DeviceLog::Done),
                    ei::device::Event::Destroyed { serial } => {
                        self.last_serial = serial;
                        data.destroyed = true;
                        data.log.push(DeviceLog::Destroyed);
                    }
                    ei::device::Event::Resumed { serial } => {
                        self.last_serial = serial;
                        data.log.push(DeviceLog::Resumed);
                    }
                    ei::device::Event::Paused { serial } => {
                        self.last_serial = serial;
                        data.log.push(DeviceLog::Paused);
                    }
                    _ => {}
                }
            }
            ei::Event::Touchscreen(_touchscreen, event) => {
                if let ei::touchscreen::Event::Destroyed { serial } = event {
                    self.last_serial = serial;
                    self.touchscreens_destroyed += 1;
                }
            }
            _ => {}
        }
    }

    /// Devices the server announced under `name`, in creation order, including dead ones.
    fn devices_named(&self, name: &str) -> Vec<ei::Device> {
        self.device_order
            .iter()
            .filter(|device| self.devices.get(*device).and_then(|data| data.name.as_deref()) == Some(name))
            .cloned()
            .collect()
    }

    fn log(&self, device: &ei::Device) -> &[DeviceLog] {
        &self.devices[device].log
    }

    fn destroyed(&self, device: &ei::Device) -> bool {
        self.devices[device].destroyed
    }

    fn interface<T: reis::Interface>(&self, device: &ei::Device) -> Option<T> {
        self.devices
            .get(device)?
            .interfaces
            .get(T::NAME)?
            .clone()
            .downcast()
    }

    fn touchscreen(&self, device: &ei::Device) -> ei::Touchscreen {
        self.interface(device)
            .unwrap_or_else(|| panic!("device has no touchscreen interface"))
    }

    fn start_emulating(&mut self, device: &ei::Device) {
        device.start_emulating(self.last_serial, self.sequence);
        self.sequence += 1;
    }

    fn frame(&mut self, device: &ei::Device) {
        self.sequence += 1;
        // A timestamp reis will stamp onto whatever the device has pending.
        device.frame(self.last_serial, u64::from(self.sequence));
    }
}

struct Harness {
    event_loop: EventLoop<'static, ServerState>,
    state: ServerState,
    client: TestClient,
}

impl Harness {
    fn new() -> Self {
        let (server_socket, client_socket) = UnixStream::pair().unwrap();
        let event_loop = EventLoop::try_new().unwrap();
        let source = EiInput::new(eis::Context::new(server_socket).unwrap());

        event_loop
            .handle()
            .insert_source(source, |event, connection, state: &mut ServerState| {
                if state.connection.is_none() {
                    state.connection = Some(connection.clone());
                }
                let rec = record(event, state.seat.as_ref());
                state.events.push(rec);
            })
            .unwrap();

        Self {
            event_loop,
            state: ServerState::default(),
            client: TestClient::new(ei::Context::new(client_socket).unwrap()),
        }
    }

    /// Runs both ends until neither makes further progress.
    fn pump(&mut self) {
        for _ in 0..MAX_PUMP_ROUNDS {
            let before = (self.client.events_seen, self.state.events.len());
            self.client.pump();
            self.event_loop
                .dispatch(Some(Duration::ZERO), &mut self.state)
                .unwrap();
            if (self.client.events_seen, self.state.events.len()) == before {
                return;
            }
        }
        panic!("harness did not settle within {MAX_PUMP_ROUNDS} rounds");
    }

    fn connection(&self) -> EiInputConnection {
        self.state
            .connection
            .clone()
            .expect("client has not connected yet")
    }

    /// Adds a seat and records it, so the server callback can classify devices against it.
    fn add_seat(&mut self, name: &str) -> EiInputSeat {
        let seat = self.connection().add_seat(name);
        self.state.seat = Some(seat.clone());
        self.flush();
        seat
    }

    /// Sends whatever the test queued from outside a callback.
    ///
    /// `process_events` flushes after each request it handles, but a test acting between
    /// pumps is not inside that path.
    fn flush(&self) {
        let _ = self.connection().flush();
    }

    fn recs(&self) -> &[Rec] {
        &self.state.events
    }

    /// Brings a client up to a bound seat, with the devices the test asked for.
    fn connect(&mut self, setup: impl FnOnce(&mut Harness) -> EiInputSeat) -> EiInputSeat {
        self.pump();
        assert!(
            matches!(self.recs(), [Rec::Connected]),
            "expected a lone Connected, got {:?}",
            self.recs()
        );
        let seat = setup(self);
        self.flush();
        self.pump();
        seat
    }
}

fn touch_regions() -> Vec<EiRegion> {
    vec![
        EiRegion {
            offset_x: 0,
            offset_y: 0,
            width: 1920,
            height: 1080,
            scale: 1.0,
        },
        // A second region pins that every region is announced, not just the first.
        EiRegion {
            offset_x: 1920,
            offset_y: 0,
            width: 1280,
            height: 720,
            scale: 2.0,
        },
    ]
}

fn absolute_regions() -> Vec<EiRegion> {
    vec![EiRegion {
        offset_x: 10,
        offset_y: 20,
        width: 640,
        height: 480,
        scale: 1.5,
    }]
}

fn expected_log_regions(regions: &[EiRegion]) -> Vec<DeviceLog> {
    regions
        .iter()
        .map(|region| DeviceLog::Region {
            offset_x: region.offset_x,
            offset_y: region.offset_y,
            width: region.width,
            height: region.height,
            scale: region.scale,
        })
        .collect()
}

/// Asserts the device was told its exact regions, and that they all arrived before `done`.
fn assert_regions_before_done(log: &[DeviceLog], regions: &[EiRegion]) {
    let done = log
        .iter()
        .position(|entry| *entry == DeviceLog::Done)
        .unwrap_or_else(|| panic!("device never got done: {log:?}"));

    let seen: Vec<_> = log[..done]
        .iter()
        .filter(|entry| matches!(entry, DeviceLog::Region { .. }))
        .collect();
    let expected = expected_log_regions(regions);
    let expected: Vec<_> = expected.iter().collect();
    assert_eq!(seen, expected, "regions before done did not match");

    assert!(
        !log[done..]
            .iter()
            .any(|entry| matches!(entry, DeviceLog::Region { .. })),
        "a region arrived after done: {log:?}"
    );
}

#[test]
fn regions_are_announced_before_device_done() {
    let mut harness = Harness::new();
    harness.connect(|harness| {
        let seat = harness.add_seat("seat");
        seat.add_pointer_absolute("abs", absolute_regions());
        seat.add_touch("touch", touch_regions());
        seat
    });

    let abs = harness.client.devices_named("abs");
    let touch = harness.client.devices_named("touch");
    assert_eq!(abs.len(), 1);
    assert_eq!(touch.len(), 1);

    assert_regions_before_done(harness.client.log(&abs[0]), &absolute_regions());
    assert_regions_before_done(harness.client.log(&touch[0]), &touch_regions());
}

#[test]
fn re_adding_a_device_removes_the_old_one_and_re_announces_regions() {
    let mut harness = Harness::new();
    let seat = harness.connect(|harness| {
        let seat = harness.add_seat("seat");
        seat.add_touch("touch", touch_regions());
        seat
    });

    let first = harness.client.devices_named("touch");
    assert_eq!(first.len(), 1);

    seat.add_touch("touch", touch_regions());
    harness.flush();
    harness.pump();

    let devices = harness.client.devices_named("touch");
    assert_eq!(devices.len(), 2, "re-adding should announce a replacement device");
    assert!(
        harness.client.destroyed(&devices[0]),
        "re-adding should destroy the old device"
    );
    assert!(!harness.client.destroyed(&devices[1]));
    assert_regions_before_done(harness.client.log(&devices[1]), &touch_regions());
}

#[test]
fn only_touch_devices_convert_frames() {
    let mut harness = Harness::new();
    harness.connect(|harness| {
        let seat = harness.add_seat("seat");
        seat.add_pointer_absolute("abs", absolute_regions());
        seat.add_touch("touch", touch_regions());
        seat
    });

    let abs = harness.client.devices_named("abs")[0].clone();
    let touch = harness.client.devices_named("touch")[0].clone();

    // reis drops a frame that has nothing pending behind it, so the pointer needs a real
    // operation queued for its frame to reach the backend at all.
    harness.client.start_emulating(&abs);
    let pointer: ei::PointerAbsolute = harness.client.interface(&abs).unwrap();
    pointer.motion_absolute(100.0, 200.0);
    harness.client.frame(&abs);

    harness.client.start_emulating(&touch);
    harness.client.touchscreen(&touch).down(1, 30.0, 40.0);
    harness.client.frame(&touch);
    harness.pump();

    let frames: Vec<_> = harness
        .recs()
        .iter()
        .filter_map(|rec| match rec {
            Rec::TouchFrame { device } => Some(device.clone()),
            _ => None,
        })
        .collect();

    assert_eq!(
        frames.len(),
        1,
        "expected exactly one touch frame: {:?}",
        harness.recs()
    );
    assert!(
        frames[0].has_capability(DeviceCapability::Touch),
        "the converted frame did not come from the touch device"
    );
    assert!(
        harness
            .recs()
            .iter()
            .any(|rec| matches!(rec, Rec::PointerMotionAbsolute { x, y, .. } if *x == 100.0 && *y == 200.0)),
        "the pointer's own operation should still convert: {:?}",
        harness.recs()
    );
}

#[test]
fn stop_emulating_converts_to_a_special_event() {
    let mut harness = Harness::new();
    harness.connect(|harness| {
        let seat = harness.add_seat("seat");
        seat.add_touch("touch", touch_regions());
        seat
    });

    let touch = harness.client.devices_named("touch")[0].clone();
    harness.client.start_emulating(&touch);
    touch.stop_emulating(harness.client.last_serial);
    harness.pump();

    let stops: Vec<_> = harness
        .recs()
        .iter()
        .filter(|rec| matches!(rec, Rec::StopEmulating { .. }))
        .collect();
    assert_eq!(stops.len(), 1, "expected one stop: {:?}", harness.recs());
}

#[test]
fn a_client_protocol_violation_reports_disconnected() {
    let mut harness = Harness::new();
    harness.connect(|harness| {
        let seat = harness.add_seat("seat");
        seat.add_touch("touch", touch_regions());
        seat
    });

    let touch = harness.client.devices_named("touch")[0].clone();
    harness.client.start_emulating(&touch);

    // Two downs with one id is a protocol violation, which is what puts the source on its
    // error path.
    let touchscreen = harness.client.touchscreen(&touch);
    touchscreen.down(1, 10.0, 10.0);
    touchscreen.down(1, 20.0, 20.0);
    harness.pump();

    assert!(
        matches!(harness.recs().last(), Some(Rec::Disconnected)),
        "an errored client should report Disconnected: {:?}",
        harness.recs()
    );
}

#[test]
fn re_creating_the_touch_device_allows_the_same_touch_id_again() {
    let mut harness = Harness::new();
    let seat = harness.connect(|harness| {
        let seat = harness.add_seat("seat");
        seat.add_touch("touch", touch_regions());
        seat
    });

    let first = harness.client.devices_named("touch")[0].clone();
    harness.client.start_emulating(&first);
    harness.client.touchscreen(&first).down(1, 10.0, 10.0);
    harness.client.frame(&first);
    harness.pump();

    first.stop_emulating(harness.client.last_serial);
    harness.pump();
    assert!(
        harness
            .recs()
            .iter()
            .any(|rec| matches!(rec, Rec::StopEmulating { .. })),
        "expected the stop to be delivered: {:?}",
        harness.recs()
    );

    // What a consumer does with a stop: re-create the touch device, which is what resets the
    // per-device touch state reis keeps. Without it the second down below would be a
    // duplicate and would kill the client.
    seat.add_touch("touch", touch_regions());
    harness.flush();
    harness.pump();

    let devices = harness.client.devices_named("touch");
    assert_eq!(devices.len(), 2, "the client should discover the replacement");
    let second = devices[1].clone();
    assert_regions_before_done(harness.client.log(&second), &touch_regions());

    harness.client.start_emulating(&second);
    harness.client.touchscreen(&second).down(1, 20.0, 20.0);
    harness.client.frame(&second);
    harness.pump();

    let downs: Vec<_> = harness
        .recs()
        .iter()
        .filter_map(|rec| match rec {
            Rec::TouchDown { device, touch_id, .. } => Some((device.clone(), *touch_id)),
            _ => None,
        })
        .collect();

    assert_eq!(
        downs.len(),
        2,
        "both downs should be delivered across the re-creation: {:?}",
        harness.recs()
    );
    assert_eq!(downs[0].1, 1);
    assert_eq!(downs[1].1, 1);
    assert_ne!(
        downs[0].0, downs[1].0,
        "the second down should be on a new device"
    );
    assert!(
        !harness.recs().iter().any(|rec| matches!(rec, Rec::Disconnected)),
        "the client should have survived the second down: {:?}",
        harness.recs()
    );
}

#[test]
fn closing_a_device_with_a_live_contact_announces_the_removal_before_draining_it() {
    let mut harness = Harness::new();
    let seat = harness.connect(|harness| {
        let seat = harness.add_seat("seat");
        seat.add_touch("touch", touch_regions());
        seat
    });

    let touch = harness.client.devices_named("touch")[0].clone();
    harness.client.start_emulating(&touch);
    harness.client.touchscreen(&touch).down(1, 10.0, 10.0);
    harness.client.frame(&touch);
    harness.pump();
    assert!(
        harness
            .recs()
            .iter()
            .any(|rec| matches!(rec, Rec::TouchDown { .. }))
    );

    let device = seat.touch_device().expect("seat should hold a touch device");
    touch.release();
    harness.pump();

    let removals: Vec<_> = harness
        .recs()
        .iter()
        .filter_map(|rec| match rec {
            Rec::DeviceRemoved { device, had_touch } => Some((device.clone(), *had_touch)),
            _ => None,
        })
        .collect();

    assert_eq!(
        removals.len(),
        1,
        "a closure should announce exactly one removal: {:?}",
        harness.recs()
    );
    assert!(
        removals[0].1,
        "the removal must be announced before the device is drained, or a consumer cannot \
         tell what kind of device it lost"
    );
    assert!(
        !device.has_capability(DeviceCapability::Touch),
        "the device should be drained once the closure completes"
    );
    assert!(
        seat.touch_device().is_none(),
        "the seat should not keep a device the client closed"
    );
    assert!(
        harness.client.destroyed(&touch),
        "the client should receive the device destructor"
    );
}

#[test]
fn releasing_the_touchscreen_keeps_the_device_and_reports_the_release() {
    let mut harness = Harness::new();
    let seat = harness.connect(|harness| {
        let seat = harness.add_seat("seat");
        seat.add_touch("touch", touch_regions());
        seat
    });

    let touch = harness.client.devices_named("touch")[0].clone();
    harness.client.start_emulating(&touch);
    harness.client.touchscreen(&touch).down(1, 10.0, 10.0);
    harness.client.frame(&touch);
    harness.pump();

    // The touchscreen goes, the device stays: no `ei_device.release` here.
    harness.client.touchscreen(&touch).release();
    harness.pump();

    assert_eq!(
        harness.client.touchscreens_destroyed, 1,
        "the client should get exactly one touchscreen destructor"
    );
    assert_eq!(
        harness.client.devices_named("touch").len(),
        1,
        "a released touchscreen is not reinitialized, so nothing should replace it"
    );
    assert!(
        !harness.client.destroyed(&touch),
        "the device itself should survive"
    );
    assert!(
        !harness
            .recs()
            .iter()
            .any(|rec| matches!(rec, Rec::DeviceRemoved { .. })),
        "releasing an interface is not a device removal: {:?}",
        harness.recs()
    );

    let device = seat
        .touch_device()
        .expect("the seat should still hold the device");
    assert!(
        !device.has_capability(DeviceCapability::Touch),
        "the device should no longer claim the capability the client gave up"
    );
    assert!(
        harness
            .recs()
            .iter()
            .any(|rec| matches!(rec, Rec::TouchscreenReleased { .. })),
        "the release should be delivered: {:?}",
        harness.recs()
    );
}

// The next three tests pin what an unframed touch does at each of the three ways a touch
// stream can end. Two of them commit the touch, one discards it, and the split is deliberate.
//
// reis synthesizes a frame before any request that carries no timestamp of its own but does
// name a device, flushing whatever that device left pending. This is libeis behavior that reis
// copies on purpose (its `queue_request` says as much), so a stop or a close commits an
// unframed touch rather than dropping it. We keep it: matching libeis is the point of an EIS
// implementation, and a consumer that diverges here would behave differently from every other
// EIS server for the same client bytes.
//
// A touchscreen release is the exception, and the reason for the pinned reis fork. Its request
// reports no device precisely so that no frame can be synthesized onto an interface that is
// being destroyed, and its pending touches are purged instead.

#[test]
fn stopping_emulation_commits_an_unframed_touch() {
    let mut harness = Harness::new();
    harness.connect(|harness| {
        let seat = harness.add_seat("seat");
        seat.add_touch("touch", touch_regions());
        seat
    });

    let touch = harness.client.devices_named("touch")[0].clone();
    harness.client.start_emulating(&touch);
    // A down the client never frames, so it is still pending when emulation stops.
    harness.client.touchscreen(&touch).down(1, 10.0, 10.0);
    touch.stop_emulating(harness.client.last_serial);
    harness.pump();

    // The frame between the down and the stop is synthesized by reis, not sent by the client.
    // The consumer therefore applies a touch the client never committed and then immediately
    // hears the stop that ends it, which nets out as a tap nobody asked for. That is what
    // libeis does with the same bytes, so it is the behavior we want.
    assert_eq!(
        tags(harness.recs()),
        [
            "connected",
            "device-added",
            "touch-down",
            "touch-frame",
            "stop-emulating"
        ],
        "an unframed down should be committed by a synthesized frame before the stop"
    );
}

#[test]
fn closing_a_device_commits_an_unframed_touch() {
    let mut harness = Harness::new();
    harness.connect(|harness| {
        let seat = harness.add_seat("seat");
        seat.add_touch("touch", touch_regions());
        seat
    });

    let touch = harness.client.devices_named("touch")[0].clone();
    harness.client.start_emulating(&touch);
    // A down the client never frames, so it is still pending when the device is closed.
    harness.client.touchscreen(&touch).down(1, 10.0, 10.0);
    touch.release();
    harness.pump();

    // Same synthesis as the stop above: closing a device names it, so reis flushes what the
    // device had pending first. The frame lands before the removal because the drain that
    // strips the device's interfaces only happens once the closure itself is handled.
    assert_eq!(
        tags(harness.recs()),
        [
            "connected",
            "device-added",
            "touch-down",
            "touch-frame",
            "device-removed"
        ],
        "an unframed down should be committed by a synthesized frame before the removal"
    );
}

// The contrast to the two tests above, and the case the reis fork exists for: a touchscreen
// release purges its own unframed touches instead of committing them, because a synthesized
// frame would be landing on an interface that is being destroyed. The touch is never delivered
// at all, so nothing downstream has a contact to tear down.
#[test]
fn releasing_the_touchscreen_purges_only_its_own_unframed_operations() {
    let mut harness = Harness::new();
    harness.connect(|harness| {
        let seat = harness.add_seat("seat");
        seat.add_pointer_absolute("abs", absolute_regions());
        seat.add_touch("touch", touch_regions());
        seat
    });

    let abs = harness.client.devices_named("abs")[0].clone();
    let touch = harness.client.devices_named("touch")[0].clone();
    harness.client.start_emulating(&abs);
    harness.client.start_emulating(&touch);

    // A touch that never gets a frame, so it is still pending when the release lands.
    harness.client.touchscreen(&touch).down(1, 10.0, 10.0);
    // A second device's operation, also pending, to pin that the purge is selective.
    let pointer: ei::PointerAbsolute = harness.client.interface(&abs).unwrap();
    pointer.motion_absolute(100.0, 200.0);
    harness.pump();

    assert!(
        !harness
            .recs()
            .iter()
            .any(|rec| matches!(rec, Rec::TouchDown { .. })),
        "an unframed down must not be delivered: {:?}",
        harness.recs()
    );

    harness.client.touchscreen(&touch).release();
    harness.pump();

    assert!(
        harness
            .recs()
            .iter()
            .any(|rec| matches!(rec, Rec::TouchscreenReleased { .. })),
        "the release should be delivered: {:?}",
        harness.recs()
    );

    // Force another drain. The purged down would surface here if it had survived, and the
    // pointer's operation must still be deliverable under its own frame.
    harness.client.frame(&abs);
    harness.pump();

    assert!(
        !harness
            .recs()
            .iter()
            .any(|rec| matches!(rec, Rec::TouchDown { .. })),
        "the purged down must not arrive under a later frame: {:?}",
        harness.recs()
    );
    assert!(
        harness
            .recs()
            .iter()
            .any(|rec| matches!(rec, Rec::PointerMotionAbsolute { x, y, .. } if *x == 100.0 && *y == 200.0)),
        "the other device's pending operation should have survived the release: {:?}",
        harness.recs()
    );
    assert!(
        !harness
            .recs()
            .iter()
            .any(|rec| matches!(rec, Rec::TouchFrame { .. })),
        "no touch frame should be synthesized: {:?}",
        harness.recs()
    );
}

#[test]
fn touch_requests_are_classified_by_the_synchronous_accessor() {
    let mut harness = Harness::new();
    let seat = harness.connect(|harness| harness.add_seat("seat"));

    seat.add_touch("touch", touch_regions());
    harness.flush();

    // The accessor answers immediately, while the `DeviceAdded` announcing the same device is
    // still queued on the internal channel. This is why a consumer cannot classify a device by
    // waiting for `DeviceAdded`: requests from the socket can be processed first.
    assert!(
        seat.touch_device().is_some(),
        "the accessor should reflect creation synchronously"
    );
    assert!(
        !harness
            .recs()
            .iter()
            .any(|rec| matches!(rec, Rec::DeviceAdded { .. })),
        "DeviceAdded should not have been delivered yet: {:?}",
        harness.recs()
    );

    harness.pump();
    let touch = harness.client.devices_named("touch")[0].clone();
    harness.client.start_emulating(&touch);
    harness.client.touchscreen(&touch).down(1, 10.0, 10.0);
    harness.client.frame(&touch);
    touch.release();
    harness.pump();

    let down = harness
        .recs()
        .iter()
        .find_map(|rec| match rec {
            Rec::TouchDown { classified_touch, .. } => Some(*classified_touch),
            _ => None,
        })
        .unwrap_or_else(|| panic!("expected a down: {:?}", harness.recs()));
    assert!(
        down,
        "the seat's accessor should have identified the touch device"
    );

    let removed = harness
        .recs()
        .iter()
        .find_map(|rec| match rec {
            Rec::DeviceRemoved { had_touch, .. } => Some(*had_touch),
            _ => None,
        })
        .unwrap_or_else(|| panic!("expected a removal: {:?}", harness.recs()));
    assert!(
        removed,
        "the closure should still identify as touch when announced"
    );
    assert!(
        seat.touch_device().is_none(),
        "the accessor should reflect the closure synchronously"
    );
}
