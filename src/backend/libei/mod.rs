//! Input backend for `libei` sender contexts
//!
//! ``` no_run
//! # let event_loop: calloop::EventLoop<()> = todo!();
//!
//! use reis::{calloop::EisListenerSource, eis};
//! use smithay::{
//!     backend::libei::{EiInput, EiInputEvent, EiRegion},
//!     input::keyboard::XkbConfig,
//! };
//!
//! let listener = eis::Listener::bind_auto().unwrap();
//! let listener_source = EisListenerSource::new(listener);
//! let handle = event_loop.handle();
//! event_loop.handle().insert_source(listener_source, |context, _, _| {
//!     let source = EiInput::new(context);
//!     handle.insert_source(source, |event, connection, _data| {
//!         match event {
//!              EiInputEvent::Connected => {
//!                 // One region per output, describing the area these devices can reach.
//!                 // Absolute pointer and touch devices must have at least one.
//!                 let regions = vec![EiRegion {
//!                     offset_x: 0,
//!                     offset_y: 0,
//!                     width: 1920,
//!                     height: 1080,
//!                     scale: 1.0,
//!                 }];
//!                 let seat = connection.add_seat("default");
//!                 let _ = seat.add_keyboard("virtual keyboard", XkbConfig::default());
//!                 seat.add_pointer("virtual pointer");
//!                 seat.add_pointer_absolute("virtual absolute pointer", regions.clone());
//!                 seat.add_touch("virtual touch", regions);
//!             }
//!             EiInputEvent::Disconnected => {}
//!             EiInputEvent::Event(event) => {
//!                 // Pass input event to compositor's input event handling logic
//!                 // ...
//!             }
//!         }
//!     }).unwrap();
//!     Ok(calloop::PostAction::Continue)
//! }).unwrap();
//! ```

// TODO: Add helper for receiver contexts

use calloop::{EventSource, PostAction, Readiness, Token, TokenFactory};
use reis::{
    calloop::EisRequestSourceEvent,
    eis,
    request::{DeviceCapability, EisRequest},
};
use std::{
    io,
    sync::{Arc, Mutex},
};

use crate::backend::input::InputEvent;

mod input;
pub use input::{EiSpecialEvent, ScrollEvent};
mod seat;
pub use seat::{EiInputSeat, EiRegion};

#[cfg(test)]
mod tests;

/// An [`EventSource`] for receiving input from an EI sender context and
/// converting to [`InputEvent`]s.
#[derive(Debug)]
pub struct EiInput {
    source: reis::calloop::EisRequestSource,
    connection: Option<EiInputConnection>,
    event_sender: calloop::channel::Sender<InputEvent<EiInput>>,
    channel_source: calloop::channel::Channel<InputEvent<EiInput>>,
}

impl EiInput {
    /// Create an EI sender event source.
    ///
    /// `context` should be a new EI socket that has not been used yet.
    pub fn new(context: eis::Context) -> Self {
        let (event_sender, channel_source) = calloop::channel::channel();
        Self {
            source: reis::calloop::EisRequestSource::new(context, 0),
            event_sender,
            channel_source,
            connection: None,
        }
    }
}

/// A connection for an EI sender context that can be used to add seats and
/// devices.
#[derive(Clone, Debug)]
pub struct EiInputConnection(Arc<EiInputConnectionInner>);

impl PartialEq for EiInputConnection {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

#[derive(Debug)]
struct EiInputConnectionInner {
    connection: reis::request::Connection,
    event_sender: calloop::channel::Sender<InputEvent<EiInput>>,
    seats: Mutex<Vec<EiInputSeat>>,
}

impl EiInputConnection {
    fn new(
        connection: reis::request::Connection,
        event_sender: calloop::channel::Sender<InputEvent<EiInput>>,
    ) -> Self {
        Self(Arc::new(EiInputConnectionInner {
            connection,
            event_sender,
            seats: Mutex::new(Vec::new()),
        }))
    }

    /// Add a seat to the EI connection
    pub fn add_seat(&self, name: &str) -> EiInputSeat {
        let seat = self.0.connection.add_seat(
            Some(name),
            // Capabilities can't be added to a seat; so advertise them
            // all, but only create relevant devices on bind.
            DeviceCapability::Pointer
                | DeviceCapability::PointerAbsolute
                | DeviceCapability::Keyboard
                | DeviceCapability::Touch
                | DeviceCapability::Scroll
                | DeviceCapability::Button,
        );
        let seat = EiInputSeat::new(self, seat, self.0.event_sender.clone());
        self.0.seats.lock().unwrap().push(seat.clone());
        seat
    }

    // Complete the teardown of a device the client closed, on whichever seat owns it.
    fn device_closed(&self, device: &reis::request::Device) {
        for seat in self.0.seats.lock().unwrap().iter() {
            if seat.device_closed(device) {
                break;
            }
        }
    }

    /// Send buffered events on EI socket
    pub fn flush(&self) -> rustix::io::Result<()> {
        self.0.connection.flush()
    }

    /// Returns the underlying `eis` connection.
    pub fn eis_connection(&self) -> &eis::Connection {
        self.0.connection.connection()
    }
}

/// An event produced by an [`EiInput`] event source.
#[derive(Debug)]
pub enum EiInputEvent {
    /// The client has finished the EI handshake. Seats and devices can
    /// then be added.
    Connected,
    /// The client has disconnected from the server.
    Disconnected,
    /// An input event has been received from the client.
    Event(InputEvent<EiInput>),
}

impl EventSource for EiInput {
    type Event = EiInputEvent;
    type Metadata = EiInputConnection;
    type Ret = ();
    type Error = io::Error;

    fn process_events<F>(
        &mut self,
        readiness: Readiness,
        token: Token,
        mut cb: F,
    ) -> Result<PostAction, <Self as EventSource>::Error>
    where
        F: FnMut(EiInputEvent, &mut EiInputConnection),
    {
        let _ = self.channel_source.process_events(readiness, token, |event, ()| {
            if let calloop::channel::Event::Msg(event) = event {
                // Can't create device until there's a connection, so no channel messages
                let connection = self.connection.as_mut().unwrap();
                cb(EiInputEvent::Event(event), connection);
            }
        });
        self.source.process_events(readiness, token, |event, connection| {
            // Wrap connection in `EiInputConnection` if not created yet
            if self.connection.is_none() {
                self.connection = Some(EiInputConnection::new(
                    connection.clone(),
                    self.event_sender.clone(),
                ));
            }
            let connection = self.connection.as_mut().unwrap();

            match event {
                Ok(EisRequestSourceEvent::Connected) => {
                    if connection.0.connection.context_type() == eis::handshake::ContextType::Receiver {
                        connection
                            .0
                            .connection
                            .disconnected(eis::connection::DisconnectReason::Disconnected, None);
                        let _ = connection.flush();
                        return Ok(PostAction::Remove);
                    }
                    cb(EiInputEvent::Connected, connection);
                }
                Ok(EisRequestSourceEvent::Request(EisRequest::Disconnect)) => {
                    cb(EiInputEvent::Disconnected, connection);
                    return Ok(PostAction::Remove);
                }
                Ok(EisRequestSourceEvent::Request(EisRequest::Bind(request))) => {
                    if let Some(seat) = connection
                        .0
                        .seats
                        .lock()
                        .unwrap()
                        .iter()
                        .find(|seat| **seat == request.seat)
                    {
                        seat.bind(request.capabilities);
                    }
                }
                Ok(EisRequestSourceEvent::Request(EisRequest::DeviceClosed(request))) => {
                    // Announce the removal before completing it. `Device::remove` drains the
                    // device's interfaces and `has_capability` reads exactly those, so a
                    // consumer told about the removal afterwards could no longer tell what
                    // kind of device it just lost.
                    cb(
                        EiInputEvent::Event(InputEvent::DeviceRemoved {
                            device: request.device.clone(),
                        }),
                        connection,
                    );
                    // reis requires `Device::remove` after a `DeviceClosed`; without it the
                    // protocol destructor never goes out and the seat keeps a dead device.
                    connection.device_closed(&request.device);
                }
                Ok(EisRequestSourceEvent::Request(request)) => {
                    if let Some(input_event) = convert_request(request) {
                        cb(EiInputEvent::Event(input_event), connection);
                    }
                }
                Err(err) => {
                    tracing::error!("Libei client error: {}", err);
                    // The connection is going away exactly as it does on a clean disconnect, so
                    // say so. Without this a consumer keeping per-connection state only ever
                    // hears about the tidy exit and leaks the entry for every client that errors.
                    cb(EiInputEvent::Disconnected, connection);
                    return Ok(PostAction::Remove);
                }
            }
            let _ = connection.flush();
            Ok(PostAction::Continue)
        })
    }

    fn register(
        &mut self,
        poll: &mut calloop::Poll,
        token_factory: &mut TokenFactory,
    ) -> Result<(), calloop::Error> {
        self.channel_source.register(poll, token_factory)?;
        self.source.register(poll, token_factory)
    }

    fn reregister(
        &mut self,
        poll: &mut calloop::Poll,
        token_factory: &mut TokenFactory,
    ) -> Result<(), calloop::Error> {
        self.channel_source.reregister(poll, token_factory)?;
        self.source.reregister(poll, token_factory)
    }

    fn unregister(&mut self, poll: &mut calloop::Poll) -> Result<(), calloop::Error> {
        self.channel_source.unregister(poll)?;
        self.source.unregister(poll)
    }
}

fn convert_request(request: EisRequest) -> Option<InputEvent<EiInput>> {
    match request {
        EisRequest::KeyboardKey(event) => Some(InputEvent::Keyboard { event }),
        EisRequest::PointerMotion(event) => Some(InputEvent::PointerMotion { event }),
        EisRequest::PointerMotionAbsolute(event) => Some(InputEvent::PointerMotionAbsolute { event }),
        EisRequest::Button(event) => Some(InputEvent::PointerButton { event }),
        EisRequest::ScrollDelta(event) => Some(InputEvent::PointerAxis {
            event: ScrollEvent::Delta(event),
        }),
        EisRequest::ScrollStop(event) => Some(InputEvent::PointerAxis {
            event: ScrollEvent::Stop(event),
        }),
        EisRequest::ScrollCancel(event) => Some(InputEvent::PointerAxis {
            event: ScrollEvent::Cancel(event),
        }),
        EisRequest::ScrollDiscrete(event) => Some(InputEvent::PointerAxis {
            event: ScrollEvent::Discrete(event),
        }),
        EisRequest::TouchDown(event) => Some(InputEvent::TouchDown { event }),
        EisRequest::TouchUp(event) => Some(InputEvent::TouchUp { event }),
        EisRequest::TouchMotion(event) => Some(InputEvent::TouchMotion { event }),
        EisRequest::TouchCancel(event) => Some(InputEvent::TouchCancel { event }),
        // A frame is the transaction boundary for the operations a device has queued, and only
        // touch has an equivalent here. Frames from other devices carry nothing to deliver.
        EisRequest::Frame(event) => event
            .device
            .has_capability(DeviceCapability::Touch)
            .then(|| InputEvent::TouchFrame { event }),
        EisRequest::DeviceStopEmulating(event) => Some(InputEvent::Special(EiSpecialEvent::StopEmulating(
            event,
        ))),
        EisRequest::TouchscreenReleased(event) => Some(InputEvent::Special(
            EiSpecialEvent::TouchscreenReleased(event),
        )),
        // Handled in `process_events`, which has the connection needed to complete the teardown.
        EisRequest::DeviceClosed(_) => None,
        // TODO: handle `TextKeysym`/`TextUtf8` once `add_text()` support is added.
        EisRequest::TextKeysym(_)
        | EisRequest::TextUtf8(_)
        | EisRequest::Disconnect
        | EisRequest::Bind(_)
        | EisRequest::RequestDevice(_)
        | EisRequest::Ready(_)
        | EisRequest::DeviceStartEmulating(_) => None,
    }
}
