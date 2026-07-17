use reis::{eis::device::DeviceType, enumflags2::BitFlags, request::DeviceCapability};
use std::sync::{Arc, Mutex, Weak};
use xkbcommon::xkb;

use crate::{
    backend::input::InputEvent,
    input::keyboard::{KeymapFile, XkbConfig},
};

use super::{EiInput, EiInputConnection, EiInputConnectionInner};

/// A region of the EI desktop layout that a device can reach.
///
/// Offsets and sizes are in logical pixels on the EI desktop layout, which the server defines.
/// A device's regions are the area it can address: the protocol silently discards absolute
/// pointer and touch events outside them.
///
/// Advertising a `device_type.virtual` device with the touch or absolute pointer capability and
/// no region is an EIS implementation bug, per the protocol.
#[derive(Clone, Debug, PartialEq)]
pub struct EiRegion {
    /// Offset of the region on the x axis, in logical pixels.
    pub offset_x: u32,
    /// Offset of the region on the y axis, in logical pixels.
    pub offset_y: u32,
    /// Width of the region, in logical pixels.
    pub width: u32,
    /// Height of the region, in logical pixels.
    pub height: u32,
    /// Scale relating this region's logical pixels to physical ones.
    pub scale: f32,
}

/// A seat advertised on an EI sender context
#[derive(Clone, Debug)]
pub struct EiInputSeat(Arc<Mutex<EiInputSeatInner>>);

impl PartialEq<reis::request::Seat> for EiInputSeat {
    fn eq(&self, other: &reis::request::Seat) -> bool {
        self.0.lock().unwrap().seat == *other
    }
}

impl PartialEq for EiInputSeat {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl EiInputSeat {
    pub(super) fn new(
        connection: &EiInputConnection,
        seat: reis::request::Seat,
        event_sender: calloop::channel::Sender<InputEvent<EiInput>>,
    ) -> Self {
        Self(Arc::new(Mutex::new(EiInputSeatInner {
            connection: Arc::downgrade(&connection.0),
            seat,
            event_sender,
            keyboard: None,
            pointer: None,
            pointer_absolute: None,
            touch: None,
            device_keyboard: None,
            device_pointer: None,
            device_pointer_absolute: None,
            device_touch: None,
            bound_capabilities: BitFlags::empty(),
        })))
    }

    pub(super) fn bind(&self, capabilities: BitFlags<DeviceCapability>) {
        let mut inner = self.0.lock().unwrap();
        inner.bound_capabilities = capabilities;
        inner.refresh_devices();
    }

    /// Add a keyboard device to the EI seat
    ///
    /// Calling on a seat that already has a keyboard device will remove
    /// that device and add a new one.
    pub fn add_keyboard(
        &self,
        name: &str,
        xkb_config: XkbConfig<'_>,
    ) -> Result<(), crate::input::keyboard::Error> {
        let mut inner = self.0.lock().unwrap();
        inner.device_keyboard = None;
        let context = xkb::Context::new(xkb::CONTEXT_NO_FLAGS);
        let keymap = xkb_config.compile_keymap(&context).map_err(|_| {
            tracing::debug!("Loading keymap from XkbConfig failed");
            crate::input::keyboard::Error::BadKeymap
        })?;
        let keymap_file = KeymapFile::new(&keymap);
        inner.keyboard = Some((name.to_string(), keymap_file));
        inner.refresh_devices();
        Ok(())
    }

    /// Remove keyboard device from the EI seat
    pub fn remove_keyboard(&self) {
        let mut inner = self.0.lock().unwrap();
        inner.device_keyboard = None;
        inner.keyboard = None;
    }

    /// Add a pointer device to the EI seat
    ///
    /// The EI device will have button and scroll capabilities in addition
    /// to the pointer capability.
    ///
    /// Calling on a seat that already has a pointer device will remove
    /// that device and add a new one.
    pub fn add_pointer(&self, name: &str) {
        let mut inner = self.0.lock().unwrap();
        inner.device_pointer = None;
        inner.pointer = Some(name.to_string());
        inner.refresh_devices();
    }

    /// Remove pointer device from the EI seat
    pub fn remove_pointer(&self) {
        let mut inner = self.0.lock().unwrap();
        inner.device_pointer = None;
        inner.pointer = None;
    }

    /// Add an absolute pointer device to the EI seat
    ///
    /// The EI device will have button and scroll capabilities in addition
    /// to the pointer absolute capability.
    ///
    /// `regions` describes the area of the EI desktop layout the device can reach. The
    /// protocol requires at least one region on a virtual absolute pointer device, and
    /// silently discards events outside them.
    ///
    /// Calling on a seat that already has an absolute pointer device will remove
    /// that device and add a new one.
    pub fn add_pointer_absolute(&self, name: &str, regions: Vec<EiRegion>) {
        let mut inner = self.0.lock().unwrap();
        inner.device_pointer_absolute = None;
        inner.pointer_absolute = Some((name.to_string(), regions));
        inner.refresh_devices();
    }

    /// Remove absolute pointer device from the EI seat
    pub fn remove_pointer_absolute(&self) {
        let mut inner = self.0.lock().unwrap();
        inner.device_pointer_absolute = None;
        inner.pointer_absolute = None;
    }

    /// Add a touch device to the EI seat
    ///
    /// `regions` describes the area of the EI desktop layout the device can reach. The
    /// protocol requires at least one region on a virtual touch device, and silently
    /// discards a touch down outside them along with the rest of that touch's sequence.
    ///
    /// Calling on a seat that already has a touch device will remove
    /// that device and add a new one.
    pub fn add_touch(&self, name: &str, regions: Vec<EiRegion>) {
        let mut inner = self.0.lock().unwrap();
        inner.device_touch = None;
        inner.touch = Some((name.to_string(), regions));
        inner.refresh_devices();
    }

    /// Remove touch device from the EI seat
    pub fn remove_touch(&self) {
        let mut inner = self.0.lock().unwrap();
        inner.device_touch = None;
        inner.touch = None;
    }

    /// The touch device currently created on this seat, if any.
    ///
    /// This reflects device creation and re-creation synchronously, which is what makes it
    /// usable for identifying a device. [`InputEvent::DeviceAdded`] cannot serve that purpose:
    /// it is queued through an internal channel while requests from the client's socket are
    /// dispatched directly, so a touch request or a device closure can be processed before the
    /// `DeviceAdded` announcing the device it refers to is ever seen.
    pub fn touch_device(&self) -> Option<reis::request::Device> {
        let inner = self.0.lock().unwrap();
        inner.device_touch.as_ref().map(|wrapper| wrapper.device.clone())
    }

    /// Remove seat from EI connection
    pub fn remove(&self) {
        let inner = self.0.lock().unwrap();
        inner.seat.remove();
        if let Some(connection) = inner.connection.upgrade() {
            let mut seats = connection.seats.lock().unwrap();
            if let Some(idx) = seats.iter().position(|s| Arc::ptr_eq(&s.0, &self.0)) {
                seats.remove(idx);
            }
        }
    }
}

#[derive(Debug)]
struct EiInputSeatInner {
    connection: Weak<EiInputConnectionInner>,
    seat: reis::request::Seat,
    bound_capabilities: BitFlags<DeviceCapability>,
    event_sender: calloop::channel::Sender<InputEvent<EiInput>>,
    // Interfaces advertised by the server
    keyboard: Option<(String, KeymapFile)>,
    pointer: Option<String>,
    pointer_absolute: Option<(String, Vec<EiRegion>)>,
    touch: Option<(String, Vec<EiRegion>)>,
    // Devices created in response to client bind
    device_keyboard: Option<DeviceDropWrapper>,
    device_pointer: Option<DeviceDropWrapper>,
    device_pointer_absolute: Option<DeviceDropWrapper>,
    device_touch: Option<DeviceDropWrapper>,
}

impl EiInputSeatInner {
    // Add any devices the server provides, for capabilities the client has bound, if no device yet
    fn refresh_devices(&mut self) {
        if self.device_keyboard.is_none() && self.bound_capabilities.contains(DeviceCapability::Keyboard) {
            if let Some((name, keymap_file)) = self.keyboard.as_ref() {
                let device = self.seat.add_device(
                    Some(name),
                    DeviceType::Virtual,
                    DeviceCapability::Keyboard.into(),
                    |device| {
                        let keyboard = device.interface::<reis::eis::Keyboard>().unwrap();
                        let _ = keymap_file.with_fd(true, |fd, len| {
                            keyboard.keymap(reis::eis::keyboard::KeymapType::Xkb, len as u32, fd);
                        });
                    },
                );
                device.resumed();
                let _ = self.event_sender.send(InputEvent::DeviceAdded {
                    device: device.clone(),
                });
                self.device_keyboard = Some(DeviceDropWrapper::new(device, &self.event_sender));
            }
        }

        if self.device_pointer.is_none() && self.bound_capabilities.contains(DeviceCapability::Pointer) {
            if let Some(name) = self.pointer.as_ref() {
                let device = self.seat.add_device(
                    Some(name),
                    DeviceType::Virtual,
                    DeviceCapability::Pointer | DeviceCapability::Button | DeviceCapability::Scroll,
                    |_| {},
                );
                device.resumed();
                let _ = self.event_sender.send(InputEvent::DeviceAdded {
                    device: device.clone(),
                });
                self.device_pointer = Some(DeviceDropWrapper::new(device, &self.event_sender));
            }
        }

        if self.device_pointer_absolute.is_none()
            && self
                .bound_capabilities
                .contains(DeviceCapability::PointerAbsolute)
        {
            if let Some((name, regions)) = self.pointer_absolute.as_ref() {
                let device = self.seat.add_device(
                    Some(name),
                    DeviceType::Virtual,
                    DeviceCapability::PointerAbsolute | DeviceCapability::Button | DeviceCapability::Scroll,
                    |device| emit_regions(device, regions),
                );
                device.resumed();
                let _ = self.event_sender.send(InputEvent::DeviceAdded {
                    device: device.clone(),
                });
                self.device_pointer_absolute = Some(DeviceDropWrapper::new(device, &self.event_sender));
            }
        }

        if self.device_touch.is_none() && self.bound_capabilities.contains(DeviceCapability::Touch) {
            if let Some((name, regions)) = self.touch.as_ref() {
                let device = self.seat.add_device(
                    Some(name),
                    DeviceType::Virtual,
                    DeviceCapability::Touch.into(),
                    |device| emit_regions(device, regions),
                );
                device.resumed();
                let _ = self.event_sender.send(InputEvent::DeviceAdded {
                    device: device.clone(),
                });
                self.device_touch = Some(DeviceDropWrapper::new(device, &self.event_sender));
            }
        }
    }
}

// Send a device's regions. Must run before `device.done()`, which is what `add_device`'s
// `before_done_cb` guarantees.
fn emit_regions(device: &reis::request::Device, regions: &[EiRegion]) {
    for region in regions {
        device.device().region(
            region.offset_x,
            region.offset_y,
            region.width,
            region.height,
            region.scale,
        );
    }
}

// Helper that remove the device on drop, and send `DeviceRemoved`
#[derive(Debug)]
struct DeviceDropWrapper {
    device: reis::request::Device,
    event_sender: calloop::channel::Sender<InputEvent<EiInput>>,
}

impl DeviceDropWrapper {
    fn new(
        device: reis::request::Device,
        event_sender: &calloop::channel::Sender<InputEvent<EiInput>>,
    ) -> Self {
        Self {
            device,
            event_sender: event_sender.clone(),
        }
    }
}

impl Drop for DeviceDropWrapper {
    fn drop(&mut self) {
        let _ = self.event_sender.send(InputEvent::DeviceRemoved {
            device: self.device.clone(),
        });
        self.device.remove();
    }
}
