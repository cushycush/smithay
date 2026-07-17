use smithay::{
    backend::libei::{EiInput, EiInputEvent, EiRegion},
    input::keyboard::XkbConfig,
    reexports::{
        calloop,
        reis::{calloop::EisListenerSource, eis},
    },
};

use crate::{state::AnvilState, udev::UdevData};

// The area the absolute pointer and touch devices can reach, one region per output. The protocol
// requires at least one region on a virtual device advertising either capability, and silently
// discards events that land outside them.
fn regions(state: &AnvilState<UdevData>) -> Vec<EiRegion> {
    state
        .space
        .outputs()
        .filter_map(|output| {
            let geo = state.space.output_geometry(output)?;
            Some(EiRegion {
                offset_x: geo.loc.x.max(0) as u32,
                offset_y: geo.loc.y.max(0) as u32,
                width: geo.size.w.max(0) as u32,
                height: geo.size.h.max(0) as u32,
                scale: output.current_scale().fractional_scale() as f32,
            })
        })
        .collect()
}

pub fn listen_eis(handle: &calloop::LoopHandle<'static, AnvilState<UdevData>>) {
    let listener = match eis::Listener::bind_auto() {
        Ok(listener) => listener,
        Err(err) => {
            tracing::error!("Failed to bind EI listener socket: {}", err);
            return;
        }
    };

    unsafe { std::env::set_var("LIBEI_SOCKET", listener.path()) };

    let listener_source = EisListenerSource::new(listener);
    let handle_clone = handle.clone();
    handle
        .insert_source(listener_source, move |context, _, _| {
            let source = EiInput::new(context);
            handle_clone
                .insert_source(source, |event, connection, data| match event {
                    EiInputEvent::Connected => {
                        let regions = regions(data);
                        let seat = connection.add_seat("default");
                        let _ = seat.add_keyboard("virtual keyboard", XkbConfig::default());
                        seat.add_pointer("virtual pointer");
                        // Advertising either of these with no region is an EIS implementation bug
                        // by the protocol's own words, so with no output mapped they are skipped.
                        if !regions.is_empty() {
                            seat.add_pointer_absolute("virtual absolute pointer", regions.clone());
                            seat.add_touch("virtual touch", regions);
                        }
                    }
                    EiInputEvent::Disconnected => {}
                    EiInputEvent::Event(event) => {
                        let dh = data.display_handle.clone();
                        data.process_input_event(&dh, event);
                    }
                })
                .unwrap();
            Ok(calloop::PostAction::Continue)
        })
        .unwrap();
}
