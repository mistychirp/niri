//! macshell-glass-v1: clients describe where liquid glass goes behind their surfaces.

use std::sync::Mutex;

use smithay::reexports::wayland_server::backend::ClientId;
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::reexports::wayland_server::{
    Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New, Resource, Weak,
};
use smithay::utils::{Logical, Rectangle};
use smithay::wayland::compositor::{self, Cacheable, SurfaceData};

use super::raw::macshell_glass::v1::server::macshell_glass_manager_v1::{
    self, MacshellGlassManagerV1,
};
use super::raw::macshell_glass::v1::server::macshell_glass_v1::{self, MacshellGlassV1};
use crate::render_helpers::framebuffer_effect::BrightnessProbe;

const VERSION: u32 = 2;

/// How much the luminance (out of 1000) has to change before clients hear about it.
const BRIGHTNESS_STEP: u32 = 20;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlassShape {
    /// Surface-local logical rectangle.
    pub rect: Rectangle<i32, Logical>,
    pub radius: f32,
    pub frosted: bool,
}

/// Double-buffered glass shapes of a surface.
#[derive(Debug, Clone, Default)]
pub struct GlassCachedState {
    pub shapes: Vec<GlassShape>,
}

impl Cacheable for GlassCachedState {
    fn commit(&mut self, _dh: &DisplayHandle) -> Self {
        self.clone()
    }

    fn merge_into(self, into: &mut Self, _dh: &DisplayHandle) {
        *into = self;
    }
}

/// Marks a surface that already has a glass object.
#[derive(Default)]
struct GlassAttached(Mutex<bool>);

/// Brightness tracking for a surface whose glass object asked for it.
#[derive(Default)]
struct BrightnessTracking(Mutex<Option<Tracker>>);

struct Tracker {
    glass: MacshellGlassV1,
    probe: BrightnessProbe,
    /// The last luminance sent, out of 1000.
    sent: Option<u32>,
}

/// The probe to measure the surface's background with, if its client tracks brightness.
pub fn brightness_probe(states: &SurfaceData) -> Option<BrightnessProbe> {
    let tracking = states.data_map.get::<BrightnessTracking>()?;
    let tracking = tracking.0.lock().unwrap();
    tracking.as_ref().map(|tracker| tracker.probe.clone())
}

/// Sends the latest measured brightness, if it changed noticeably since the last one sent.
pub fn report_brightness(states: &SurfaceData) {
    let Some(tracking) = states.data_map.get::<BrightnessTracking>() else {
        return;
    };
    let mut tracking = tracking.0.lock().unwrap();
    let Some(tracker) = tracking.as_mut() else {
        return;
    };
    let Some(luminance) = tracker.probe.luminance() else {
        return;
    };
    let luminance = (luminance.clamp(0., 1.) * 1000.).round() as u32;
    if tracker
        .sent
        .is_some_and(|sent| sent.abs_diff(luminance) < BRIGHTNESS_STEP)
    {
        return;
    }
    if tracker.glass.is_alive() {
        tracker.glass.brightness(luminance);
    }
    tracker.sent = Some(luminance);
}

fn set_tracking(surface: &WlSurface, tracker: Option<Tracker>) {
    compositor::with_states(surface, |states| {
        let tracking = states
            .data_map
            .get_or_insert_threadsafe(BrightnessTracking::default);
        *tracking.0.lock().unwrap() = tracker;
    });
}

pub struct GlassUserData(Weak<WlSurface>);

pub struct MacshellGlassManagerState;

pub trait MacshellGlassHandler {
    /// The surface's pending glass shapes changed; they apply on its next commit.
    fn glass_pending(&mut self, surface: &WlSurface);
}

impl MacshellGlassManagerState {
    pub fn new<D>(display: &DisplayHandle) -> Self
    where
        D: GlobalDispatch<MacshellGlassManagerV1, ()>,
        D: Dispatch<MacshellGlassManagerV1, ()>,
        D: Dispatch<MacshellGlassV1, GlassUserData>,
        D: MacshellGlassHandler,
        D: 'static,
    {
        display.create_global::<D, MacshellGlassManagerV1, _>(VERSION, ());
        Self
    }
}

fn set_attached(surface: &WlSurface, attached: bool) -> bool {
    compositor::with_states(surface, |states| {
        let flag = states
            .data_map
            .get_or_insert_threadsafe(GlassAttached::default);
        let mut flag = flag.0.lock().unwrap();
        std::mem::replace(&mut *flag, attached)
    })
}

fn edit_pending(surface: &WlSurface, f: impl FnOnce(&mut Vec<GlassShape>)) {
    compositor::with_states(surface, |states| {
        let mut cached = states.cached_state.get::<GlassCachedState>();
        f(&mut cached.pending().shapes);
    });
}

impl<D> GlobalDispatch<MacshellGlassManagerV1, (), D> for MacshellGlassManagerState
where
    D: GlobalDispatch<MacshellGlassManagerV1, ()>,
    D: Dispatch<MacshellGlassManagerV1, ()>,
    D: Dispatch<MacshellGlassV1, GlassUserData>,
    D: MacshellGlassHandler,
    D: 'static,
{
    fn bind(
        _state: &mut D,
        _handle: &DisplayHandle,
        _client: &Client,
        manager: New<MacshellGlassManagerV1>,
        _global_data: &(),
        data_init: &mut DataInit<'_, D>,
    ) {
        data_init.init(manager, ());
    }
}

impl<D> Dispatch<MacshellGlassManagerV1, (), D> for MacshellGlassManagerState
where
    D: Dispatch<MacshellGlassManagerV1, ()>,
    D: Dispatch<MacshellGlassV1, GlassUserData>,
    D: MacshellGlassHandler,
    D: 'static,
{
    fn request(
        _state: &mut D,
        _client: &Client,
        manager: &MacshellGlassManagerV1,
        request: <MacshellGlassManagerV1 as Resource>::Request,
        _data: &(),
        _dhandle: &DisplayHandle,
        data_init: &mut DataInit<'_, D>,
    ) {
        match request {
            macshell_glass_manager_v1::Request::GetGlass { id, surface } => {
                if set_attached(&surface, true) {
                    manager.post_error(
                        macshell_glass_manager_v1::Error::GlassExists,
                        "the surface already has a glass object",
                    );
                    return;
                }
                data_init.init(id, GlassUserData(surface.downgrade()));
            }
            macshell_glass_manager_v1::Request::Destroy => (),
        }
    }
}

impl<D> Dispatch<MacshellGlassV1, GlassUserData, D> for MacshellGlassManagerState
where
    D: Dispatch<MacshellGlassV1, GlassUserData>,
    D: MacshellGlassHandler,
    D: 'static,
{
    fn request(
        state: &mut D,
        _client: &Client,
        glass: &MacshellGlassV1,
        request: <MacshellGlassV1 as Resource>::Request,
        data: &GlassUserData,
        _dhandle: &DisplayHandle,
        _data_init: &mut DataInit<'_, D>,
    ) {
        // Inert once the surface is gone.
        let Ok(surface) = data.0.upgrade() else {
            return;
        };

        match request {
            macshell_glass_v1::Request::Clear => edit_pending(&surface, Vec::clear),
            macshell_glass_v1::Request::AddRoundedRect {
                x,
                y,
                width,
                height,
                radius,
                style,
            } => {
                if width <= 0 || height <= 0 {
                    glass.post_error(
                        macshell_glass_v1::Error::InvalidShape,
                        "glass shapes need a positive width and height",
                    );
                    return;
                }

                let frosted = matches!(style.into_result(), Ok(macshell_glass_v1::Style::Frosted));
                let shape = GlassShape {
                    rect: Rectangle::new((x, y).into(), (width, height).into()),
                    radius: radius as f32,
                    frosted,
                };
                edit_pending(&surface, |shapes| shapes.push(shape));
            }
            macshell_glass_v1::Request::TrackBrightness => {
                set_tracking(
                    &surface,
                    Some(Tracker {
                        glass: glass.clone(),
                        probe: BrightnessProbe::default(),
                        sent: None,
                    }),
                );
                // Measuring happens while rendering, so make sure the glass renders soon.
            }
            // Handled in destroyed(), which also covers clients that disconnect.
            macshell_glass_v1::Request::Destroy => return,
        }

        state.glass_pending(&surface);
    }

    fn destroyed(state: &mut D, _client: ClientId, _resource: &MacshellGlassV1, data: &GlassUserData) {
        let Ok(surface) = data.0.upgrade() else {
            return;
        };
        set_attached(&surface, false);
        set_tracking(&surface, None);
        edit_pending(&surface, Vec::clear);
        state.glass_pending(&surface);
    }
}

#[macro_export]
macro_rules! delegate_macshell_glass {
    ($(@<$( $lt:tt $( : $clt:tt $(+ $dlt:tt )* )? ),+>)? $ty: ty) => {
        smithay::reexports::wayland_server::delegate_global_dispatch!($(@< $( $lt $( : $clt $(+ $dlt )* )? ),+ >)? $ty: [
            $crate::protocols::raw::macshell_glass::v1::server::macshell_glass_manager_v1::MacshellGlassManagerV1: ()
        ] => $crate::protocols::macshell_glass::MacshellGlassManagerState);

        smithay::reexports::wayland_server::delegate_dispatch!($(@< $( $lt $( : $clt $(+ $dlt )* )? ),+ >)? $ty: [
            $crate::protocols::raw::macshell_glass::v1::server::macshell_glass_manager_v1::MacshellGlassManagerV1: ()
        ] => $crate::protocols::macshell_glass::MacshellGlassManagerState);

        smithay::reexports::wayland_server::delegate_dispatch!($(@< $( $lt $( : $clt $(+ $dlt )* )? ),+ >)? $ty: [
            $crate::protocols::raw::macshell_glass::v1::server::macshell_glass_v1::MacshellGlassV1: $crate::protocols::macshell_glass::GlassUserData
        ] => $crate::protocols::macshell_glass::MacshellGlassManagerState);
    };
}
