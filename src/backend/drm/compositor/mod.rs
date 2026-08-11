//! Composition for [`Element`]s using drm planes
//!
//! When possible composition can be (partially) offloaded to the display driver by assigning
//! elements to drm planes. This is especially important for latency intensive fullscreen clients
//! like video renderers or games.
//!
//! The [`DrmCompositor`] does so by walking the stack of provided [`Element`]s from front to back
//! while trying to assign each element to a drm overlay plane. Each item that fails the plane test
//! will be rendered on the primary plane using the provided [`Renderer`].
//! Additionally it will try to assign the top most element that fit's into the cursor size (as specified
//! by the [`DrmDevice`](crate::backend::drm::DrmDevice)) on the cursor plane. If the element can not be
//! directly scanned out, pixman will be used to render the element.
//!
//! Note: While the [`DrmCompositor`] also works on *legacy* drm the use of overlay and cursor planes is disabled in that case.
//! Direct scan-out will only work with an atomic [`DrmSurface`].
//!
//! ## What makes a [`Element`] eligible for direct scan-out
//!
//! ### General
//!
//! First the element has to provide a [`UnderlyingStorage`] which can be exported as a drm framebuffer.
//! Currently this is limited to wayland buffers, but may be extended in the future.
//! This module provides a default exporter based on [`gbm`] which should fit most use-cases.
//!
//! If a certain combination of elements works can only be determined by asking the driver by submitting
//! a atomic commit test. If that test fails the element is scheduled to be rendered on the primary plane.
//!
//! ### Overlay planes
//!
//! The element can only be directly scanned out if it's geometry does not overlap with an already assigned
//! element on a plane higher in the stack.
//!
//! ### Underlay planes
//!
//! An underlay plane is only used if it does not overlap with an already assigned plane lower in the stack
//! and the element is fully opaque.
//!
//! ### Primary plane
//!
//! For an element to be considered to be directly scanned out on the primary plane it has to be the last remaining
//! visible element on the output and no other element has been assigned to the primary plane. If there are multiple
//! element assigned to the primary plane the renderer will be used to composite the primary plane into a allocator
//! provided buffer. Additionally the element has to be either fully opaque or the clear color has to match the CRTC
//! background color and no overlap with an underlay is found.
//!
//! # How to use it
//!
//! ```no_run
//! # use smithay::backend::{
//! #     allocator::gbm::{GbmAllocator, GbmDevice},
//! #     drm::{DrmDevice, DrmDeviceFd},
//! #     renderer::{
//! #       element::surface::WaylandSurfaceRenderElement,
//! #       gles::{GlesTexture, GlesRenderer},
//! #     },
//! # };
//! # use drm_fourcc::{DrmFormat, DrmFourcc, DrmModifier};
//! # use std::{collections::HashSet, mem::MaybeUninit};
//! #
//! use smithay::{
//!     backend::drm::{
//!         compositor::{DrmCompositor, FrameFlags},
//!         exporter::gbm::GbmFramebufferExporter,
//!         DrmSurface,
//!     },
//!     output::{Output, PhysicalProperties, Subpixel},
//!     utils::Size,
//! };
//!
//! // ...initialize the output, drm device, drm surface and allocator
//! #
//! # const CLEAR_COLOR: [f32; 4] = [0f32, 0f32, 0f32, 0f32];
//! #
//! let output = Output::new(
//!     "e-DP".into(),
//!     PhysicalProperties {
//!         size: Size::from((800, 600)),
//!         make: "N/A".into(),
//!         model: "N/A".into(),
//!         subpixel: Subpixel::Unknown,
//!         serial_number: "N/A".into(),
//!     },
//! );
//!
//! # let device: DrmDevice = todo!();
//! # let surface: DrmSurface = todo!();
//! # let allocator: GbmAllocator<DrmDeviceFd> = todo!();
//! # let exporter: GbmFramebufferExporter<DrmDeviceFd> = todo!();
//! # let color_formats = [DrmFourcc::Argb8888];
//! # let renderer_formats = HashSet::from([DrmFormat {
//! #     code: DrmFourcc::Argb8888,
//! #     modifier: DrmModifier::Linear,
//! # }]);
//! # let gbm: GbmDevice<DrmDeviceFd> = todo!();
//! # let mut renderer: GlesRenderer = todo!();
//! #
//! let mut compositor: DrmCompositor<_, _, (), _> = DrmCompositor::new(
//!     &output,
//!     surface,
//!     None,
//!     allocator,
//!     exporter,
//!     color_formats,
//!     renderer_formats,
//!     device.cursor_size(),
//!     Some(gbm),
//! )
//! .expect("failed to initialize drm compositor");
//!
//! # let elements: Vec<WaylandSurfaceRenderElement<GlesRenderer>> = Vec::new();
//! let render_frame_result = compositor
//!     .render_frame::<_, _>(&mut renderer, &elements, CLEAR_COLOR, FrameFlags::DEFAULT)
//!     .expect("failed to render frame");
//!
//! if !render_frame_result.is_empty {
//!     compositor.queue_frame(()).expect("failed to queue frame");
//!
//!     // ...wait for VBlank event
//!
//!     compositor
//!         .frame_submitted()
//!         .expect("failed to mark frame as submitted");
//! } else {
//!     // ...re-schedule frame
//! }
//! ```
use std::{
    collections::HashMap,
    fmt::Debug,
    io::ErrorKind,
    os::unix::io::{AsFd, OwnedFd},
    str::FromStr,
    sync::Arc,
};

use drm::{
    Device, DriverCapability,
    control::{Device as _, Mode, PlaneType, connector, crtc, framebuffer, plane},
};
use drm_fourcc::{DrmFormat, DrmFourcc, DrmModifier};
use indexmap::{IndexMap, IndexSet};
use smallvec::SmallVec;
use tracing::{debug, error, info, info_span, instrument, trace, warn};
use wayland_server::{Resource, protocol::wl_buffer::WlBuffer};

#[cfg(feature = "renderer_pixman")]
use crate::backend::renderer::{
    Frame as _, ImportAll,
    pixman::{PixmanError, PixmanRenderer, PixmanTexture},
};
use crate::{
    backend::{
        SwapBuffersError,
        allocator::{
            Allocator, Buffer, Slot, Swapchain,
            dmabuf::{AsDmabuf, Dmabuf},
            format::{get_opaque, has_alpha},
            gbm::{GbmAllocator, GbmBuffer, GbmBufferFlags, GbmDevice},
        },
        drm::{CursorPlanePolicy, DrmError, PlaneDamageClips, plane_has_property},
        renderer::{
            Bind, Color32F, DebugFlags, Renderer, RendererSuper, Texture, buffer_y_inverted,
            damage::{Error as OutputDamageTrackerError, OutputDamageTracker},
            element::{
                Element, Id, Kind, RenderElement, RenderElementPresentationState, RenderElementState,
                RenderElementStates, RenderingReason, UnderlyingStorage,
            },
            sync::SyncPoint,
            utils::{CommitCounter, DamageBag},
        },
    },
    output::OutputModeSource,
    utils::{Buffer as BufferCoords, DevPath, Physical, Point, Rectangle, Scale, Size, Transform},
    wayland::{shm, single_pixel_buffer},
};

use super::{
    DrmSurface, Framebuffer, PlaneClaim, PlaneInfo, Planes,
    error::AccessError,
    exporter::{ExportBuffer, ExportFramebuffer, gbm::GbmFramebufferExporter, gbm::NodeFilter},
    surface::{PresentationMode, VrrSupport},
};

mod elements;
mod frame_result;

use elements::*;
pub use frame_result::*;

impl RenderElementState {
    pub(crate) fn zero_copy(visible_area: usize) -> Self {
        RenderElementState {
            visible_area,
            presentation_state: RenderElementPresentationState::ZeroCopy,
            needs_capture: false,
        }
    }

    pub(crate) fn rendering_with_reason(reason: RenderingReason) -> Self {
        RenderElementState {
            visible_area: 0,
            presentation_state: RenderElementPresentationState::Rendering { reason: Some(reason) },
            needs_capture: false,
        }
    }
}
#[allow(dead_code)] // This structs purpose is to keep buffer objects alive, most variants won't be read
#[derive(Debug)]
enum ScanoutBuffer<B: Buffer> {
    Wayland(crate::backend::renderer::utils::Buffer),
    Swapchain(Arc<Slot<B>>),
    Cursor(Arc<GbmBuffer>),
}

impl<B: Buffer> Clone for ScanoutBuffer<B> {
    fn clone(&self) -> Self {
        match self {
            Self::Wayland(arg0) => Self::Wayland(arg0.clone()),
            Self::Swapchain(arg0) => Self::Swapchain(arg0.clone()),
            Self::Cursor(arg0) => Self::Cursor(arg0.clone()),
        }
    }
}

impl<B: Buffer> ScanoutBuffer<B> {
    fn acquire_point(
        &self,
        signaled_fence: Option<&Arc<OwnedFd>>,
    ) -> Option<(SyncPoint, Option<Arc<OwnedFd>>)> {
        if let Self::Wayland(buffer) = self {
            // Assume `DrmSyncobjBlocker` is used, so acquire point has already
            // been signaled. Instead of converting with `SyncPoint::from`.
            if buffer.acquire_point().is_some() {
                return Some((SyncPoint::signaled(), signaled_fence.cloned()));
            }
        }
        None
    }
}

impl<B: Buffer> ScanoutBuffer<B> {
    #[inline]
    fn from_underlying_storage(storage: UnderlyingStorage<'_>) -> Option<Self> {
        match storage {
            UnderlyingStorage::Wayland(buffer) => Some(Self::Wayland(buffer.clone())),
            UnderlyingStorage::Memory { .. } => None,
        }
    }
}

enum DrmFramebuffer<F: Framebuffer> {
    Exporter(F),
    Gbm(super::gbm::GbmFramebuffer),
}

impl<F> AsRef<framebuffer::Handle> for DrmFramebuffer<F>
where
    F: Framebuffer,
{
    #[inline]
    fn as_ref(&self) -> &framebuffer::Handle {
        match self {
            DrmFramebuffer::Exporter(e) => e.as_ref(),
            DrmFramebuffer::Gbm(g) => g.as_ref(),
        }
    }
}

impl<F> Framebuffer for DrmFramebuffer<F>
where
    F: Framebuffer,
{
    #[inline]
    fn format(&self) -> drm_fourcc::DrmFormat {
        match self {
            DrmFramebuffer::Exporter(e) => e.format(),
            DrmFramebuffer::Gbm(g) => g.format(),
        }
    }
}

impl<F> std::fmt::Debug for DrmFramebuffer<F>
where
    F: Framebuffer + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Exporter(arg0) => f.debug_tuple("Exporter").field(arg0).finish(),
            Self::Gbm(arg0) => f.debug_tuple("Gbm").field(arg0).finish(),
        }
    }
}

struct DrmScanoutBuffer<B: Buffer, F: Framebuffer> {
    buffer: ScanoutBuffer<B>,
    fb: CachedDrmFramebuffer<F>,
}

impl<B: Buffer, F: Framebuffer> Clone for DrmScanoutBuffer<B, F> {
    fn clone(&self) -> Self {
        DrmScanoutBuffer {
            buffer: self.buffer.clone(),
            fb: self.fb.clone(),
        }
    }
}

impl<B, F> std::fmt::Debug for DrmScanoutBuffer<B, F>
where
    B: Buffer + std::fmt::Debug,
    F: Framebuffer + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DrmScanoutBuffer")
            .field("buffer", &self.buffer)
            .field("fb", &self.fb)
            .finish()
    }
}

impl<B: Buffer, F: Framebuffer> AsRef<framebuffer::Handle> for DrmScanoutBuffer<B, F> {
    #[inline]
    fn as_ref(&self) -> &drm::control::framebuffer::Handle {
        self.fb.as_ref()
    }
}

impl<B: Buffer, F: Framebuffer> Framebuffer for DrmScanoutBuffer<B, F> {
    #[inline]
    fn format(&self) -> drm_fourcc::DrmFormat {
        self.fb.format()
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
enum ElementFramebufferCacheBuffer {
    Wayland(wayland_server::Weak<WlBuffer>),
}

impl ElementFramebufferCacheBuffer {
    #[inline]
    fn from_underlying_storage(storage: &UnderlyingStorage<'_>) -> Option<Self> {
        match storage {
            UnderlyingStorage::Wayland(buffer) => Some(Self::Wayland(buffer.downgrade())),
            UnderlyingStorage::Memory { .. } => None,
        }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct ElementFramebufferCacheKey {
    allow_opaque_fallback: bool,
    buffer: ElementFramebufferCacheBuffer,
}

impl ElementFramebufferCacheKey {
    #[inline]
    fn from_underlying_storage(storage: &UnderlyingStorage<'_>, allow_opaque_fallback: bool) -> Option<Self> {
        let buffer = ElementFramebufferCacheBuffer::from_underlying_storage(storage)?;
        Some(Self {
            allow_opaque_fallback,
            buffer,
        })
    }
}

impl ElementFramebufferCacheKey {
    #[inline]
    fn is_alive(&self) -> bool {
        match self.buffer {
            ElementFramebufferCacheBuffer::Wayland(ref buffer) => buffer.is_alive(),
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq)]
struct PlanesSnapshot {
    primary: bool,
    cursor_bitmask: u32,
    overlay_bitmask: u32,
}

#[derive(Debug)]
struct ElementInstanceState {
    properties: PlaneProperties,
    active_planes: PlanesSnapshot,
    failed_planes: PlanesSnapshot,
}

#[derive(Debug)]
struct ElementState<B: Framebuffer> {
    instances: SmallVec<[ElementInstanceState; 1]>,
    fb_cache: ElementFramebufferCache<B>,
}

#[derive(Debug)]
struct ElementFramebufferCache<B>
where
    B: Framebuffer,
{
    /// Cache for framebuffer handles per cache key (e.g. wayland buffer)
    fb_cache: SmallVec<
        [(
            ElementFramebufferCacheKey,
            Result<CachedDrmFramebuffer<B>, ExportBufferError>,
        ); 4],
    >,
}

impl<B> ElementFramebufferCache<B>
where
    B: Framebuffer,
{
    #[inline]
    fn get(
        &self,
        cache_key: &ElementFramebufferCacheKey,
    ) -> Option<Result<&CachedDrmFramebuffer<B>, ExportBufferError>> {
        self.fb_cache.iter().find_map(|(k, r)| {
            if k == cache_key {
                Some(r.as_ref().map_err(|err| *err))
            } else {
                None
            }
        })
    }

    #[inline]
    fn insert(
        &mut self,
        cache_key: ElementFramebufferCacheKey,
        fb: Result<CachedDrmFramebuffer<B>, ExportBufferError>,
    ) {
        self.fb_cache.push((cache_key, fb));
    }

    fn cleanup(&mut self) {
        self.fb_cache.retain(|(key, _)| key.is_alive());
    }
}

impl<B> Default for ElementFramebufferCache<B>
where
    B: Framebuffer,
{
    #[inline]
    fn default() -> Self {
        Self {
            fb_cache: Default::default(),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
struct PlaneProperties {
    pub src: Rectangle<f64, BufferCoords>,
    pub dst: Rectangle<i32, Physical>,
    pub transform: Transform,
    pub alpha: f32,
    pub format: DrmFormat,
}

impl PlaneProperties {
    #[inline]
    fn is_compatible(&self, other: &PlaneProperties) -> bool {
        self.src == other.src
            && self.dst == other.dst
            && self.transform == other.transform
            && self.alpha == other.alpha
            && self.format == other.format
    }
}

struct ElementPlaneConfig<'a, B: Buffer, F: Framebuffer> {
    z_index: usize,
    geometry: Rectangle<i32, Physical>,
    properties: PlaneProperties,
    buffer: DrmScanoutBuffer<B, F>,
    failed_planes: &'a mut PlanesSnapshot,
}

#[derive(Debug)]
struct PlaneConfig<B: Buffer, F: Framebuffer> {
    pub properties: PlaneProperties,
    pub buffer: DrmScanoutBuffer<B, F>,
    pub damage_clips: Option<PlaneDamageClips>,
    pub plane_claim: PlaneClaim,
    pub sync: Option<(SyncPoint, Option<Arc<OwnedFd>>)>,
}

impl<B: Buffer, F: Framebuffer> PlaneConfig<B, F> {
    #[inline]
    pub fn is_compatible(&self, other: &PlaneConfig<B, F>) -> bool {
        self.properties.is_compatible(&other.properties)
    }
}

impl<B: Buffer, F: Framebuffer> Clone for PlaneConfig<B, F> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            properties: self.properties,
            buffer: self.buffer.clone(),
            damage_clips: self.damage_clips.clone(),
            plane_claim: self.plane_claim.clone(),
            sync: self.sync.clone(),
        }
    }
}

#[derive(Debug, Clone)]
struct PlaneElementState {
    id: Id,
    commit: CommitCounter,
    z_index: usize,
    cursor_size: Option<Size<i32, Physical>>,
}

#[derive(Debug)]
struct PlaneState<B: Buffer, F: Framebuffer> {
    skip: bool,
    needs_test: bool,
    element_state: Option<PlaneElementState>,
    config: Option<PlaneConfig<B, F>>,
}

impl<B: Buffer, F: Framebuffer> Default for PlaneState<B, F> {
    #[inline]
    fn default() -> Self {
        Self {
            skip: true,
            needs_test: false,
            element_state: Default::default(),
            config: Default::default(),
        }
    }
}

impl<B: Buffer, F: Framebuffer> PlaneState<B, F> {
    #[inline]
    fn buffer(&self) -> Option<&DrmScanoutBuffer<B, F>> {
        self.config.as_ref().map(|config| &config.buffer)
    }

    #[inline]
    fn is_compatible(&self, other: &Self) -> bool {
        match (self.config.as_ref(), other.config.as_ref()) {
            (Some(a), Some(b)) => a.is_compatible(b),
            (None, None) => true,
            _ => false,
        }
    }
}

impl<B: Buffer, F: Framebuffer> Clone for PlaneState<B, F> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            skip: self.skip,
            needs_test: self.needs_test,
            element_state: self.element_state.clone(),
            config: self.config.clone(),
        }
    }
}

#[derive(Debug)]
struct FrameState<B: Buffer, F: Framebuffer> {
    planes: SmallVec<[(plane::Handle, PlaneState<B, F>); 10]>,
}

impl<B: Buffer, F: Framebuffer> FrameState<B, F> {
    #[inline]
    fn is_assigned(&self, handle: plane::Handle) -> bool {
        self.planes
            .iter()
            .find_map(|(p, state)| {
                if *p == handle {
                    Some(state.config.is_some())
                } else {
                    None
                }
            })
            .unwrap_or(false)
    }

    #[inline]
    fn overlaps(&self, handle: plane::Handle, element_geometry: Rectangle<i32, Physical>) -> bool {
        self.planes
            .iter()
            .find(|(p, _)| *p == handle)
            .and_then(|(_, state)| {
                state
                    .config
                    .as_ref()
                    .map(|config| config.properties.dst.overlaps(element_geometry))
            })
            .unwrap_or(false)
    }

    #[inline]
    fn plane_state(&self, handle: plane::Handle) -> Option<&PlaneState<B, F>> {
        self.planes
            .iter()
            .find_map(|(p, state)| if *p == handle { Some(state) } else { None })
    }

    #[inline]
    fn plane_state_mut(&mut self, handle: plane::Handle) -> Option<&mut PlaneState<B, F>> {
        self.planes
            .iter_mut()
            .find_map(|(p, state)| if *p == handle { Some(state) } else { None })
    }

    #[inline]
    fn plane_properties(&self, handle: plane::Handle) -> Option<&PlaneProperties> {
        self.plane_state(handle)
            .and_then(|state| state.config.as_ref())
            .map(|config| &config.properties)
    }

    #[inline]
    fn plane_buffer(&self, handle: plane::Handle) -> Option<&DrmScanoutBuffer<B, F>> {
        self.plane_state(handle)
            .and_then(|state| state.config.as_ref().map(|config| &config.buffer))
    }
}

impl<B: Buffer, F: Framebuffer> FrameState<B, F> {
    fn from_planes(primary_plane: plane::Handle, planes: &Planes, include_cursor_planes: bool) -> Self {
        let mut tmp = SmallVec::with_capacity(planes.overlay.len() + planes.cursor.len() + 1);
        tmp.push((primary_plane, PlaneState::default()));
        if include_cursor_planes {
            tmp.extend(
                planes
                    .cursor
                    .iter()
                    .map(|info| (info.handle, PlaneState::default())),
            );
        }
        tmp.extend(
            planes
                .overlay
                .iter()
                .map(|info| (info.handle, PlaneState::default())),
        );

        FrameState { planes: tmp }
    }

    fn add_cursor_planes(&mut self, planes: &Planes) {
        for info in &planes.cursor {
            if self.plane_state(info.handle).is_none() {
                self.planes.push((info.handle, PlaneState::default()));
            }
        }
    }
}

impl<B: Buffer, F: Framebuffer> FrameState<B, F> {
    #[profiling::function]
    #[inline]
    fn set_state(&mut self, plane: plane::Handle, state: PlaneState<B, F>) {
        let current_config = match self.plane_state_mut(plane) {
            Some(config) => config,
            None => return,
        };
        *current_config = state;
    }

    #[profiling::function]
    fn test_state(
        &mut self,
        surface: &DrmSurface,
        supports_fencing: bool,
        plane: plane::Handle,
        state: PlaneState<B, F>,
        allow_modeset: bool,
    ) -> Result<(), DrmError> {
        let current_config = match self.plane_state_mut(plane) {
            Some(config) => config,
            None => return Ok(()),
        };
        let backup = current_config.clone();
        *current_config = state;

        let res = surface.test_state(self.build_planes(surface, supports_fencing, true), allow_modeset);

        if res.is_err() {
            // test failed, restore previous state
            *self.plane_state_mut(plane).unwrap() = backup;
        } else {
            self.planes
                .iter_mut()
                .for_each(|(_, state)| state.needs_test = false);
        }

        res
    }

    #[profiling::function]
    fn test_state_complete(
        &mut self,
        previous_frame: &Self,
        surface: &DrmSurface,
        supports_fencing: bool,
        allow_modeset: bool,
        allow_partial_update: bool,
    ) -> Result<(), DrmError> {
        let needs_test = self.planes.iter().any(|(_, state)| state.needs_test);
        let is_fully_compatible = self.planes.iter().all(|(handle, state)| {
            previous_frame
                .plane_state(*handle)
                .map(|other| state.is_compatible(other))
                .unwrap_or(false)
        });

        if allow_partial_update && (!needs_test || is_fully_compatible) {
            trace!("skipping fully compatible state test");
            self.planes
                .iter_mut()
                .for_each(|(_, state)| state.needs_test = false);
            return Ok(());
        }

        let res = surface.test_state(
            self.build_planes(surface, supports_fencing, allow_partial_update),
            allow_modeset,
        );

        if res.is_ok() {
            self.planes
                .iter_mut()
                .for_each(|(_, state)| state.needs_test = false);
        }

        res
    }

    #[profiling::function]
    fn commit(
        &mut self,
        surface: &DrmSurface,
        supports_fencing: bool,
        allow_partial_update: bool,
        event: bool,
    ) -> Result<(), crate::backend::drm::error::Error> {
        debug_assert!(!self.planes.iter().any(|(_, state)| state.needs_test));
        surface.commit(
            self.build_planes(surface, supports_fencing, allow_partial_update),
            event,
        )
    }

    #[profiling::function]
    fn page_flip(
        &mut self,
        surface: &DrmSurface,
        supports_fencing: bool,
        allow_partial_update: bool,
        event: bool,
        async_flip: bool,
    ) -> Result<PresentationMode, crate::backend::drm::error::Error> {
        debug_assert!(!self.planes.iter().any(|(_, state)| state.needs_test));
        surface.page_flip(
            self.build_planes(surface, supports_fencing, allow_partial_update),
            event,
            async_flip,
        )
    }

    #[profiling::function]
    fn build_planes<'a>(
        &'a mut self,
        surface: &'a DrmSurface,
        supports_fencing: bool,
        allow_partial_update: bool,
    ) -> impl IntoIterator<Item = super::PlaneState<'a>> {
        for (_, state) in self.planes.iter_mut().filter(|(_, state)| !state.skip) {
            if let Some(config) = state.config.as_mut() {
                // Try to extract a native fence out of the supplied sync point if any
                // If the sync point has no native fence or the surface does not support
                // fencing force a wait
                if let Some((sync, fence)) = config.sync.as_mut() {
                    if supports_fencing && fence.is_none() {
                        *fence = sync.export().map(Arc::new);
                    }
                }
            }
        }

        self.planes
            .iter_mut()
            .filter(move |(handle, state)| {
                // If we are not allowed to do an partial update we want to update all
                // planes we can claim. This makes sure we also reset planes we never
                // actually used. We can skip getting a claim here if we have a
                // config as this means we already claimed the plane for us.
                if allow_partial_update {
                    // A partial update would technically only have to include planes that
                    // actually changed. This includes planes we previously used and have to
                    // reset and planes we use and want to update.
                    // Both is already encoded into state.skip, so this should be the only
                    // thing we have to consider here.
                    //
                    // But...Unfortunately some drivers seem to have issues with partial
                    // updates, at least when it does not contain the primary plane, resulting
                    // in strange issues like e.g. repeating plane content, side-scrolling planes,
                    // wrapping planes around edges...
                    //
                    // So until these things are fixed just always send the whole state. We do not
                    // have to send planes we never used, but we include planes we want to reset or
                    // that explicitly changed represented by !state.skip and all planes currently in
                    // use represented by having an config defined.
                    !state.skip || state.config.is_some()
                } else {
                    state.config.is_some() || surface.claim_plane(*handle).is_some()
                }
            })
            .map(move |(handle, state)| super::surface::PlaneState {
                handle: *handle,
                config: state.config.as_mut().map(|config| super::PlaneConfig {
                    src: config.properties.src,
                    dst: config.properties.dst,
                    alpha: config.properties.alpha,
                    transform: config.properties.transform,
                    damage_clips: config.damage_clips.as_ref().map(|d| d.blob()),
                    fb: *config.buffer.as_ref(),
                    fence: config
                        .sync
                        .as_ref()
                        .and_then(|(_, fence)| fence.as_ref().map(|fence| fence.as_fd())),
                }),
            })
    }
}

type CompositorFrameState<A, F> =
    FrameState<<A as Allocator>::Buffer, <F as ExportFramebuffer<<A as Allocator>::Buffer>>::Framebuffer>;

type FrameErrorType<A, F> = FrameError<
    <A as Allocator>::Error,
    <<A as Allocator>::Buffer as AsDmabuf>::Error,
    <F as ExportFramebuffer<<A as Allocator>::Buffer>>::Error,
>;

pub(crate) type FrameResult<T, A, F> = Result<T, FrameErrorType<A, F>>;

pub(crate) type RenderFrameErrorType<A, F, R> = RenderFrameError<
    <A as Allocator>::Error,
    <<A as Allocator>::Buffer as AsDmabuf>::Error,
    <F as ExportFramebuffer<<A as Allocator>::Buffer>>::Error,
    <R as RendererSuper>::Error,
>;

#[derive(Debug)]
struct CursorState<G: AsFd + 'static> {
    allocator: GbmAllocator<G>,
    framebuffer_exporter: GbmFramebufferExporter<G>,
    previous_output_transform: Option<Transform>,
    previous_output_scale: Option<Scale<f64>>,
    legacy: LegacyCursorState,
    #[cfg(feature = "renderer_pixman")]
    pixman_renderer: Option<PixmanRenderer>,
}

/// Capability token for moving one installed legacy cursor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LegacyCursorToken(u64);

/// Identity and placement of a cursor presented through legacy DRM ioctls.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LegacyCursorPresentation {
    /// Token accepted by [`DrmCompositor::move_legacy_cursor`].
    pub token: LegacyCursorToken,
    /// Current physical cursor-plane origin.
    pub physical_origin: Point<i32, Physical>,
    /// Render element presented by the cursor plane.
    pub element_id: Id,
    /// Commit presented by the cursor plane.
    pub commit: CommitCounter,
    /// Buffer size installed on the legacy cursor plane.
    pub plane_size: Size<i32, Physical>,
}

/// Result of a cursor-only legacy DRM move.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LegacyCursorMoveResult {
    /// The cursor moved and this is its updated presentation identity.
    Moved(LegacyCursorPresentation),
    /// The token belongs to an earlier cursor installation.
    StaleToken,
    /// No legacy cursor is currently installed.
    NotActive,
    /// The driver permanently rejected cursor-only legacy moves.
    Rejected,
}

#[derive(Debug)]
enum LegacyCursorOwnership {
    Candidate,
    Active(LegacyCursorActive),
    Disabled,
    Atomic,
}

enum LegacyCursorAssignment {
    Presented(LegacyCursorPresentation),
    Software,
    Atomic,
}

trait LegacyCursorIo<B> {
    fn disable(&self) -> std::io::Result<()>;
    fn install(&self, buffer: &B) -> std::io::Result<()>;
    fn move_to(&self, origin: Point<i32, Physical>) -> std::io::Result<()>;
}

struct SurfaceLegacyCursorIo<'a>(&'a DrmSurface);

#[allow(deprecated)]
impl<B: drm::buffer::Buffer> LegacyCursorIo<B> for SurfaceLegacyCursorIo<'_> {
    fn disable(&self) -> std::io::Result<()> {
        self.0
            .device_fd()
            .set_cursor2::<B>(self.0.crtc(), None, (0, 0))
    }

    fn install(&self, buffer: &B) -> std::io::Result<()> {
        self.0
            .device_fd()
            .set_cursor2(self.0.crtc(), Some(buffer), (0, 0))
    }

    fn move_to(&self, origin: Point<i32, Physical>) -> std::io::Result<()> {
        self.0.device_fd().move_cursor(self.0.crtc(), origin.into())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LegacyCursorInstallResult {
    Installed,
    PreserveActive,
    DisabledSoftware,
    CandidateSoftware,
    Atomic,
}

fn install_legacy_cursor<B>(
    io: &impl LegacyCursorIo<B>,
    buffer: &B,
    origin: Point<i32, Physical>,
    replacing_active: bool,
    legacy_succeeded: bool,
) -> LegacyCursorInstallResult {
    if replacing_active && io.disable().is_err() {
        return LegacyCursorInstallResult::PreserveActive;
    }
    if legacy_succeeded && io.move_to(origin).is_err() {
        return LegacyCursorInstallResult::DisabledSoftware;
    }
    if let Err(error) = io.install(buffer) {
        return if !legacy_succeeded && classify_legacy_cursor_error(&error) == LegacyCursorIoError::Permanent {
            LegacyCursorInstallResult::Atomic
        } else if legacy_succeeded {
            LegacyCursorInstallResult::DisabledSoftware
        } else {
            LegacyCursorInstallResult::CandidateSoftware
        };
    }
    if !legacy_succeeded && io.move_to(origin).is_err() {
        let _ = io.disable();
        return LegacyCursorInstallResult::DisabledSoftware;
    }
    LegacyCursorInstallResult::Installed
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LegacyCursorIoError {
    Permanent,
    Inactive,
    Busy,
}

fn classify_legacy_cursor_error(error: &std::io::Error) -> LegacyCursorIoError {
    match error.raw_os_error() {
        Some(22 | 25 | 38 | 95) => LegacyCursorIoError::Permanent,
        Some(1 | 13) => LegacyCursorIoError::Inactive,
        _ => LegacyCursorIoError::Busy,
    }
}

/// Calculate the physical cursor-plane origin used by both atomic and legacy cursor paths.
pub fn cursor_plane_location(
    element_location: Point<i32, Physical>,
    cursor_plane_size: Size<i32, Physical>,
    output_geometry: Rectangle<i32, Physical>,
    output_transform: Transform,
) -> Point<i32, Physical> {
    output_transform.transform_point_in(element_location, &output_geometry.size)
        - output_transform.transform_point_in(Point::default(), &cursor_plane_size)
}

#[derive(Debug)]
struct LegacyCursorActive {
    _buffer: Arc<GbmBuffer>,
    presentation: LegacyCursorPresentation,
    element_size: Size<i32, Physical>,
    output_scale: Scale<f64>,
    output_transform: Transform,
}

#[derive(Debug)]
struct LegacyCursorState {
    ownership: LegacyCursorOwnership,
    next_token: u64,
    legacy_succeeded: bool,
    fast_motion_permanently_rejected: bool,
}

impl LegacyCursorState {
    fn new(policy: CursorPlanePolicy) -> Self {
        Self {
            ownership: if policy.reserves_legacy_cursor() {
                LegacyCursorOwnership::Candidate
            } else {
                LegacyCursorOwnership::Atomic
            },
            next_token: 1,
            legacy_succeeded: false,
            fast_motion_permanently_rejected: false,
        }
    }

    fn invalidate(&mut self, policy: CursorPlanePolicy) {
        self.ownership = if policy.reserves_legacy_cursor() {
            LegacyCursorOwnership::Candidate
        } else {
            LegacyCursorOwnership::Atomic
        };
        self.legacy_succeeded = false;
        self.fast_motion_permanently_rejected = false;
        self.next_token = self.next_token.wrapping_add(1).max(1);
    }

    fn token(&mut self) -> LegacyCursorToken {
        let token = LegacyCursorToken(self.next_token);
        self.next_token = self.next_token.wrapping_add(1).max(1);
        token
    }
}

#[derive(Debug, thiserror::Error, Copy, Clone)]
enum ExportBufferError {
    #[error("the buffer has no underlying storage")]
    NoUnderlyingStorage,
    #[error("exporting the framebuffer failed")]
    ExportFailed,
    #[error("no framebuffer could be exported")]
    Unsupported,
}

impl From<ExportBufferError> for Option<RenderingReason> {
    #[inline]
    fn from(err: ExportBufferError) -> Self {
        if matches!(err, ExportBufferError::ExportFailed) {
            // Export failed could mean the buffer could
            // not be used to add a drm framebuffer. This
            // especially can happen on kmsro devices where
            // a buffer format not usable for scan-out can
            // not be used to add a framebuffer
            // We can try to give the client another chance
            // by announcing a scan-out tranche
            Some(RenderingReason::ScanoutFailed)
        } else {
            // We provide no reason for rendering here as there
            // is no action that can be taken to make it work
            None
        }
    }
}

#[derive(Debug)]
struct OverlayPlaneElementIds {
    plane_ids: Vec<(plane::Handle, Id, Id)>,
}

impl OverlayPlaneElementIds {
    fn from_planes(planes: &Planes) -> Self {
        let overlay_plane_count = planes.overlay.len();

        Self {
            plane_ids: Vec::with_capacity(overlay_plane_count),
        }
    }

    fn plane_id_for_element_id(&mut self, plane: &plane::Handle, element_id: &Id) -> Id {
        // Either get the existing plane id for the plane when the stored element id
        // matches or generate a new Id (and update the element id)
        let existing = self.plane_ids.iter_mut().find(|(p, _, _)| p == plane);
        if let Some((_, plane_id, current_element_id)) = existing {
            if current_element_id != element_id {
                *plane_id = Id::new();
                *current_element_id = element_id.clone();
            }

            plane_id.clone()
        } else {
            let plane_id = Id::new();

            self.plane_ids
                .push((*plane, plane_id.clone(), element_id.clone()));

            plane_id
        }
    }

    fn contains_plane_id(&self, plane_id: &Id) -> bool {
        self.plane_ids.iter().any(|(_, p, _)| p == plane_id)
    }

    fn remove_plane(&mut self, plane: &plane::Handle) {
        self.plane_ids.retain(|(p, _, _)| p != plane);
    }
}

struct PlaneAssignment {
    handle: plane::Handle,
    type_: PlaneType,
}

impl From<&PlaneInfo> for PlaneAssignment {
    #[inline]
    fn from(value: &PlaneInfo) -> Self {
        PlaneAssignment {
            handle: value.handle,
            type_: value.type_,
        }
    }
}

struct PendingFrame<A: Allocator, F: ExportFramebuffer<<A as Allocator>::Buffer>, U> {
    frame: CompositorFrameState<A, F>,
    user_data: U,
    /// The mode this frame's flip actually ran in (async vs the vsync fallback), stamped at
    /// submit time and read back once the flip completes (DRIFT-984).
    presentation_mode: PresentationMode,
}

impl<A, F, U> std::fmt::Debug for PendingFrame<A, F, U>
where
    A: Allocator,
    <A as Allocator>::Buffer: std::fmt::Debug,
    F: ExportFramebuffer<<A as Allocator>::Buffer>,
    <F as ExportFramebuffer<<A as Allocator>::Buffer>>::Framebuffer: std::fmt::Debug,
    U: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PendingFrame")
            .field("frame", &self.frame)
            .field("user_data", &self.user_data)
            .finish()
    }
}

struct QueuedFrame<A: Allocator, F: ExportFramebuffer<<A as Allocator>::Buffer>, U> {
    prepared_frame: PreparedFrame<A, F>,
    user_data: U,
}

impl<A, F, U> std::fmt::Debug for QueuedFrame<A, F, U>
where
    A: Allocator,
    <A as Allocator>::Buffer: std::fmt::Debug,
    F: ExportFramebuffer<<A as Allocator>::Buffer>,
    <F as ExportFramebuffer<<A as Allocator>::Buffer>>::Framebuffer: std::fmt::Debug,
    U: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueuedFrame")
            .field("prepared_frame", &self.prepared_frame)
            .field("user_data", &self.user_data)
            .finish()
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum PreparedFrameKind {
    Full,
    Partial,
}

struct PreparedFrame<A: Allocator, F: ExportFramebuffer<<A as Allocator>::Buffer>> {
    frame: CompositorFrameState<A, F>,
    kind: PreparedFrameKind,
}

impl<A: Allocator, F: ExportFramebuffer<<A as Allocator>::Buffer>> PreparedFrame<A, F> {
    #[inline]
    fn is_empty(&self) -> bool {
        // It can happen that we have no changes, but there is a pending commit or
        // we are forced to do a full update in which case we just set the previous state again
        self.kind == PreparedFrameKind::Partial && self.frame.planes.iter().all(|p| p.1.skip)
    }
}

impl<A, F> std::fmt::Debug for PreparedFrame<A, F>
where
    A: Allocator,
    <A as Allocator>::Buffer: std::fmt::Debug,
    F: ExportFramebuffer<<A as Allocator>::Buffer>,
    <F as ExportFramebuffer<<A as Allocator>::Buffer>>::Framebuffer: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedFrame")
            .field("frame", &self.frame)
            .field("kind", &self.kind)
            .finish()
    }
}

bitflags::bitflags! {
    /// Possible flags for a DMA buffer
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct FrameFlags: u32 {
        /// Allow to realize the frame by scanning out elements on the primary plane
        /// with the same pixel format as the main swapchain
        const ALLOW_PRIMARY_PLANE_SCANOUT = 1;
        /// Allow to realize the frame by scanning out elements on the primary plane
        /// regardless of their format
        const ALLOW_PRIMARY_PLANE_SCANOUT_ANY = 2;
        /// Allow to realize the frame by scanning out elements on overlay planes
        const ALLOW_OVERLAY_PLANE_SCANOUT = 4;
        /// Allow to realize the frame by scanning out elements on cursor planes
        const ALLOW_CURSOR_PLANE_SCANOUT = 8;
        /// Return `EmptyFrame`, if only the cursor plane would have been updated
        const SKIP_CURSOR_ONLY_UPDATES = 16;
        /// Allow to realize a cursor element through legacy cursor ioctls.
        const ALLOW_LEGACY_CURSOR = 32;
        /// Allow to realize the frame by assigning elements on any plane
        const ALLOW_SCANOUT = Self::ALLOW_PRIMARY_PLANE_SCANOUT.bits() | Self::ALLOW_OVERLAY_PLANE_SCANOUT.bits() | Self::ALLOW_CURSOR_PLANE_SCANOUT.bits();
        /// Safe default set of flags
        const DEFAULT = Self::ALLOW_SCANOUT.bits() | Self::ALLOW_LEGACY_CURSOR.bits();
    }
}

/// Composite an output using a combination of planes and rendering
///
/// see the [`module docs`](crate::backend::drm::compositor) for more information
#[derive(Debug)]
pub struct DrmCompositor<A, F, U, G>
where
    A: Allocator,
    F: ExportFramebuffer<A::Buffer>,
    <F as ExportFramebuffer<A::Buffer>>::Framebuffer: std::fmt::Debug + Send + Sync + 'static,
    G: AsFd + 'static,
{
    output_mode_source: OutputModeSource,
    surface: Arc<DrmSurface>,
    planes: Planes,
    overlay_plane_element_ids: OverlayPlaneElementIds,
    damage_tracker: OutputDamageTracker,
    primary_is_opaque: bool,
    primary_plane_element_id: Id,
    primary_plane_damage_bag: DamageBag<i32, BufferCoords>,
    supports_fencing: bool,
    reset_pending: bool,
    signaled_fence: Option<Arc<OwnedFd>>,
    /// Requested pacing for the next flip (sticky, DRIFT-984). Consulted in [`Self::submit`].
    presentation_mode: PresentationMode,

    framebuffer_exporter: F,

    current_frame: CompositorFrameState<A, F>,
    pending_frame: Option<PendingFrame<A, F, U>>,
    queued_frame: Option<QueuedFrame<A, F, U>>,
    next_frame: Option<PreparedFrame<A, F>>,

    swapchain: Swapchain<A>,

    cursor_size: Size<i32, Physical>,
    cursor_state: Option<CursorState<G>>,
    cursor_plane_policy: CursorPlanePolicy,

    element_states: IndexMap<Id, ElementState<<F as ExportFramebuffer<A::Buffer>>::Framebuffer>>,
    previous_element_states: IndexMap<Id, ElementState<<F as ExportFramebuffer<A::Buffer>>::Framebuffer>>,
    opaque_regions: Vec<Rectangle<i32, Physical>>,
    element_opaque_regions_workhouse: Vec<Rectangle<i32, Physical>>,
    /// Reused across frames so the effect scan allocates nothing in steady state.
    /// Cleared and refilled at the top of every `render_frame`, so it carries no
    /// state between frames; only its capacity survives.
    framebuffer_effect_regions_workhouse: Vec<(usize, Rectangle<i32, Physical>)>,

    debug_flags: DebugFlags,
    span: tracing::Span,
}

/// Tearing-control accessors (DRIFT-984). These are plain field/surface reads that do not need
/// the allocator/exporter bounds of the main `impl`, so they live in a minimal block that the
/// equally-light [`DrmOutput`](super::DrmOutput) passthroughs can call.
impl<A, F, U, G> DrmCompositor<A, F, U, G>
where
    A: Allocator,
    F: ExportFramebuffer<A::Buffer>,
    <F as ExportFramebuffer<A::Buffer>>::Framebuffer: std::fmt::Debug + Send + Sync + 'static,
    G: AsFd + 'static,
{
    /// Whether the driver supports async (tearing) page flips for the underlying surface.
    ///
    /// [`set_presentation_mode`](DrmCompositor::set_presentation_mode) with
    /// [`PresentationMode::Async`] only tears when this is `true`; otherwise it degrades to
    /// vsync.
    pub fn supports_async_page_flip(&self) -> bool {
        self.surface.supports_async_page_flip()
    }

    /// The pacing requested for the next flip. See
    /// [`set_presentation_mode`](DrmCompositor::set_presentation_mode).
    pub fn presentation_mode(&self) -> PresentationMode {
        self.presentation_mode
    }

    /// Requests a pacing mode for subsequent flips (sticky until changed).
    ///
    /// [`PresentationMode::Async`] asks for tearing (immediate) page flips. It only takes effect
    /// on a plain page flip (never a modeset commit) and only when the driver supports async
    /// flips; a frame that cannot fly async silently runs vsynced instead. Read
    /// [`pending_presentation_mode`](DrmCompositor::pending_presentation_mode) in the vblank
    /// handler BEFORE [`frame_submitted`](DrmCompositor::frame_submitted) for what the completing
    /// flip actually did.
    pub fn set_presentation_mode(&mut self, mode: PresentationMode) {
        self.presentation_mode = mode;
    }

    /// The mode the currently pending (in-flight) flip actually ran in, i.e. the frame that the
    /// next [`frame_submitted`](DrmCompositor::frame_submitted) will acknowledge.
    ///
    /// Read this in the vblank handler BEFORE calling `frame_submitted` to classify the flip that
    /// just completed by what really happened (a frame that requested async but fell back reports
    /// [`PresentationMode::Vsync`]), so pacing and presentation-feedback key off the real outcome
    /// rather than the current requested intent, which may have changed since the flip was queued.
    /// Returns [`PresentationMode::Vsync`] when no flip is pending.
    pub fn pending_presentation_mode(&self) -> PresentationMode {
        self.pending_frame
            .as_ref()
            .map(|frame| frame.presentation_mode)
            .unwrap_or(PresentationMode::Vsync)
    }

    /// Invalidate tokens and retained cursor buffers across a DRM lifecycle boundary.
    pub fn invalidate_legacy_cursor_lifecycle(&mut self) {
        if let Some(cursor_state) = self.cursor_state.as_mut() {
            cursor_state.legacy.invalidate(self.cursor_plane_policy);
        }
    }

    /// Move an active legacy cursor without submitting an atomic frame.
    #[allow(deprecated)]
    pub fn move_legacy_cursor(
        &mut self,
        token: LegacyCursorToken,
        physical_origin: Point<i32, Physical>,
    ) -> LegacyCursorMoveResult {
        let Some(cursor_state) = self.cursor_state.as_mut() else {
            return LegacyCursorMoveResult::NotActive;
        };
        if cursor_state.legacy.fast_motion_permanently_rejected {
            return LegacyCursorMoveResult::Rejected;
        }
        let LegacyCursorOwnership::Active(active) = &mut cursor_state.legacy.ownership else {
            return LegacyCursorMoveResult::NotActive;
        };
        if active.presentation.token != token {
            return LegacyCursorMoveResult::StaleToken;
        }

        match self
            .surface
            .device_fd()
            .move_cursor(self.surface.crtc(), physical_origin.into())
        {
            Ok(()) => {
                active.presentation.physical_origin = physical_origin;
                LegacyCursorMoveResult::Moved(active.presentation.clone())
            }
            Err(error) => {
                if classify_legacy_cursor_error(&error) == LegacyCursorIoError::Permanent {
                    cursor_state.legacy.fast_motion_permanently_rejected = true;
                    LegacyCursorMoveResult::Rejected
                } else {
                    LegacyCursorMoveResult::NotActive
                }
            }
        }
    }
}

/// Collect the framebuffer-effect elements of a frame into `out`, paired with the
/// geometry each one samples, in the order given.
///
/// Split out from its call site so a test can drive it directly: the ordering and
/// the filter are the whole rule, and a `position`/`rposition`-shaped mistake here
/// would keep only one effect and let everything between two stacked effects
/// through.
///
/// Fills a caller-owned buffer rather than returning a fresh collection: the caller
/// keeps the `Vec` across frames and its capacity settles, so this scan costs no
/// allocation in steady state. Returning a `SmallVec` instead was a per-frame heap
/// allocation as soon as three effects were on screen. That is one fewer per-frame
/// allocation and matches `element_opaque_regions_workhouse` beside it; it does not
/// on its own make the whole of `render_frame` allocation-free, which still builds
/// several sized collections per frame.
///
/// A module-level free function, deliberately not an associated function on
/// [`DrmCompositor`]: nothing here mentions `Self` or any of the four generic
/// parameters, and reaching it through `Self::` would force a test to name a
/// concrete instantiation satisfying every bound on that impl just to pass a
/// `usize` and two rectangles. The path of least resistance at that point is to
/// re-implement the rule inside the test, which is exactly what splitting it out
/// is meant to prevent.
fn collect_framebuffer_effect_regions(
    out: &mut Vec<(usize, Rectangle<i32, Physical>)>,
    elements: impl IntoIterator<Item = (bool, Rectangle<i32, Physical>)>,
) {
    out.clear();
    out.extend(
        elements
            .into_iter()
            .enumerate()
            .filter(|(_, (is_effect, _))| *is_effect)
            .map(|(index, (_, geometry))| (index, geometry)),
    );
}

/// Whether the element at `index` sits BELOW a framebuffer effect it overlaps.
///
/// `elements` for a frame are ordered front-to-back (see [`DrmCompositor::render_frame`]),
/// so an element sits below another exactly when its index is greater. Such an
/// element must be composited into the primary framebuffer over the part it shares
/// with the effect, because that framebuffer is what the effect samples: a plane
/// assignment would replace it with a hole punch (underlay) or with hardware
/// composition above the primary plane (overlay, cursor), and either way the effect
/// reads pixels that are not there.
///
/// The region is the effect's element geometry, the same region
/// [`OutputDamageTracker`] calls "behind the element" when it decides what a
/// framebuffer effect must have available to capture.
///
/// `>` and not `>=`: the effect element passes its own barrier here and is refused
/// just below, by the `is_framebuffer_effect` early-out in
/// [`DrmCompositor::try_assign_element`]. The two are one rule.
///
/// `overlaps` and not `overlaps_or_touches`: an edge-adjacent element contributes
/// no pixels inside the effect's region. Note this is adjacency and not area, since
/// `overlaps` is true for a zero-area rectangle strictly inside another, so
/// degenerate geometry is suppressed rather than exempted. That errs toward
/// compositing, which is the safe direction.
///
/// This does not close the boundary in general, and `overlaps_or_touches` would not
/// either. The region here IS the element geometry, so a consumer whose capture
/// reaches outside its own geometry is outside what any geometry-keyed rule can
/// promise. Drift's is bounded to a sub-pixel row under fractional radius and
/// fractional output scale; see its renderer-gles trap notes.
///
/// Note for anyone adding a scanout-candidate heuristic: a candidate ordered BELOW
/// a framebuffer effect it overlaps will never be promoted, by design. Buying that
/// back needs the effect to declare its sampled region rather than inheriting its
/// geometry.
///
/// A candidate ABOVE an effect it overlaps is untouched by this rule, and does not
/// endanger that effect: the effect is BELOW the candidate, so what it samples is
/// what sits behind itself, which the candidate is not. The case that would hurt is
/// the other one, an effect above a promoted candidate it overlaps, and that is
/// exactly what this function refuses. Captures are safe for a second reason too:
/// `capture_framebuffer` runs inline just before its own element draws, while a hole
/// punch draws last, so it can never poison a capture.
///
/// Promotion is still not free, but for an ordinary underlay reason rather than an
/// effect one. Only an UNDERLAY assignment produces a `HolepunchRenderElement` (an
/// overlay gets an `OverlayPlaneElement`), and that hole punch is inserted at the
/// FRONT of the render list, so it draws last and clears its rect over whatever
/// non-effect primary-plane elements were painted above the promoted candidate.
/// That is upstream underlay behaviour, unchanged here.
fn below_framebuffer_effect(
    index: usize,
    geometry: Rectangle<i32, Physical>,
    effect_regions: &[(usize, Rectangle<i32, Physical>)],
) -> bool {
    effect_regions
        .iter()
        .any(|(effect_index, effect_geometry)| {
            index > *effect_index && effect_geometry.overlaps(geometry)
        })
}

impl<A, F, U, G> DrmCompositor<A, F, U, G>
where
    A: Allocator,
    <A as Allocator>::Error: std::error::Error + Send + Sync,
    <A as Allocator>::Buffer: AsDmabuf,
    <A::Buffer as AsDmabuf>::Error: std::error::Error + Send + Sync + std::fmt::Debug,
    F: ExportFramebuffer<A::Buffer>,
    <F as ExportFramebuffer<A::Buffer>>::Framebuffer: std::fmt::Debug + Send + Sync + 'static,
    <F as ExportFramebuffer<A::Buffer>>::Error: std::error::Error + Send + Sync,
    G: AsFd + Clone,
{
    /// Initialize a new [`DrmCompositor`].
    ///
    /// The [`OutputModeSource`] can be created from an [`Output`](crate::output::Output), which will automatically track
    /// the output's mode changes. An [`OutputModeSource::Static`] variant should only be used when
    /// manually updating modes using [`DrmCompositor::set_output_mode_source`].
    ///
    /// - `output_mode_source` is used to determine the current mode, scale and transform
    /// - `surface` for the compositor to use
    /// - `planes` defines which planes the compositor is allowed to use for direct scan-out.
    ///           `None` will result in the compositor to use all planes as specified by [`DrmSurface::planes`]
    /// - `allocator` used for the primary plane swapchain
    /// - `color_formats` are tested in order until a working configuration is found
    /// - `renderer_formats` as reported by the used renderer, used to build the intersection between
    ///                      the possible scan-out formats of the primary plane and the renderer
    /// - `framebuffer_exporter` is used to create drm framebuffers for the swapchain buffers (and if possible
    ///                          for element buffers) for scan-out
    /// - `cursor_size` as reported by the drm device, used for creating buffer for the cursor plane
    /// - `gbm` device used for creating buffers for the cursor plane, `None` will disable the cursor plane
    #[allow(clippy::too_many_arguments)]
    #[instrument(skip_all)]
    pub fn new(
        output_mode_source: impl Into<OutputModeSource> + Debug,
        surface: DrmSurface,
        planes: Option<Planes>,
        allocator: A,
        framebuffer_exporter: F,
        color_formats: impl IntoIterator<Item = DrmFourcc>,
        renderer_formats: impl IntoIterator<Item = DrmFormat>,
        cursor_size: Size<u32, BufferCoords>,
        gbm: Option<GbmDevice<G>>,
    ) -> FrameResult<Self, A, F> {
        Self::new_with_cursor_plane_policy(
            output_mode_source,
            surface,
            planes,
            allocator,
            framebuffer_exporter,
            color_formats,
            renderer_formats,
            cursor_size,
            gbm,
            CursorPlanePolicy::Atomic,
        )
    }

    /// Initialize a new compositor with an explicit cursor-plane ownership policy.
    #[allow(clippy::too_many_arguments)]
    #[instrument(skip_all)]
    pub fn new_with_cursor_plane_policy(
        output_mode_source: impl Into<OutputModeSource> + Debug,
        surface: DrmSurface,
        planes: Option<Planes>,
        mut allocator: A,
        framebuffer_exporter: F,
        color_formats: impl IntoIterator<Item = DrmFourcc>,
        renderer_formats: impl IntoIterator<Item = DrmFormat>,
        cursor_size: Size<u32, BufferCoords>,
        gbm: Option<GbmDevice<G>>,
        cursor_plane_policy: CursorPlanePolicy,
    ) -> FrameResult<Self, A, F> {
        let signaled_fence = match surface.create_syncobj(true) {
            Ok(signaled_syncobj) => match surface.syncobj_to_fd(signaled_syncobj, true) {
                Ok(signaled_fence) => {
                    let _ = surface.destroy_syncobj(signaled_syncobj);
                    Some(Arc::new(signaled_fence))
                }
                Err(err) => {
                    tracing::warn!(?err, "failed to export signaled syncobj");
                    let _ = surface.destroy_syncobj(signaled_syncobj);
                    None
                }
            },
            Err(err) => {
                tracing::warn!(?err, "failed to create signaled syncobj");
                None
            }
        };

        let span = info_span!(
            parent: None,
            "drm_compositor",
            device = ?surface.dev_path(),
            crtc = ?surface.crtc(),
        );

        let output_mode_source = output_mode_source.into();
        let renderer_formats = renderer_formats.into_iter().collect::<Vec<_>>();

        let mut error = None;
        let surface = Arc::new(surface);
        let mut planes = match planes {
            Some(planes) => planes,
            None => surface.planes().clone(),
        };

        // We do not support direct scan-out on legacy
        if surface.is_legacy() {
            planes.cursor.clear();
            planes.overlay.clear();
        }

        // The selection algorithm expects the planes to be ordered form front to back
        planes
            .overlay
            .sort_by_key(|p| std::cmp::Reverse(p.zpos.unwrap_or_default()));

        let driver = surface.get_driver().map_err(|err| {
            FrameError::DrmError(DrmError::Access(AccessError {
                errmsg: "Failed to query drm driver",
                dev: surface.dev_path(),
                source: err,
            }))
        })?;
        // `IN_FENCE_FD` makes commit fail on Nvidia driver
        // https://github.com/NVIDIA/open-gpu-kernel-modules/issues/622
        let is_nvidia = driver.name().to_string_lossy().to_lowercase().contains("nvidia")
            || driver
                .description()
                .to_string_lossy()
                .to_lowercase()
                .contains("nvidia");

        let cursor_size = Size::from((cursor_size.w as i32, cursor_size.h as i32));
        let damage_tracker = OutputDamageTracker::from_mode_source(output_mode_source.clone());
        let supports_fencing = !surface.is_legacy()
            && surface
                .get_driver_capability(DriverCapability::SyncObj)
                .map(|val| val != 0)
                .map_err(|err| {
                    FrameError::DrmError(DrmError::Access(AccessError {
                        errmsg: "Failed to query driver capability",
                        dev: surface.dev_path(),
                        source: err,
                    }))
                })?
            && plane_has_property(&*surface, surface.plane(), "IN_FENCE_FD")?
            && !(is_nvidia && nvidia_drm_version().unwrap_or((0, 0, 0)) < (560, 35, 3));

        for format in color_formats {
            debug!("Testing color format: {}", format);
            match Self::find_supported_format(
                surface.clone(),
                supports_fencing,
                &planes,
                !cursor_plane_policy.reserves_legacy_cursor(),
                allocator,
                &framebuffer_exporter,
                renderer_formats.clone(),
                format,
            ) {
                Ok((swapchain, is_opaque)) => {
                    let cursor_state = gbm.map(|gbm| {
                        #[cfg(feature = "renderer_pixman")]
                        let pixman_renderer = match PixmanRenderer::new() {
                            Ok(pixman_renderer) => Some(pixman_renderer),
                            Err(err) => {
                                tracing::warn!(?err, "failed to initialize pixman renderer for cursor plane");
                                None
                            }
                        };

                        let cursor_allocator =
                            GbmAllocator::new(gbm.clone(), GbmBufferFlags::CURSOR | GbmBufferFlags::WRITE);
                        let framebuffer_exporter = GbmFramebufferExporter::new(gbm.clone(), NodeFilter::None);
                        CursorState {
                            allocator: cursor_allocator,
                            framebuffer_exporter,
                            previous_output_scale: None,
                            previous_output_transform: None,
                            legacy: LegacyCursorState::new(cursor_plane_policy),
                            #[cfg(feature = "renderer_pixman")]
                            pixman_renderer,
                        }
                    });

                    let overlay_plane_element_ids = OverlayPlaneElementIds::from_planes(&planes);
                    let current_frame = FrameState::from_planes(
                        surface.plane(),
                        &planes,
                        !cursor_plane_policy.reserves_legacy_cursor(),
                    );

                    let drm_renderer = DrmCompositor {
                        primary_plane_element_id: Id::new(),
                        primary_plane_damage_bag: DamageBag::new(4),
                        primary_is_opaque: is_opaque,
                        reset_pending: true,
                        signaled_fence,
                        presentation_mode: PresentationMode::Vsync,
                        current_frame,
                        pending_frame: None,
                        queued_frame: None,
                        next_frame: None,
                        swapchain,
                        framebuffer_exporter,
                        cursor_size,
                        cursor_state,
                        cursor_plane_policy,
                        surface,
                        damage_tracker,
                        output_mode_source,
                        planes,
                        overlay_plane_element_ids,
                        element_states: IndexMap::new(),
                        previous_element_states: IndexMap::new(),
                        opaque_regions: Vec::new(),
                        element_opaque_regions_workhouse: Vec::new(),
                        framebuffer_effect_regions_workhouse: Vec::new(),
                        supports_fencing,
                        debug_flags: DebugFlags::empty(),
                        span,
                    };

                    return Ok(drm_renderer);
                }
                Err((alloc, err)) => {
                    warn!("Preferred format {} not available: {:?}", format, err);
                    allocator = alloc;
                    error = Some(err);
                }
            }
        }
        Err(error.unwrap())
    }

    /// Initialize a new [`DrmCompositor`] with a pre-selected format.
    ///
    /// The [`OutputModeSource`] can be created from an [`Output`](crate::output::Output), which will automatically track
    /// the output's mode changes. An [`OutputModeSource::Static`] variant should only be used when
    /// manually updating modes using [`DrmCompositor::set_output_mode_source`].
    ///
    /// - `output_mode_source` is used to determine the current mode, scale and transform
    /// - `surface` for the compositor to use
    /// - `planes` defines which planes the compositor is allowed to use for direct scan-out.
    ///   `None` will result in the compositor to use all planes as specified by [`DrmSurface::planes`]
    /// - `allocator` used for the primary plane swapchain
    /// - `framebuffer_exporter` is used to create drm framebuffers for the swapchain buffers (and if possible
    ///   for element buffers) for scan-out
    /// - `code` is the fixed format to initialize the framebuffer with
    /// - `modifiers` is the set of modifiers allowed, when allocating buffers with the specified color format
    /// - `cursor_size` as reported by the drm device, used for creating buffer for the cursor plane
    /// - `gbm` device used for creating buffers for the cursor plane, `None` will disable the cursor plane
    #[allow(clippy::too_many_arguments)]
    pub fn with_format(
        output_mode_source: impl Into<OutputModeSource> + Debug,
        surface: DrmSurface,
        planes: Option<Planes>,
        allocator: A,
        framebuffer_exporter: F,
        code: DrmFourcc,
        modifiers: impl IntoIterator<Item = DrmModifier>,
        cursor_size: Size<u32, BufferCoords>,
        gbm: Option<GbmDevice<G>>,
    ) -> FrameResult<Self, A, F> {
        Self::with_format_and_cursor_plane_policy(
            output_mode_source,
            surface,
            planes,
            allocator,
            framebuffer_exporter,
            code,
            modifiers,
            cursor_size,
            gbm,
            CursorPlanePolicy::Atomic,
        )
    }

    /// Initialize a compositor with a fixed format and cursor-plane policy.
    #[allow(clippy::too_many_arguments)]
    pub fn with_format_and_cursor_plane_policy(
        output_mode_source: impl Into<OutputModeSource> + Debug,
        surface: DrmSurface,
        planes: Option<Planes>,
        allocator: A,
        framebuffer_exporter: F,
        code: DrmFourcc,
        modifiers: impl IntoIterator<Item = DrmModifier>,
        cursor_size: Size<u32, BufferCoords>,
        gbm: Option<GbmDevice<G>>,
        cursor_plane_policy: CursorPlanePolicy,
    ) -> FrameResult<Self, A, F> {
        let signaled_fence = match surface.create_syncobj(true) {
            Ok(signaled_syncobj) => match surface.syncobj_to_fd(signaled_syncobj, true) {
                Ok(signaled_fence) => {
                    let _ = surface.destroy_syncobj(signaled_syncobj);
                    Some(Arc::new(signaled_fence))
                }
                Err(err) => {
                    tracing::warn!(?err, "failed to export signaled syncobj");
                    let _ = surface.destroy_syncobj(signaled_syncobj);
                    None
                }
            },
            Err(err) => {
                tracing::warn!(?err, "failed to create signaled syncobj");
                None
            }
        };

        let span = info_span!(
            parent: None,
            "drm_compositor",
            device = ?surface.dev_path(),
            crtc = ?surface.crtc(),
        );

        let output_mode_source = output_mode_source.into();

        let surface = Arc::new(surface);
        let mut planes = match planes {
            Some(planes) => planes,
            None => surface.planes().clone(),
        };

        // We do not support direct scan-out on legacy
        if surface.is_legacy() {
            planes.cursor.clear();
            planes.overlay.clear();
        }

        // The selection algorithm expects the planes to be ordered form front to back
        planes
            .overlay
            .sort_by_key(|p| std::cmp::Reverse(p.zpos.unwrap_or_default()));

        let driver = surface.get_driver().map_err(|err| {
            FrameError::DrmError(DrmError::Access(AccessError {
                errmsg: "Failed to query drm driver",
                dev: surface.dev_path(),
                source: err,
            }))
        })?;
        // `IN_FENCE_FD` makes commit fail on Nvidia driver
        // https://github.com/NVIDIA/open-gpu-kernel-modules/issues/622
        let is_nvidia = driver.name().to_string_lossy().to_lowercase().contains("nvidia")
            || driver
                .description()
                .to_string_lossy()
                .to_lowercase()
                .contains("nvidia");

        let cursor_size = Size::from((cursor_size.w as i32, cursor_size.h as i32));
        let damage_tracker = OutputDamageTracker::from_mode_source(output_mode_source.clone());
        let supports_fencing = !surface.is_legacy()
            && surface
                .get_driver_capability(DriverCapability::SyncObj)
                .map(|val| val != 0)
                .map_err(|err| {
                    FrameError::DrmError(DrmError::Access(AccessError {
                        errmsg: "Failed to query driver capability",
                        dev: surface.dev_path(),
                        source: err,
                    }))
                })?
            && plane_has_property(&*surface, surface.plane(), "IN_FENCE_FD")?
            && !(is_nvidia && nvidia_drm_version().unwrap_or((0, 0, 0)) < (560, 35, 3));

        let (swapchain, is_opaque) = Self::test_format(
            &surface,
            supports_fencing,
            &planes,
            !cursor_plane_policy.reserves_legacy_cursor(),
            allocator,
            &framebuffer_exporter,
            code,
            modifiers,
        )
        .map_err(|(_, err)| err)?;

        let cursor_state = gbm.map(|gbm| {
            #[cfg(feature = "renderer_pixman")]
            let pixman_renderer = match PixmanRenderer::new() {
                Ok(pixman_renderer) => Some(pixman_renderer),
                Err(err) => {
                    tracing::warn!(?err, "failed to initialize pixman renderer for cursor plane");
                    None
                }
            };

            let cursor_allocator =
                GbmAllocator::new(gbm.clone(), GbmBufferFlags::CURSOR | GbmBufferFlags::WRITE);
            let framebuffer_exporter = GbmFramebufferExporter::new(gbm.clone(), NodeFilter::None);
            CursorState {
                allocator: cursor_allocator,
                framebuffer_exporter,
                previous_output_scale: None,
                previous_output_transform: None,
                legacy: LegacyCursorState::new(cursor_plane_policy),
                #[cfg(feature = "renderer_pixman")]
                pixman_renderer,
            }
        });

        let overlay_plane_element_ids = OverlayPlaneElementIds::from_planes(&planes);
        let current_frame = FrameState::from_planes(
            surface.plane(),
            &planes,
            !cursor_plane_policy.reserves_legacy_cursor(),
        );

        let drm_renderer = DrmCompositor {
            primary_plane_element_id: Id::new(),
            primary_plane_damage_bag: DamageBag::new(4),
            primary_is_opaque: is_opaque,
            reset_pending: true,
            signaled_fence,
            presentation_mode: PresentationMode::Vsync,
            current_frame,
            pending_frame: None,
            queued_frame: None,
            next_frame: None,
            swapchain,
            framebuffer_exporter,
            cursor_size,
            cursor_state,
            cursor_plane_policy,
            surface,
            damage_tracker,
            output_mode_source,
            planes,
            overlay_plane_element_ids,
            element_states: IndexMap::new(),
            previous_element_states: IndexMap::new(),
            opaque_regions: Vec::new(),
            element_opaque_regions_workhouse: Vec::new(),
            framebuffer_effect_regions_workhouse: Vec::new(),
            supports_fencing,
            debug_flags: DebugFlags::empty(),
            span,
        };

        Ok(drm_renderer)
    }

    fn test_format(
        drm: &DrmSurface,
        supports_fencing: bool,
        planes: &Planes,
        include_cursor_planes: bool,
        allocator: A,
        framebuffer_exporter: &F,
        code: DrmFourcc,
        modifiers: impl IntoIterator<Item = DrmModifier>,
    ) -> Result<(Swapchain<A>, bool), (A, FrameErrorType<A, F>)> {
        let modifiers = modifiers.into_iter().collect::<IndexSet<_>>();
        let mut plane_formats = drm.plane_info().formats.iter().copied().collect::<IndexSet<_>>();

        let opaque_code = get_opaque(code).unwrap_or(code);
        if !plane_formats
            .iter()
            .any(|fmt| fmt.code == code || fmt.code == opaque_code)
        {
            return Err((allocator, FrameError::NoSupportedPlaneFormat));
        }
        plane_formats.retain(|fmt| fmt.code == code || fmt.code == opaque_code);

        if plane_formats.is_empty() {
            return Err((allocator, FrameError::NoSupportedPlaneFormat));
        }

        let plane_modifiers = plane_formats
            .iter()
            .map(|fmt| fmt.modifier)
            .collect::<IndexSet<_>>();

        let swapchain_modifiers = plane_modifiers
            .intersection(&modifiers)
            .copied()
            .collect::<Vec<_>>();

        if swapchain_modifiers.is_empty() {
            return Err((allocator, FrameError::NoSupportedPlaneFormat));
        }

        let mode = drm.pending_mode();

        let mut swapchain: Swapchain<A> = Swapchain::new(
            allocator,
            mode.size().0 as u32,
            mode.size().1 as u32,
            code,
            swapchain_modifiers,
        );

        // Test format
        let buffer = match swapchain.acquire() {
            Ok(buffer) => buffer.unwrap(),
            Err(err) => return Err((swapchain.allocator, FrameError::Allocator(err))),
        };

        let dmabuf = match buffer.export() {
            Ok(dmabuf) => dmabuf,
            Err(err) => {
                return Err((swapchain.allocator, FrameError::AsDmabufError(err)));
            }
        };

        let use_opaque = !plane_formats.iter().any(|f| f.code == code);
        let fb_buffer = match framebuffer_exporter.add_framebuffer(
            drm.device_fd(),
            ExportBuffer::Allocator(&buffer),
            use_opaque,
        ) {
            Ok(Some(fb_buffer)) => fb_buffer,
            Ok(None) => return Err((swapchain.allocator, FrameError::NoFramebuffer)),
            Err(err) => return Err((swapchain.allocator, FrameError::FramebufferExport(err))),
        };
        buffer
            .userdata()
            .insert_if_missing_threadsafe(|| CachedDrmFramebuffer::new(DrmFramebuffer::Exporter(fb_buffer)));

        let mode = drm.pending_mode();
        let handle = buffer
            .userdata()
            .get::<CachedDrmFramebuffer<<F as ExportFramebuffer<A::Buffer>>::Framebuffer>>()
            .unwrap()
            .clone();

        let mode_size = Size::from((mode.size().0 as i32, mode.size().1 as i32));

        let mut current_frame_state = FrameState::from_planes(drm.plane(), planes, include_cursor_planes);
        let plane_claim = match drm.claim_plane(drm.plane()) {
            Some(claim) => claim,
            None => {
                warn!("failed to claim primary plane",);
                return Err((swapchain.allocator, FrameError::PrimaryPlaneClaimFailed));
            }
        };

        let plane_state = PlaneState {
            skip: false,
            needs_test: true,
            element_state: None,
            config: Some(PlaneConfig {
                properties: PlaneProperties {
                    src: Rectangle::from_size(dmabuf.size()).to_f64(),
                    dst: Rectangle::from_size(mode_size),
                    transform: Transform::Normal,
                    alpha: 1.0,
                    format: buffer.format(),
                },
                buffer: DrmScanoutBuffer {
                    buffer: ScanoutBuffer::Swapchain(Arc::new(buffer)),
                    fb: handle,
                },
                damage_clips: None,
                plane_claim,
                sync: None,
            }),
        };

        match current_frame_state.test_state(drm, supports_fencing, drm.plane(), plane_state, true) {
            Ok(_) => Ok((swapchain, use_opaque)),
            Err(err) => {
                warn!(
                    "Mode-setting failed with buffer format {:?}: {}",
                    dmabuf.format(),
                    err
                );
                Err((swapchain.allocator, err.into()))
            }
        }
    }

    fn find_supported_format(
        drm: Arc<DrmSurface>,
        supports_fencing: bool,
        planes: &Planes,
        include_cursor_planes: bool,
        allocator: A,
        framebuffer_exporter: &F,
        mut renderer_formats: Vec<DrmFormat>,
        code: DrmFourcc,
    ) -> Result<(Swapchain<A>, bool), (A, FrameErrorType<A, F>)> {
        // select a format
        let mut plane_formats = drm.plane_info().formats.iter().copied().collect::<IndexSet<_>>();

        let opaque_code = get_opaque(code).unwrap_or(code);
        if !plane_formats
            .iter()
            .any(|fmt| fmt.code == code || fmt.code == opaque_code)
        {
            return Err((allocator, FrameError::NoSupportedPlaneFormat));
        }
        plane_formats.retain(|fmt| fmt.code == code || fmt.code == opaque_code);
        renderer_formats.retain(|fmt| fmt.code == code);

        trace!("Plane formats: {:?}", plane_formats);
        trace!("Renderer formats: {:?}", renderer_formats);

        let plane_modifiers = plane_formats
            .iter()
            .map(|fmt| fmt.modifier)
            .collect::<IndexSet<_>>();
        let renderer_modifiers = renderer_formats
            .iter()
            .map(|fmt| fmt.modifier)
            .collect::<IndexSet<_>>();
        debug!(
            "Remaining intersected modifiers: {:?}",
            plane_modifiers
                .intersection(&renderer_modifiers)
                .collect::<IndexSet<_>>()
        );

        if plane_formats.is_empty() {
            return Err((allocator, FrameError::NoSupportedPlaneFormat));
        } else if renderer_formats.is_empty() {
            return Err((allocator, FrameError::NoSupportedRendererFormat));
        }

        let formats = {
            // Special case: if a format supports explicit LINEAR (but no implicit Modifiers)
            // and the other doesn't support any modifier, force Implicit.
            // This should at least result in a working pipeline possibly with a linear buffer,
            // but we cannot be sure.
            if (plane_formats.len() == 1
                && plane_formats.iter().next().unwrap().modifier == DrmModifier::Invalid
                && renderer_formats
                    .iter()
                    .all(|x| x.modifier != DrmModifier::Invalid)
                && renderer_formats.iter().any(|x| x.modifier == DrmModifier::Linear))
                || (renderer_formats.len() == 1
                    && renderer_formats.first().unwrap().modifier == DrmModifier::Invalid
                    && plane_formats.iter().all(|x| x.modifier != DrmModifier::Invalid)
                    && plane_formats.iter().any(|x| x.modifier == DrmModifier::Linear))
            {
                vec![DrmFormat {
                    code,
                    modifier: DrmModifier::Invalid,
                }]
            } else {
                plane_modifiers
                    .intersection(&renderer_modifiers)
                    .cloned()
                    .map(|modifier| DrmFormat { code, modifier })
                    .collect::<Vec<_>>()
            }
        };

        debug!("Testing Formats: {:?}", formats);

        let modifiers = formats.iter().map(|x| x.modifier).collect::<Vec<_>>();

        let (swapchain, use_opaque) = Self::test_format(
            &drm,
            supports_fencing,
            planes,
            include_cursor_planes,
            allocator,
            framebuffer_exporter,
            code,
            modifiers,
        )?;

        Ok((swapchain, use_opaque))
    }

    /// Render the next frame
    ///
    /// - `elements` for this frame in front-to-back order
    /// - `frame_flags` specifies techniques allowed to realize the frame
    #[instrument(level = "trace", parent = &self.span, skip_all)]
    #[profiling::function]
    pub fn render_frame<'a, R, E>(
        &mut self,
        renderer: &mut R,
        elements: &'a [E],
        clear_color: impl Into<Color32F>,
        frame_flags: FrameFlags,
    ) -> Result<RenderFrameResult<'a, A::Buffer, F::Framebuffer, E>, RenderFrameErrorType<A, F, R>>
    where
        E: RenderElement<R>,
        R: Renderer + Bind<Dmabuf>,
        R::TextureId: Texture + 'static,
    {
        let mut clear_color = clear_color.into();

        if !self.surface.is_active() {
            return Err(RenderFrameErrorType::<A, F, R>::PrepareFrame(
                FrameError::DrmError(DrmError::DeviceInactive),
            ));
        }

        // Just reset any next state, this will put
        // any already acquired slot back to the swapchain
        std::mem::drop(self.next_frame.take());

        // If a commit is pending we may still be able to just use a previous
        // state, but we want to queue a frame so we just fake the damage to
        // make sure queue_frame won't be skipped because of no damage
        let allow_partial_update = !self.reset_pending && !self.surface.commit_pending();

        let (current_size, output_scale, output_transform) = (&self.output_mode_source)
            .try_into()
            .map_err(OutputDamageTrackerError::OutputNoMode)?;

        // Output transform is specified in surface-rotation, so inversion gives us the
        // render transform for the output itself.
        let output_transform = output_transform.invert();

        // Geometry of the output derived from the output mode including the transform
        // This is used to calculate the intersection between elements and the output.
        // The renderer (and also the logic for direct scan-out) will take care of the
        // actual transform during rendering
        let output_geometry: Rectangle<_, Physical> =
            Rectangle::from_size(output_transform.transform_size(current_size));

        // We always acquire a buffer from the swapchain even
        // if we could end up doing direct scan-out on the primary plane.
        // The reason is that we can't know upfront and we need a framebuffer
        // on the primary plane to test overlay/cursor planes
        let primary_plane_buffer = self
            .swapchain
            .acquire()
            .map_err(FrameError::Allocator)?
            .ok_or(FrameError::NoFreeSlotsError)?;

        // It is safe to call export multiple times as the Slot will cache the dmabuf for us
        let dmabuf = primary_plane_buffer.export().map_err(FrameError::AsDmabufError)?;

        // Let's check if we already have a cached framebuffer for this Slot, if not try to export
        // it and use the Slot userdata to cache it
        let maybe_buffer = primary_plane_buffer
            .userdata()
            .get::<CachedDrmFramebuffer<<F as ExportFramebuffer<A::Buffer>>::Framebuffer>>();
        if maybe_buffer.is_none() {
            let fb_buffer = self
                .framebuffer_exporter
                .add_framebuffer(
                    self.surface.device_fd(),
                    ExportBuffer::Allocator(&primary_plane_buffer),
                    self.primary_is_opaque,
                )
                .map_err(FrameError::FramebufferExport)?
                .ok_or(FrameError::NoFramebuffer)?;
            primary_plane_buffer.userdata().insert_if_missing_threadsafe(|| {
                CachedDrmFramebuffer::new(DrmFramebuffer::Exporter(fb_buffer))
            });
        }

        // This unwrap is safe as we error out above if we were unable to export a framebuffer
        let fb = primary_plane_buffer
            .userdata()
            .get::<CachedDrmFramebuffer<<F as ExportFramebuffer<A::Buffer>>::Framebuffer>>()
            .unwrap()
            .clone();

        let mut opaque_regions: Vec<Rectangle<i32, Physical>> = std::mem::take(&mut self.opaque_regions);
        std::mem::swap(&mut self.previous_element_states, &mut self.element_states);
        let mut element_states = std::mem::take(&mut self.element_states);
        element_states.reserve(std::cmp::min(elements.len(), self.planes.overlay.len()));
        let mut render_element_states = RenderElementStates {
            states: HashMap::with_capacity(elements.len()),
        };

        // So first we want to create a clean state, for that we have to reset all overlay and cursor planes
        // to nothing. We only want to test if the primary plane alone can be used for scan-out.
        let mut next_frame_state: FrameState<
            <A as Allocator>::Buffer,
            <F as ExportFramebuffer<<A as Allocator>::Buffer>>::Framebuffer,
        > = {
            let previous_state = self
                .pending_frame
                .as_ref()
                .map(|pending| &pending.frame)
                .unwrap_or(&self.current_frame);

            // This will create an empty frame state, all planes are skipped by default
            let mut next_frame_state = FrameState::from_planes(
                self.surface.plane(),
                &self.planes,
                !self.cursor_plane_policy.reserves_legacy_cursor(),
            );

            // We want to set skip to false on all planes that previously had something assigned so that
            // they get cleared when they are not longer used
            for (handle, plane_state) in next_frame_state.planes.iter_mut() {
                let reset_state = previous_state
                    .plane_state(*handle)
                    .map(|state| state.config.is_some())
                    .unwrap_or(false);

                if reset_state {
                    plane_state.skip = false;
                }
            }

            next_frame_state
        };

        // We want to make sure we can actually scan-out the primary plane, so
        // explicitly set skip to false
        let plane_claim = self.surface.claim_plane(self.surface.plane()).ok_or_else(|| {
            error!("failed to claim primary plane");
            FrameError::PrimaryPlaneClaimFailed
        })?;
        let primary_plane_state: PlaneState<
            <A as Allocator>::Buffer,
            <F as ExportFramebuffer<<A as Allocator>::Buffer>>::Framebuffer,
        > = PlaneState {
            skip: false,
            needs_test: false,
            element_state: None,
            config: Some(PlaneConfig {
                properties: PlaneProperties {
                    src: Rectangle::from_size(dmabuf.size()).to_f64(),
                    dst: Rectangle::from_size(current_size),
                    // NOTE: We do not apply the transform to the primary plane as this is handled by the dtr/renderer
                    transform: Transform::Normal,
                    alpha: 1.0,
                    format: primary_plane_buffer.format(),
                },
                buffer: DrmScanoutBuffer {
                    buffer: ScanoutBuffer::Swapchain(Arc::new(primary_plane_buffer)),
                    fb,
                },
                damage_clips: None,
                plane_claim,
                sync: None,
            }),
        };

        // unconditionally set the primary plane state
        // if this would fail the test we are screwed anyway
        next_frame_state.set_state(self.surface.plane(), primary_plane_state.clone());

        // This holds all elements that are visible on the output
        // A element is considered visible if it intersects with the output geometry
        // AND is not completely hidden behind opaque regions
        let mut output_elements: Vec<(&'a E, Rectangle<i32, Physical>, usize, bool)> =
            Vec::with_capacity(elements.len());

        let mut element_opaque_regions_workhouse = std::mem::take(&mut self.element_opaque_regions_workhouse);
        for (index, element) in elements.iter().enumerate() {
            let element_id = element.id();
            let element_geometry = element.geometry(output_scale);
            let element_loc = element_geometry.loc;

            // First test if the element overlaps with the output
            // if not we can skip it
            let element_output_geometry = match element_geometry.intersection(output_geometry) {
                Some(geo) => geo,
                None => continue,
            };

            // Then test if the element is completely hidden behind opaque regions
            element_opaque_regions_workhouse.clear();
            element_opaque_regions_workhouse.push(element_output_geometry);
            element_opaque_regions_workhouse = Rectangle::subtract_rects_many_in_place(
                element_opaque_regions_workhouse,
                opaque_regions.iter().copied(),
            );
            let element_visible_area = element_opaque_regions_workhouse
                .iter()
                .fold(0usize, |acc, item| acc + (item.size.w * item.size.h) as usize);

            if element_visible_area == 0 {
                // No need to draw a completely hidden element
                trace!("skipping completely obscured element {:?}", element.id());

                // We allow multiple instance of a single element, so do not
                // override the state if we already have one
                if !render_element_states.states.contains_key(element_id) {
                    render_element_states
                        .states
                        .insert(element_id.clone(), RenderElementState::skipped());
                }
                continue;
            }

            let element_opaque_regions = element.opaque_regions(output_scale);
            element_opaque_regions_workhouse.clear();
            element_opaque_regions_workhouse.push(element_output_geometry);
            element_opaque_regions_workhouse = Rectangle::subtract_rects_many_in_place(
                element_opaque_regions_workhouse,
                element_opaque_regions.iter().copied(),
            );
            let element_is_opaque = element_opaque_regions_workhouse.is_empty();

            opaque_regions.extend(
                element_opaque_regions
                    .into_iter()
                    .map(|mut region| {
                        region.loc += element_loc;
                        region
                    })
                    .filter_map(|geo| geo.intersection(output_geometry)),
            );

            // If the element is completely opaque and spans the whole output nothing below
            // will be visible. In this case we can short-cut the whole loop and just mark all
            // remaining elements as skipped.
            //
            // We also use this to special case for single pixel buffers that span the whole
            // output. If the last visible element is a solid color we can override the clear
            // color and remove the element completely. This will make the element directly above
            // this element the last element, enabling direct scan-out on the primary plane for it.
            if element_is_opaque && element_output_geometry.contains_rect(output_geometry) {
                let element_color = element.underlying_storage(renderer).and_then(|storage| {
                    if let UnderlyingStorage::Wayland(buffer) = storage {
                        single_pixel_buffer::get_single_pixel_buffer(buffer)
                            .ok()
                            .map(|spb| Color32F::from(spb.rgba32f()))
                    } else {
                        None
                    }
                });

                if let Some(color) = element_color {
                    clear_color = color;

                    render_element_states
                        .states
                        .entry(element_id.clone())
                        .and_modify(|state| {
                            if matches!(state.presentation_state, RenderElementPresentationState::Skipped) {
                                *state = RenderElementState::rendered(element_visible_area);
                            } else {
                                state.visible_area += element_visible_area;
                            }
                        })
                        .or_insert_with(|| RenderElementState::rendered(element_visible_area));
                } else {
                    output_elements.push((
                        element,
                        element_geometry,
                        element_visible_area,
                        element_is_opaque,
                    ));
                }

                for element in elements.iter().skip(index + 1) {
                    let element_id = element.id();
                    // We allow multiple instance of a single element, so do not
                    // override the state if we already have one
                    if !render_element_states.states.contains_key(element_id) {
                        render_element_states
                            .states
                            .insert(element_id.clone(), RenderElementState::skipped());
                    }
                }
                break;
            }

            output_elements.push((element, element_geometry, element_visible_area, element_is_opaque));
        }
        self.element_opaque_regions_workhouse = element_opaque_regions_workhouse;

        // This will hold the element that has been selected for direct scan-out on
        // the primary plane if any
        let mut primary_plane_scanout_element: Option<&'a E> = None;
        // This will hold all elements that have been assigned to the primary plane
        // for rendering
        let mut primary_plane_elements: Vec<&'a E> = Vec::with_capacity(elements.len());
        // This will hold the element per plane that has been assigned to a overlay/underlay
        // plane for direct scan-out
        let mut overlay_plane_elements: IndexMap<plane::Handle, &'a E> =
            IndexMap::with_capacity(self.planes.overlay.len());
        // This will hold the element assigned on the cursor plane if any
        let mut cursor_plane_element: Option<&'a E> = None;
        let mut legacy_cursor: Option<LegacyCursorPresentation> = None;

        // DRIFT-1427: the framebuffer effects of this frame and the region each one
        // samples. `output_elements` is front-to-back, so everything after an entry
        // here is below that effect. Computed BEFORE the assignment loop opens, not
        // accumulated inside it: there is then no ordering between "observe the
        // effect" and "decide about this element" for a later edit to get wrong, and
        // no state that could outlive the frame.
        // Taken and put back like `element_opaque_regions_workhouse` above, so the scan
        // reuses one buffer across frames rather than allocating a fresh one per frame.
        let mut effect_regions = std::mem::take(&mut self.framebuffer_effect_regions_workhouse);
        collect_framebuffer_effect_regions(
            &mut effect_regions,
            output_elements
                .iter()
                .map(|(element, geometry, ..)| (element.is_framebuffer_effect(), *geometry)),
        );

        let legacy_cursor_candidates = output_elements
            .iter()
            .enumerate()
            .filter_map(|(index, (element, ..))| (element.kind() == Kind::Cursor).then_some(index))
            .collect::<SmallVec<[_; 2]>>();
        let legacy_cursor_index = if frame_flags.contains(FrameFlags::ALLOW_LEGACY_CURSOR)
            && self.cursor_plane_policy.reserves_legacy_cursor()
            && legacy_cursor_candidates.len() == 1
        {
            Some(legacy_cursor_candidates[0])
        } else {
            let retained = self.disable_legacy_cursor();
            if let Some(presentation) = retained {
                legacy_cursor = Some(presentation.clone());
                output_elements
                    .iter()
                    .position(|(element, ..)| element.id() == &presentation.element_id)
            } else {
                None
            }
        };

        let output_elements_len = output_elements.len();
        for (index, (element, element_geometry, element_visible_area, element_is_opaque)) in
            output_elements.iter().enumerate()
        {
            let element_id = element.id();
            let element_geometry = *element_geometry;
            let remaining_elements = output_elements_len - index;
            let element_is_opaque = *element_is_opaque;

            if legacy_cursor_index == Some(index) {
                let assignment = if let Some(presentation) = legacy_cursor.as_ref() {
                    LegacyCursorAssignment::Presented(presentation.clone())
                } else {
                    self.try_assign_legacy_cursor(
                        renderer,
                        *element,
                        element_geometry,
                        output_scale,
                        output_transform,
                        output_geometry,
                    )
                };
                match assignment {
                    LegacyCursorAssignment::Presented(presentation) => {
                        render_element_states
                            .states
                            .entry(element_id.clone())
                            .and_modify(|state| {
                                state.presentation_state = RenderElementPresentationState::ZeroCopy;
                                state.visible_area += element_visible_area;
                            })
                            .or_insert_with(|| RenderElementState::zero_copy(*element_visible_area));
                        legacy_cursor = Some(presentation);
                        continue;
                    }
                    LegacyCursorAssignment::Atomic => {
                        next_frame_state.add_cursor_planes(&self.planes);
                    }
                    LegacyCursorAssignment::Software => {}
                }
            }

            // Check if we found our last item, we can try to do
            // direct scan-out on the primary plane
            // If we already assigned an element to
            // an underlay plane we will have a hole punch element
            // on the primary plane, this will disable direct scan-out
            // on the primary plane.
            let try_assign_primary_plane = if remaining_elements == 1 && primary_plane_elements.is_empty() {
                let crtc_background_matches_clear_color =
                    (clear_color.r() == 0f32 && clear_color.g() == 0f32 && clear_color.b() == 0f32)
                        || clear_color.a() == 0f32;
                let element_spans_complete_output = element_geometry.contains_rect(output_geometry);
                let overlaps_with_underlay = self
                    .planes
                    .overlay
                    .iter()
                    .filter(|p| {
                        p.zpos.unwrap_or_default() < self.surface.plane_info().zpos.unwrap_or_default()
                    })
                    .any(|p| next_frame_state.overlaps(p.handle, element_geometry));
                !overlaps_with_underlay
                    && (crtc_background_matches_clear_color
                        || (element_spans_complete_output && element_is_opaque))
            } else {
                false
            };

            match self.try_assign_element(
                renderer,
                *element,
                index,
                element_geometry,
                element_is_opaque,
                &mut element_states,
                &primary_plane_elements,
                &effect_regions,
                output_scale,
                &mut next_frame_state,
                output_transform,
                output_geometry,
                try_assign_primary_plane,
                frame_flags,
            ) {
                Ok(direct_scan_out_plane) => {
                    match direct_scan_out_plane.type_ {
                        drm::control::PlaneType::Overlay => {
                            overlay_plane_elements.insert(direct_scan_out_plane.handle, element);
                        }
                        drm::control::PlaneType::Primary => primary_plane_scanout_element = Some(element),
                        drm::control::PlaneType::Cursor => cursor_plane_element = Some(element),
                    }

                    if let Some(state) = render_element_states.states.get_mut(element_id) {
                        state.presentation_state = RenderElementPresentationState::ZeroCopy;
                        state.visible_area += element_visible_area;
                    } else {
                        render_element_states.states.insert(
                            element_id.clone(),
                            RenderElementState::zero_copy(*element_visible_area),
                        );
                    }
                }
                Err(reason) => {
                    if let Some(reason) = reason {
                        if !render_element_states.states.contains_key(element_id) {
                            render_element_states.states.insert(
                                element_id.clone(),
                                RenderElementState::rendering_with_reason(reason),
                            );
                        }
                    }

                    primary_plane_elements.push(element);
                }
            }
        }
        // Put the scan buffer back so the next frame reuses its capacity.
        self.framebuffer_effect_regions_workhouse = effect_regions;

        // Cleanup old state (e.g. old dmabuffers)
        for element_state in element_states.values_mut() {
            element_state.fb_cache.cleanup();
        }
        self.element_states = element_states;
        self.previous_element_states.clear();
        opaque_regions.clear();
        self.opaque_regions = opaque_regions;

        let previous_state = self
            .pending_frame
            .as_ref()
            .map(|pending| &pending.frame)
            .unwrap_or(&self.current_frame);

        // Check if the next frame state is fully compatible with the previous frame state.
        // If not do a single atomic commit test and when that fails render everything that failed
        // the test on the primary plane. This will also automatically correct any mistake we made
        // during plane assignment and start the full test cycle on the next frame.
        if next_frame_state
            .test_state_complete(
                previous_state,
                &self.surface,
                self.supports_fencing,
                false,
                allow_partial_update,
            )
            .is_err()
        {
            trace!("atomic test failed for frame, resetting frame");

            let mut removed_overlay_elements: Vec<(usize, &E)> = Vec::with_capacity(
                next_frame_state
                    .planes
                    .iter()
                    .filter(|(_, state)| state.needs_test)
                    .count(),
            );
            for (plane, state) in next_frame_state.planes.iter_mut() {
                // We can skip everything that is known to work already
                if !state.needs_test {
                    continue;
                }

                // Check if the element we are potentially going to remove is
                // on the primary plane, cursor plane or an overlay plane
                let element = if *plane == self.surface.plane() {
                    primary_plane_scanout_element.take()
                } else if self.planes.cursor.iter().any(|p| *plane == p.handle) {
                    cursor_plane_element.take()
                } else {
                    overlay_plane_elements.shift_remove(plane)
                };

                // If we have no element on this plane skip the rest
                let Some(element) = element else {
                    continue;
                };

                // Reset the plane config and state
                state.config = None;
                state.skip = false;
                state.needs_test = false;
                let element_z_index = state.element_state.take().map(|s| s.z_index).unwrap_or_default();
                removed_overlay_elements.push((element_z_index, element));
                // Note: This might not be completely correct if the same element is present
                // multiple times and only gets removed once. But this is pretty unlikely to
                // happen and will only result in reporting wrong visible area size and scan-out state
                // for a single frame.
                render_element_states.states.remove(element.id());
            }

            // If we removed any element from some plane we have
            // to make sure we actually have a slot on the primary
            // plane we can render into
            if !removed_overlay_elements.is_empty() {
                next_frame_state.set_state(self.surface.plane(), primary_plane_state);
            }

            removed_overlay_elements.sort_by_key(|(z_index, _)| *z_index);
            primary_plane_elements = removed_overlay_elements
                .into_iter()
                .map(|(_, element)| element)
                .chain(primary_plane_elements.into_iter())
                .collect();
        }

        // If a plane has been moved or no longer has a buffer we need to report that as damage
        for (handle, previous_plane_state) in previous_state.planes.iter() {
            // plane has been removed, so remove the plane from the plane id cache
            if previous_plane_state.config.is_some()
                && next_frame_state
                    .plane_state(*handle)
                    .as_ref()
                    .and_then(|state| state.config.as_ref())
                    .is_none()
            {
                self.overlay_plane_element_ids.remove_plane(handle);
            }
        }

        let render = next_frame_state
            .plane_buffer(self.surface.plane())
            .map(|config| matches!(config.buffer, ScanoutBuffer::Swapchain(_)))
            .unwrap_or(false);

        if render {
            trace!(
                "rendering {} elements on the primary {:?}",
                primary_plane_elements.len(),
                self.surface.plane(),
            );
            let (mut dmabuf, age) = {
                let primary_plane_state = next_frame_state.plane_state(self.surface.plane()).unwrap();
                let config = primary_plane_state.config.as_ref().unwrap();
                let slot = match &config.buffer.buffer {
                    ScanoutBuffer::Swapchain(slot) => slot,
                    _ => unreachable!(),
                };

                // It is safe to call export multiple times as the Slot will cache the dmabuf for us
                let dmabuf = slot.export().map_err(FrameError::AsDmabufError)?;
                let age = slot.age().into();
                (dmabuf, age)
            };

            // store the current renderer debug flags and replace them
            // with our own
            let renderer_debug_flags = renderer.debug_flags();
            renderer.set_debug_flags(self.debug_flags);

            // First we collect all our fake elements for overlay and underlays
            // This is used to transport the opaque regions for elements that
            // have been assigned to planes and to realize hole punching for
            // underlays. We use an Id per plane/element combination to not
            // interfere with the element damage state in the output damage tracker.
            // Using the original element id would store the commit in the
            // OutputDamageTracker without actual rendering anything -> bad
            // Using a id per plane could result in an issue when a different
            // element with the same geometry gets assigned and has the same
            // commit -> unlikely but possible
            // So we use an Id per plane for as long as we have the same element
            // on that plane.
            let overlay_plane_elements = overlay_plane_elements.iter().filter_map(|(p, element)| {
                let id = self
                    .overlay_plane_element_ids
                    .plane_id_for_element_id(p, element.id());

                let plane_z_pos = self
                    .planes
                    .overlay
                    .iter()
                    .find_map(|info| {
                        if info.handle == *p {
                            Some(info.zpos.unwrap_or_default())
                        } else {
                            None
                        }
                    })
                    .unwrap_or_default();
                let is_underlay = plane_z_pos < self.surface.plane_info().zpos.unwrap_or_default();
                if is_underlay {
                    Some(HolepunchRenderElement::from_render_element(id, element, output_scale).into())
                } else {
                    OverlayPlaneElement::from_render_element(id, *element, output_scale)
                        .map(DrmRenderElements::from)
                }
            });
            // Then render all remaining elements assigned to the primary plane
            let elements = overlay_plane_elements
                .chain(
                    primary_plane_elements
                        .into_iter()
                        .map(|e| DrmRenderElements::Other(e)),
                )
                .collect::<Vec<_>>();

            let mut framebuffer = renderer
                .bind(&mut dmabuf)
                .map_err(|err| RenderFrameError::RenderFrame(OutputDamageTrackerError::Rendering(err)))?;
            let render_res =
                self.damage_tracker
                    .render_output(renderer, &mut framebuffer, age, &elements, clear_color);

            // restore the renderer debug flags
            renderer.set_debug_flags(renderer_debug_flags);

            match render_res {
                Ok(render_output_result) => {
                    if render_output_result.damage.is_none() {
                        // if we receive no damage we can assume no rendering took place
                        // and we should trigger a cleanup of the renderer texture cache
                        // to prevent holding textures longer then necessary
                        let _ = renderer.cleanup_texture_cache();
                    }

                    for (id, state) in render_output_result.states.states.into_iter() {
                        // Skip the state for our fake elements
                        if self.overlay_plane_element_ids.contains_plane_id(&id) {
                            continue;
                        }

                        if let Some(existing_state) = render_element_states.states.get_mut(&id) {
                            if matches!(
                                existing_state.presentation_state,
                                RenderElementPresentationState::Skipped
                            ) {
                                *existing_state = state;
                            } else {
                                existing_state.visible_area += state.visible_area;
                            }
                        } else {
                            render_element_states.states.insert(id.clone(), state);
                        }
                    }

                    // Fixup damage on plane, if we used the plane for direct scan-out before
                    // but now use it for rendering we do not replace the damage which is
                    // the whole plane initially.
                    let had_direct_scan_out = previous_state
                        .plane_state(self.surface.plane())
                        .map(|state| state.element_state.is_some())
                        .unwrap_or(true);

                    let primary_plane_state = next_frame_state.plane_state_mut(self.surface.plane()).unwrap();
                    let config = primary_plane_state.config.as_mut().unwrap();

                    if !had_direct_scan_out {
                        if let Some(render_damage) = render_output_result.damage {
                            trace!("rendering damage: {:?}", render_damage);

                            self.primary_plane_damage_bag.add(render_damage.iter().map(|d| {
                                d.to_logical(1).to_buffer(
                                    1,
                                    Transform::Normal,
                                    &output_geometry.size.to_logical(1),
                                )
                            }));

                            // Here we need to apply the output_transform to the damage since we
                            // haven't rotated our framebuffer. dst is already in the same
                            // coordinate space src is, so no transform needed.
                            config.damage_clips = PlaneDamageClips::from_damage(
                                self.surface.device_fd(),
                                config.properties.src,
                                config.properties.dst,
                                Transform::Normal,
                                output_transform,
                                render_damage.iter().copied(),
                            )
                            .ok()
                            .flatten();
                            config.sync = Some((render_output_result.sync.clone(), None));
                        } else {
                            trace!("skipping primary plane, no damage");

                            primary_plane_state.skip = true;
                            *config = previous_state
                                .plane_state(self.surface.plane())
                                .and_then(|state| state.config.as_ref().cloned())
                                .unwrap_or_else(|| config.clone());
                        }
                    } else {
                        trace!(
                            "clearing previous direct scan-out on primary plane, damaging complete output"
                        );
                        self.primary_plane_damage_bag
                            .add([output_geometry.to_logical(1).to_buffer(
                                1,
                                Transform::Normal,
                                &output_geometry.size.to_logical(1),
                            )]);

                        config.sync = Some((render_output_result.sync.clone(), None));
                    }
                }
                Err(err) => {
                    // Rendering failed at some point, reset the buffers
                    // as we probably now have some half drawn buffer
                    self.swapchain.reset_buffers();
                    return Err(RenderFrameError::from(err));
                }
            }
        } else {
            // if we are constantly doing direct scan-out on the primary plane
            // we have to cleanup the renderer texture cache as this would
            // only happen implicit during rendering otherwise
            let _ = renderer.cleanup_texture_cache();
        }

        let primary_plane_element = if render {
            let (slot, sync) = {
                let primary_plane_state = next_frame_state.plane_state(self.surface.plane()).unwrap();
                let config = primary_plane_state.config.as_ref().unwrap();
                (
                    config.buffer.clone(),
                    config
                        .sync
                        .as_ref()
                        .map(|(sync, _)| sync.clone())
                        .unwrap_or_default(),
                )
            };

            PrimaryPlaneElement::Swapchain(PrimarySwapchainElement {
                slot,
                transform: output_transform,
                damage: self.primary_plane_damage_bag.snapshot(),
                sync,
            })
        } else {
            PrimaryPlaneElement::Element(primary_plane_scanout_element.unwrap())
        };

        // if the update only contains a cursor position update, skip it for vrr
        if frame_flags.contains(FrameFlags::SKIP_CURSOR_ONLY_UPDATES)
            && allow_partial_update
            && next_frame_state.planes.iter().all(|(plane, state)| {
                state.skip
                    || (self.planes.cursor.iter().any(|p| *plane == p.handle)
                        && state.buffer().map(|b| &b.fb)
                            == previous_state.plane_buffer(*plane).map(|b| &b.fb))
            })
        {
            for plane in self.planes.cursor.iter() {
                let Some(state) = next_frame_state.plane_state_mut(plane.handle) else {
                    continue;
                };
                state.skip = true;
            }
        }

        let next_frame = PreparedFrame {
            kind: if allow_partial_update {
                PreparedFrameKind::Partial
            } else {
                PreparedFrameKind::Full
            },
            frame: next_frame_state,
        };
        let frame_reference: RenderFrameResult<'a, A::Buffer, F::Framebuffer, E> = RenderFrameResult {
            is_empty: next_frame.is_empty(),
            primary_element: primary_plane_element,
            overlay_elements: overlay_plane_elements.into_values().collect(),
            cursor_element: cursor_plane_element,
            legacy_cursor,
            states: render_element_states,
            primary_plane_element_id: self.primary_plane_element_id.clone(),
            supports_fencing: self.supports_fencing,
        };

        // We only store the next frame if it actually contains any changes or if a commit is pending
        // Storing the (empty) frame could keep a reference to wayland buffers which
        // could otherwise be potentially released on `frame_submitted`
        if !next_frame.is_empty() {
            self.next_frame = Some(next_frame);
        }

        Ok(frame_reference)
    }

    /// Queues the current frame for scan-out.
    ///
    /// If `render_frame` has not been called prior to this function or returned no damage
    /// this function will return [`FrameError::EmptyFrame`]. Instead of calling `queue_frame` it
    /// is the callers responsibility to re-schedule the frame. A simple strategy for frame
    /// re-scheduling is to queue a one-shot timer that will trigger after approximately one
    /// retrace duration.
    ///
    /// *Note*: It is your responsibility to synchronize rendering if the [`RenderFrameResult`]
    /// returned by the previous [`render_frame`](DrmCompositor::render_frame) call returns `true` on [`RenderFrameResult::needs_sync`].
    ///
    /// *Note*: This function needs to be followed up with [`DrmCompositor::frame_submitted`]
    /// when a vblank event is received, that denotes successful scan-out of the frame.
    /// Otherwise the underlying swapchain will eventually run out of buffers.
    ///
    /// `user_data` can be used to attach some data to a specific buffer and later retrieved with [`DrmCompositor::frame_submitted`]
    #[profiling::function]
    pub fn queue_frame(&mut self, user_data: U) -> FrameResult<(), A, F> {
        if !self.surface.is_active() {
            return Err(FrameErrorType::<A, F>::DrmError(DrmError::DeviceInactive));
        }

        let prepared_frame = self.next_frame.take().ok_or(FrameErrorType::<A, F>::EmptyFrame)?;
        if prepared_frame.is_empty() {
            return Err(FrameErrorType::<A, F>::EmptyFrame);
        }

        if let Some(plane_state) = prepared_frame.frame.plane_state(self.surface.plane()) {
            if !plane_state.skip {
                let slot = plane_state.buffer().and_then(|config| match &config.buffer {
                    ScanoutBuffer::Swapchain(slot) => Some(slot),
                    _ => None,
                });

                if let Some(slot) = slot {
                    self.swapchain.submitted(slot);
                }
            }
        }

        self.queued_frame = Some(QueuedFrame {
            prepared_frame,
            user_data,
        });
        if self.pending_frame.is_none() {
            self.submit()?;
        }
        Ok(())
    }

    /// Commits the current frame for scan-out.
    ///
    /// If `render_frame` has not been called prior to this function or returned no damage
    /// this function will return [`FrameError::EmptyFrame`]. Instead of calling `commit_frame` it
    /// is the callers responsibility to re-schedule the frame. A simple strategy for frame
    /// re-scheduling is to queue a one-shot timer that will trigger after approximately one
    /// retrace duration.
    ///
    /// *Note*: It is your responsibility to synchronize rendering if the [`RenderFrameResult`]
    /// returned by the previous [`render_frame`](DrmCompositor::render_frame) call returns `true` on [`RenderFrameResult::needs_sync`].
    ///
    /// *Note*: This function should not be followed up with [`DrmCompositor::frame_submitted`]
    /// and will not generate a vblank event on the underlying device.
    pub fn commit_frame(&mut self) -> FrameResult<(), A, F> {
        if !self.surface.is_active() {
            return Err(FrameErrorType::<A, F>::DrmError(DrmError::DeviceInactive));
        }

        let mut prepared_frame = self.next_frame.take().ok_or(FrameErrorType::<A, F>::EmptyFrame)?;
        if prepared_frame.is_empty() {
            return Err(FrameErrorType::<A, F>::EmptyFrame);
        }

        if let Some(plane_state) = prepared_frame.frame.plane_state(self.surface.plane()) {
            if !plane_state.skip {
                let slot = plane_state.buffer().and_then(|config| match &config.buffer {
                    ScanoutBuffer::Swapchain(slot) => Some(slot),
                    _ => None,
                });

                if let Some(slot) = slot {
                    self.swapchain.submitted(slot);
                }
            }
        }

        let flip = prepared_frame
            .frame
            .commit(&self.surface, self.supports_fencing, false, false);

        let res = self.handle_flip(&prepared_frame, flip);

        if res.is_ok() {
            self.queued_frame = None;
            self.pending_frame = None;
            self.current_frame = prepared_frame.frame;
        }

        res
    }

    /// Re-evaluates the current state of the crtc and forces calls to [`render_frame`](DrmCompositor::render_frame)
    /// to return `false` for [`RenderFrameResult::is_empty`] until a frame is queued with [`queue_frame`](DrmCompositor::queue_frame).
    ///
    /// It is recommended to call this function after this used [`Session`](crate::backend::session::Session)
    /// gets re-activated / VT switched to.
    ///
    /// Usually you do not need to call this in other circumstances, but if
    /// the state of the crtc is modified elsewhere, you may call this function
    /// to reset it's internal state.
    pub fn reset_state(&mut self) -> Result<(), DrmError> {
        self.surface.reset_state()?;
        self.reset_pending = true;
        Ok(())
    }

    #[profiling::function]
    fn submit(&mut self) -> FrameResult<(), A, F> {
        let QueuedFrame {
            mut prepared_frame,
            user_data,
        } = self.queued_frame.take().unwrap();

        let allow_partial_update = prepared_frame.kind == PreparedFrameKind::Partial;
        // Only a plain page flip can tear; a modeset commit is always vsync (DRIFT-984). The async
        // request is the ordinary request plus PAGE_FLIP_ASYNC: IN_FENCE_FD and FB_DAMAGE_CLIPS are
        // kept, since recent kernels exempt both from the async-flip property check, and keeping the
        // fence means the kernel still waits for the buffer to be ready (no scanout of an unfinished
        // buffer, and the needs_sync() contract stays consistent). A kernel that refuses the async
        // commit falls back to a synchronous flip of the same request inside page_flip.
        let want_async = self.presentation_mode == PresentationMode::Async;

        let (flip, presentation_mode) = if self.surface.commit_pending() {
            (
                prepared_frame
                    .frame
                    .commit(&self.surface, self.supports_fencing, allow_partial_update, true),
                PresentationMode::Vsync,
            )
        } else {
            match prepared_frame.frame.page_flip(
                &self.surface,
                self.supports_fencing,
                allow_partial_update,
                true,
                want_async,
            ) {
                Ok(mode) => (Ok(()), mode),
                Err(err) => (Err(err), PresentationMode::Vsync),
            }
        };

        let res = self.handle_flip(&prepared_frame, flip);

        if res.is_ok() {
            self.pending_frame = Some(PendingFrame {
                frame: prepared_frame.frame,
                user_data,
                presentation_mode,
            });
        }

        res
    }

    fn handle_flip(
        &mut self,
        prepared_frame: &PreparedFrame<A, F>,
        flip: Result<(), crate::backend::drm::error::Error>,
    ) -> FrameResult<(), A, F> {
        match flip {
            Ok(_) => {
                if prepared_frame.kind == PreparedFrameKind::Full {
                    self.reset_pending = false;
                }
            }
            Err(crate::backend::drm::error::Error::Access(ref access))
                if access.source.kind() == ErrorKind::InvalidInput =>
            {
                // In case the commit/flip failed while we tried to directly scan-out
                // something on the primary plane we can try to mark this as failed for
                // the next call to render_frame
                let primary_plane_element_state = prepared_frame
                    .frame
                    .plane_state(self.surface.plane())
                    .and_then(|plane_state| {
                        plane_state
                            .element_state
                            .as_ref()
                            .map(|element_state| &element_state.id)
                    })
                    .and_then(|primary_plane_element_id| {
                        self.element_states.get_mut(primary_plane_element_id)
                    });

                if let Some(primary_plane_element_state) = primary_plane_element_state {
                    for instance in primary_plane_element_state.instances.iter_mut() {
                        instance.failed_planes.primary = true;
                    }
                }
            }
            Err(_) => {}
        };

        flip.map_err(FrameError::DrmError)
    }

    /// Marks the current frame as submitted.
    ///
    /// *Note*: Needs to be called, after the vblank event of the matching [`DrmDevice`](super::DrmDevice)
    /// was received after calling [`DrmCompositor::queue_frame`] on this surface.
    /// Otherwise the underlying swapchain will run out of buffers eventually.
    #[profiling::function]
    pub fn frame_submitted(&mut self) -> FrameResult<Option<U>, A, F> {
        if let Some(PendingFrame {
            mut frame,
            user_data,
            presentation_mode: _,
        }) = self.pending_frame.take()
        {
            std::mem::swap(&mut frame, &mut self.current_frame);
            if self.queued_frame.is_some() {
                self.submit()?;
            }
            Ok(Some(user_data))
        } else {
            Ok(None)
        }
    }

    /// Reset the underlying buffers
    pub fn reset_buffers(&mut self) {
        self.disable_legacy_cursor();
        self.invalidate_legacy_cursor_lifecycle();
        self.swapchain.reset_buffers();
    }

    /// Reset the age for all buffers.
    ///
    /// This can be used to efficiently clear the damage history without having to
    /// modify the damage for each surface.
    pub fn reset_buffer_ages(&mut self) {
        self.swapchain.reset_buffer_ages();
    }

    /// Returns the underlying [`crtc`] of this surface
    pub fn crtc(&self) -> crtc::Handle {
        self.surface.crtc()
    }

    /// Returns the underlying [`plane`] of this surface
    pub fn plane(&self) -> plane::Handle {
        self.surface.plane()
    }

    /// Currently used [`connector`]s of this `Surface`
    pub fn current_connectors(&self) -> impl IntoIterator<Item = connector::Handle> {
        self.surface.current_connectors()
    }

    /// Returns the pending [`connector`]s
    /// used for the next frame queued via [`queue_frame`](DrmCompositor::queue_frame).
    pub fn pending_connectors(&self) -> impl IntoIterator<Item = connector::Handle> {
        self.surface.pending_connectors()
    }

    /// Tries to add a new [`connector`]
    /// to be used after the next commit.
    ///
    /// **Warning**: You need to make sure, that the connector is not used with another surface
    /// or was properly removed via `remove_connector` + `commit` before adding it to another surface.
    /// Behavior if failing to do so is undefined, but might result in rendering errors or the connector
    /// getting removed from the other surface without updating it's internal state.
    ///
    /// Fails if the `connector` is not compatible with the underlying [`crtc`]
    /// (e.g. no suitable [`encoder`](drm::control::encoder) may be found)
    /// or is not compatible with the currently pending
    /// [`Mode`].
    pub fn add_connector(&self, connector: connector::Handle) -> FrameResult<(), A, F> {
        self.surface
            .add_connector(connector)
            .map_err(FrameError::DrmError)
    }

    /// Tries to mark a [`connector`]
    /// for removal on the next commit.
    pub fn remove_connector(&self, connector: connector::Handle) -> FrameResult<(), A, F> {
        self.surface
            .remove_connector(connector)
            .map_err(FrameError::DrmError)
    }

    /// Tries to replace the current connector set with the newly provided one on the next commit.
    ///
    /// Fails if one new `connector` is not compatible with the underlying [`crtc`]
    /// (e.g. no suitable [`encoder`](drm::control::encoder) may be found)
    /// or is not compatible with the currently pending
    /// [`Mode`].
    pub fn set_connectors(&self, connectors: &[connector::Handle]) -> FrameResult<(), A, F> {
        self.surface
            .set_connectors(connectors)
            .map_err(FrameError::DrmError)
    }

    /// Returns the currently active [`Mode`]
    /// of the underlying [`crtc`]
    pub fn current_mode(&self) -> Mode {
        self.surface.current_mode()
    }

    /// Returns the currently pending [`Mode`]
    /// to be used after the next commit.
    pub fn pending_mode(&self) -> Mode {
        self.surface.pending_mode()
    }

    /// Tries to set a new [`Mode`]
    /// to be used after the next commit.
    ///
    /// Fails if the mode is not compatible with the underlying
    /// [`crtc`] or any of the
    /// pending [`connector`]s.
    pub fn use_mode(&mut self, mode: Mode) -> FrameResult<(), A, F> {
        self.disable_legacy_cursor();
        self.invalidate_legacy_cursor_lifecycle();
        self.surface.use_mode(mode).map_err(FrameError::DrmError)?;
        let (w, h) = mode.size();
        self.swapchain.resize(w as _, h as _);
        Ok(())
    }

    /// Returns if Variable Refresh Rate is advertised as supported by the given connector.
    ///
    /// See [`DrmSurface::vrr_supported`] for more details.
    pub fn vrr_supported(&self, conn: connector::Handle) -> FrameResult<VrrSupport, A, F> {
        self.surface.vrr_supported(conn).map_err(FrameError::DrmError)
    }

    /// Returns if Variable Refresh Rate is currently enabled for frames composed by this [`DrmCompositor`].
    pub fn vrr_enabled(&self) -> bool {
        self.surface.vrr_enabled()
    }

    /// Tries to set variable refresh rate (VRR) for the next frame.
    ///
    /// Doing so might cause the next frame to trigger a modeset.
    /// Check [`DrmCompositor::vrr_supported`], which indicates if VRR can be
    /// used without a modeset on the attached connectors.
    pub fn use_vrr(&mut self, vrr: bool) -> FrameResult<(), A, F> {
        self.surface.use_vrr(vrr).map_err(FrameError::DrmError)
    }

    /// Set the [`DebugFlags`] to use
    ///
    /// Note: This will reset the primary plane swapchain if
    /// the flags differ from the current flags
    pub fn set_debug_flags(&mut self, flags: DebugFlags) {
        if self.debug_flags != flags {
            self.debug_flags = flags;
            self.swapchain.reset_buffers();
        }
    }

    /// Returns the current enabled [`DebugFlags`]
    pub fn debug_flags(&self) -> DebugFlags {
        self.debug_flags
    }

    /// Returns a reference to the underlying drm surface
    pub fn surface(&self) -> &DrmSurface {
        &self.surface
    }

    /// Get the format of the underlying swapchain
    pub fn format(&self) -> DrmFourcc {
        self.swapchain.format()
    }

    /// Get the allowed modifiers of the underlying swapchain
    pub fn modifiers(&self) -> &[DrmModifier] {
        self.swapchain.modifiers()
    }

    /// Reset the underlying swapchain and assign a new color format.
    pub fn set_format(
        &mut self,
        allocator: A,
        code: DrmFourcc,
        modifiers: impl IntoIterator<Item = DrmModifier>,
    ) -> Result<(), FrameErrorType<A, F>> {
        let (swapchain, is_oapque) = Self::test_format(
            &self.surface,
            self.supports_fencing,
            &self.planes,
            !self.cursor_plane_policy.reserves_legacy_cursor(),
            allocator,
            &self.framebuffer_exporter,
            code,
            modifiers,
        )
        .map_err(|(_, err)| err)?;

        self.swapchain = swapchain;
        self.primary_is_opaque = is_oapque;

        Ok(())
    }

    /// Change the output mode source.
    pub fn set_output_mode_source(&mut self, output_mode_source: OutputModeSource) {
        // Avoid clearing damage if mode source did not change.
        if output_mode_source == self.output_mode_source {
            return;
        }

        self.disable_legacy_cursor();
        self.damage_tracker = OutputDamageTracker::from_mode_source(output_mode_source.clone());
        self.output_mode_source = output_mode_source;
        self.invalidate_legacy_cursor_lifecycle();
    }

    fn enable_atomic_cursor_planes(&mut self) {
        self.cursor_plane_policy = CursorPlanePolicy::Atomic;
        self.current_frame.add_cursor_planes(&self.planes);
        if let Some(frame) = self.pending_frame.as_mut() {
            frame.frame.add_cursor_planes(&self.planes);
        }
        if let Some(frame) = self.queued_frame.as_mut() {
            frame.prepared_frame.frame.add_cursor_planes(&self.planes);
        }
        if let Some(frame) = self.next_frame.as_mut() {
            frame.frame.add_cursor_planes(&self.planes);
        }
    }

    #[allow(deprecated)]
    fn disable_legacy_cursor(&mut self) -> Option<LegacyCursorPresentation> {
        let active = match self.cursor_state.as_ref().map(|state| &state.legacy.ownership) {
            Some(LegacyCursorOwnership::Active(active)) => active.presentation.clone(),
            _ => return None,
        };
        match self
            .surface
            .device_fd()
            .set_cursor2::<GbmBuffer>(self.surface.crtc(), None, (0, 0))
        {
            Ok(()) => {
                self.cursor_state.as_mut().unwrap().legacy.ownership = LegacyCursorOwnership::Disabled;
                None
            }
            Err(error) => {
                debug!(?error, "failed to disable legacy cursor");
                Some(active)
            }
        }
    }

    #[allow(deprecated)]
    fn try_assign_legacy_cursor<R, E>(
        &mut self,
        renderer: &mut R,
        element: &E,
        element_geometry: Rectangle<i32, Physical>,
        output_scale: Scale<f64>,
        output_transform: Transform,
        output_geometry: Rectangle<i32, Physical>,
    ) -> LegacyCursorAssignment
    where
        R: Renderer,
        E: RenderElement<R>,
    {
        let element_size = output_transform.transform_size(element_geometry.size);
        if element_size.w > self.cursor_size.w || element_size.h > self.cursor_size.h {
            return LegacyCursorAssignment::Software;
        }
        let physical_origin = cursor_plane_location(
            element.location(output_scale),
            self.cursor_size,
            output_geometry,
            output_transform,
        );

        let Some(cursor_state) = self.cursor_state.as_ref() else {
            return LegacyCursorAssignment::Software;
        };
        if matches!(cursor_state.legacy.ownership, LegacyCursorOwnership::Atomic) {
            return LegacyCursorAssignment::Atomic;
        }

        let unchanged = match &cursor_state.legacy.ownership {
            LegacyCursorOwnership::Active(active) => {
                active.presentation.element_id == *element.id()
                    && active.presentation.commit == element.current_commit()
                    && active.element_size == element_size
                    && active.output_scale == output_scale
                    && active.output_transform == output_transform
            }
            _ => false,
        };
        if unchanged {
            let active = match &self.cursor_state.as_ref().unwrap().legacy.ownership {
                LegacyCursorOwnership::Active(active) => active,
                _ => unreachable!(),
            };
            if active.presentation.physical_origin == physical_origin {
                return LegacyCursorAssignment::Presented(active.presentation.clone());
            }
            let token = active.presentation.token;
            if let LegacyCursorMoveResult::Moved(presentation) =
                self.move_legacy_cursor(token, physical_origin)
            {
                return LegacyCursorAssignment::Presented(presentation);
            }
        }

        let replacing_active = matches!(
            self.cursor_state.as_ref().unwrap().legacy.ownership,
            LegacyCursorOwnership::Active(_)
        );
        let cursor_buffer = {
            let cursor_state = self.cursor_state.as_mut().unwrap();
            render_legacy_cursor_buffer(
                cursor_state,
                renderer,
                element,
                element_geometry,
                self.cursor_size,
                output_transform,
            )
        };
        let Some(cursor_buffer) = cursor_buffer else {
            return LegacyCursorAssignment::Software;
        };

        let legacy_succeeded = self.cursor_state.as_ref().unwrap().legacy.legacy_succeeded;
        match install_legacy_cursor(
            &SurfaceLegacyCursorIo(&self.surface),
            &cursor_buffer,
            physical_origin,
            replacing_active,
            legacy_succeeded,
        ) {
            LegacyCursorInstallResult::Installed => {}
            LegacyCursorInstallResult::PreserveActive => {
                let active = match &self.cursor_state.as_ref().unwrap().legacy.ownership {
                    LegacyCursorOwnership::Active(active) => active,
                    _ => unreachable!(),
                };
                return LegacyCursorAssignment::Presented(active.presentation.clone());
            }
            LegacyCursorInstallResult::DisabledSoftware => {
                let legacy = &mut self.cursor_state.as_mut().unwrap().legacy;
                legacy.legacy_succeeded = true;
                legacy.ownership = LegacyCursorOwnership::Disabled;
                return LegacyCursorAssignment::Software;
            }
            LegacyCursorInstallResult::CandidateSoftware => {
                self.cursor_state.as_mut().unwrap().legacy.ownership = LegacyCursorOwnership::Candidate;
                return LegacyCursorAssignment::Software;
            }
            LegacyCursorInstallResult::Atomic => {
                self.cursor_state.as_mut().unwrap().legacy.ownership = LegacyCursorOwnership::Atomic;
                self.enable_atomic_cursor_planes();
                return LegacyCursorAssignment::Atomic;
            }
        }

        let cursor_state = self.cursor_state.as_mut().unwrap();
        cursor_state.legacy.legacy_succeeded = true;
        let presentation = LegacyCursorPresentation {
            token: cursor_state.legacy.token(),
            physical_origin,
            element_id: element.id().clone(),
            commit: element.current_commit(),
            plane_size: self.cursor_size,
        };
        cursor_state.legacy.ownership = LegacyCursorOwnership::Active(LegacyCursorActive {
            _buffer: Arc::new(cursor_buffer),
            presentation: presentation.clone(),
            element_size,
            output_scale,
            output_transform,
        });
        LegacyCursorAssignment::Presented(presentation)
    }

    #[allow(clippy::too_many_arguments)]
    #[instrument(level = "trace", skip_all)]
    #[profiling::function]
    fn try_assign_element<'a, R, E>(
        &mut self,
        renderer: &mut R,
        element: &'a E,
        element_zindex: usize,
        element_geometry: Rectangle<i32, Physical>,
        element_is_opaque: bool,
        element_states: &mut IndexMap<Id, ElementState<<F as ExportFramebuffer<A::Buffer>>::Framebuffer>>,
        primary_plane_elements: &[&'a E],
        effect_regions: &[(usize, Rectangle<i32, Physical>)],
        scale: Scale<f64>,
        frame_state: &mut CompositorFrameState<A, F>,
        output_transform: Transform,
        output_geometry: Rectangle<i32, Physical>,
        try_assign_primary_plane: bool,
        frame_flags: FrameFlags,
    ) -> Result<PlaneAssignment, Option<RenderingReason>>
    where
        R: Renderer + Bind<Dmabuf>,
        E: RenderElement<R>,
    {
        // Check if we have a free plane, otherwise we can exit early
        if !frame_flags.intersects(FrameFlags::ALLOW_SCANOUT) {
            trace!(
                "skipping direct scan-out for element {:?}, no free planes",
                element.id()
            );
            return Err(None);
        };
        if element.is_framebuffer_effect() {
            return Err(None);
        }
        // DRIFT-1427: and neither may anything below one it overlaps, or the effect
        // samples a hole punch instead of the content. Same rule, other half.
        if below_framebuffer_effect(element_zindex, element_geometry, effect_regions) {
            trace!(
                "skipping direct scan-out for element {:?}, it is below a framebuffer effect it overlaps",
                element.id()
            );
            return Err(None);
        }

        let mut rendering_reason: Option<RenderingReason> = None;

        if try_assign_primary_plane {
            match self.try_assign_primary_plane(
                renderer,
                element,
                element_zindex,
                element_geometry,
                element_states,
                scale,
                frame_state,
                output_transform,
                output_geometry,
                frame_flags,
            ) {
                Ok(plane) => {
                    trace!(
                        "assigned element {:?} to primary {:?}",
                        element.id(),
                        self.surface.plane()
                    );
                    return Ok(plane);
                }
                Err(err) => rendering_reason = rendering_reason.or(err),
            };
        }

        if let Some(plane) = self.try_assign_cursor_plane(
            renderer,
            element,
            element_zindex,
            element_geometry,
            scale,
            frame_state,
            output_transform,
            output_geometry,
            frame_flags,
        ) {
            trace!("assigned element {:?} to cursor {:?}", element.id(), plane.handle);
            return Ok(plane);
        }

        match self.try_assign_overlay_plane(
            renderer,
            element,
            element_zindex,
            element_geometry,
            element_is_opaque,
            element_states,
            primary_plane_elements,
            scale,
            frame_state,
            output_transform,
            output_geometry,
            frame_flags,
        ) {
            Ok(plane) => {
                trace!(
                    "assigned element {:?} to overlay plane {:?}",
                    element.id(),
                    plane.handle
                );
                return Ok(plane);
            }
            Err(err) => rendering_reason = rendering_reason.or(err),
        }

        Err(rendering_reason)
    }

    #[allow(clippy::too_many_arguments)]
    #[instrument(level = "trace", skip_all)]
    #[profiling::function]
    fn try_assign_primary_plane<'a, R, E>(
        &mut self,
        renderer: &mut R,
        element: &'a E,
        element_zindex: usize,
        element_geometry: Rectangle<i32, Physical>,
        element_states: &mut IndexMap<Id, ElementState<<F as ExportFramebuffer<A::Buffer>>::Framebuffer>>,
        scale: Scale<f64>,
        frame_state: &mut CompositorFrameState<A, F>,
        output_transform: Transform,
        output_geometry: Rectangle<i32, Physical>,
        frame_flags: FrameFlags,
    ) -> Result<PlaneAssignment, Option<RenderingReason>>
    where
        R: Renderer,
        E: RenderElement<R>,
    {
        if !frame_flags
            .intersects(FrameFlags::ALLOW_PRIMARY_PLANE_SCANOUT | FrameFlags::ALLOW_PRIMARY_PLANE_SCANOUT_ANY)
        {
            return Err(None);
        }

        if frame_state
            .plane_state(self.surface.plane())
            .map(|state| state.element_state.is_some())
            .unwrap_or(true)
        {
            return Err(None);
        }

        let element_config = self.element_config(
            renderer,
            element,
            element_zindex,
            element_geometry,
            element_states,
            frame_state,
            output_transform,
            output_geometry,
            true,
        )?;

        if let ScanoutBuffer::Swapchain(slot) = &frame_state
            .plane_buffer(self.surface.plane())
            .expect("We have a buffer for the primary plane")
            .buffer
        {
            if !frame_flags.contains(FrameFlags::ALLOW_PRIMARY_PLANE_SCANOUT_ANY)
                && slot.format() != element_config.properties.format
            {
                trace!(
                    "failed to assign element {:?} to primary {:?}, format doesn't match",
                    element.id(),
                    self.surface.plane()
                );
                return Err(None);
            }
        }

        let has_underlay = self
            .planes
            .overlay
            .iter()
            .filter(|plane| {
                self.surface.plane_info().zpos.unwrap_or_default() > plane.zpos.unwrap_or_default()
            })
            .any(|plane| frame_state.is_assigned(plane.handle));

        if has_underlay {
            trace!(
                "failed to assign element {:?} to primary {:?}, already has underlay",
                element.id(),
                self.surface.plane()
            );
            return Err(None);
        }

        if element_config.failed_planes.primary {
            return Err(Some(RenderingReason::ScanoutFailed));
        }

        let res = self.try_assign_plane(
            element,
            &element_config,
            self.surface.plane_info(),
            scale,
            frame_state,
        );

        if let Err(Some(RenderingReason::ScanoutFailed)) = res {
            element_config.failed_planes.primary = true;
        }

        res
    }

    #[allow(clippy::too_many_arguments)]
    #[instrument(level = "trace", skip_all)]
    #[profiling::function]
    fn try_assign_cursor_plane<R, E>(
        &mut self,
        renderer: &mut R,
        element: &E,
        element_zindex: usize,
        element_geometry: Rectangle<i32, Physical>,
        scale: Scale<f64>,
        frame_state: &mut CompositorFrameState<A, F>,
        output_transform: Transform,
        output_geometry: Rectangle<i32, Physical>,
        frame_flags: FrameFlags,
    ) -> Option<PlaneAssignment>
    where
        R: Renderer,
        E: RenderElement<R>,
    {
        if self.cursor_plane_policy.reserves_legacy_cursor() {
            return None;
        }
        if !frame_flags.contains(FrameFlags::ALLOW_CURSOR_PLANE_SCANOUT) {
            return None;
        }

        let Some(cursor_state) = self.cursor_state.as_mut() else {
            trace!("no cursor state, skipping cursor rendering");
            return None;
        };

        // only try to assign elements on a cursor plane that indicate so
        if element.kind() != Kind::Cursor {
            trace!(
                "skipping element {:?} on cursor plane(s), element kind not cursor",
                element.id(),
            );
            return None;
        }

        let element_size = output_transform.transform_size(element_geometry.size);

        // if the element is greater than the cursor size we can not
        // use the cursor plane to scan out the element
        if element_size.w > self.cursor_size.w || element_size.h > self.cursor_size.h {
            trace!("element {:?} too big for cursor plane(s), skipping", element.id(),);
            return None;
        }

        // For now we only support a single cursor plane, so first test if we already
        // assigned something to any cursor plane
        if let Some(plane_info) = self
            .planes
            .cursor
            .iter()
            .find(|plane_info| frame_state.is_assigned(plane_info.handle))
        {
            trace!(
                "skipping element {:?} on cursor {:?}, plane already has element assigned",
                element.id(),
                plane_info.handle
            );
            return None;
        }

        let previous_state = self
            .pending_frame
            .as_ref()
            .map(|pending| &pending.frame)
            .unwrap_or(&self.current_frame);

        // In case we have multiple cursor planes we will try to keep using
        // the same cursor plane for as long as possible.
        // So first we test the previous state for an assigned cursor plane and
        // if that fails we will try to pick the first unclaimed cursor plane.
        let Some((plane_info, plane_claim)) = self
            .planes
            .cursor
            .iter()
            .find_map(|plane_info: &PlaneInfo| {
                if previous_state.is_assigned(plane_info.handle) {
                    self.surface
                        .claim_plane(plane_info.handle)
                        .map(|claim| (plane_info, claim))
                } else {
                    None
                }
            })
            .or_else(|| {
                self.planes.cursor.iter().find_map(|plane_info| {
                    self.surface
                        .claim_plane(plane_info.handle)
                        .map(|claim| (plane_info, claim))
                })
            })
        else {
            trace!(
                "skipping element {:?} on cursor plane(s), no free plane found",
                element.id(),
            );
            return None;
        };

        let cursor_plane_size = if let Some(size_hints) = plane_info.size_hints.as_deref() {
            // size hints are in order of preference, so we can choose the first one
            // that can hold the whole element
            //
            // Note: we use the legacy cursor size as a pre-check and expect it to hold the
            // biggest possible size
            size_hints
                .iter()
                .find(|hint| hint.w as i32 >= element_size.w && hint.h as i32 >= element_size.h)
                .map(|hint| Size::<i32, Physical>::from((hint.w as i32, hint.h as i32)))
                .unwrap_or(self.cursor_size)
        } else {
            self.cursor_size
        };

        // this calculates the location of the cursor plane taking the simulated transform
        // into consideration
        let cursor_plane_location = cursor_plane_location(
            element.location(scale),
            cursor_plane_size,
            output_geometry,
            output_transform,
        );

        let previous_state = self
            .pending_frame
            .as_ref()
            .map(|pending| &pending.frame)
            .unwrap_or(&self.current_frame);

        let previous_element_state = previous_state
            .plane_state(plane_info.handle)
            .and_then(|state| state.element_state.as_ref());

        // if the output transform or scale change we have to (re-)render the cursor plane,
        // also if the element changed or reports damage we have to render it
        let render = cursor_state
            .previous_output_transform
            .map(|t| t != output_transform)
            .unwrap_or(true)
            || cursor_state
                .previous_output_scale
                .map(|s| s != scale)
                .unwrap_or(true)
            || previous_element_state
                .map(|element_state| {
                    element_state.id != *element.id()
                        || element.current_commit() != element_state.commit
                        || element_state.cursor_size != Some(element_size)
                })
                .unwrap_or(true)
            || previous_state
                .plane_state(plane_info.handle)
                .and_then(|state| {
                    state
                        .config
                        .as_ref()
                        .map(|config| config.properties.dst.size != cursor_plane_size)
                })
                .unwrap_or(true);

        // check if the cursor plane location changed
        let reposition = previous_state
            .plane_state(plane_info.handle)
            .and_then(|state| {
                state
                    .config
                    .as_ref()
                    .map(|config| config.properties.dst.loc != cursor_plane_location)
            })
            .unwrap_or(true);

        // ok, nothing changed, try to keep the previous state
        if !render && !reposition {
            let mut plane_state = previous_state.plane_state(plane_info.handle).unwrap().clone();
            plane_state.skip = true;
            // Note: we know that we had a cursor plane in the
            // previous frame and that nothing changed. In this
            // case skip the whole testing
            plane_state.needs_test = false;
            frame_state.set_state(plane_info.handle, plane_state);
            return Some(plane_info.into());
        }

        // we no not have to re-render but update the planes location
        if !render && reposition {
            trace!("repositioning cursor plane");
            let mut plane_state = previous_state.plane_state(plane_info.handle).unwrap().clone();
            plane_state.skip = false;
            // Note: we know that we had a cursor plane in the
            // previous frame, so we assume a simple location change
            // does not not to be tested
            plane_state.needs_test = false;
            let config = plane_state.config.as_mut().unwrap();
            config.properties.dst.loc = cursor_plane_location;
            frame_state.set_state(plane_info.handle, plane_state);
            return Some(plane_info.into());
        }

        trace!(
            "trying to render element {:?} on cursor {:?}",
            element.id(),
            plane_info.handle
        );

        // if we fail to create a buffer we can just return false and
        // force the cursor to be rendered on the primary plane
        let mut cursor_buffer = match cursor_state.allocator.create_buffer(
            cursor_plane_size.w as u32,
            cursor_plane_size.h as u32,
            DrmFourcc::Argb8888,
            &[DrmModifier::Linear],
        ) {
            Ok(buffer) => buffer,
            Err(err) => {
                debug!("failed to create cursor buffer: {}", err);
                return None;
            }
        };

        // if we fail to export a framebuffer for our buffer we can skip the rest
        let framebuffer = match cursor_state.framebuffer_exporter.add_framebuffer(
            self.surface.device_fd(),
            ExportBuffer::Allocator(&cursor_buffer),
            false,
        ) {
            Ok(Some(fb)) => fb,
            Ok(None) => {
                debug!(
                    "failed to export framebuffer for cursor {:?}: no framebuffer available",
                    plane_info.handle
                );
                return None;
            }
            Err(err) => {
                debug!(
                    "failed to export framebuffer for cursor {:?}: {}",
                    plane_info.handle, err
                );
                return None;
            }
        };

        let cursor_buffer_size = cursor_plane_size.to_logical(1).to_buffer(1, Transform::Normal);

        #[cfg(not(feature = "renderer_pixman"))]
        if !copy_element_to_cursor_bo(
            renderer,
            element,
            element_size,
            cursor_plane_size,
            output_transform,
            &mut cursor_buffer,
        ) {
            tracing::trace!("failed to copy element to cursor bo, skipping element on cursor plane");
            return None;
        }

        #[cfg(feature = "renderer_pixman")]
        if !copy_element_to_cursor_bo(
            renderer,
            element,
            element_size,
            cursor_plane_size,
            output_transform,
            &mut cursor_buffer,
        ) {
            profiling::scope!("render cursor plane");
            tracing::trace!("cursor fast-path copy failed, falling back to rendering using offscreen buffer");

            let Some(storage) = element.underlying_storage(renderer) else {
                trace!("Can't obtain cursor's underlying storage");
                return None;
            };

            let pixman_renderer = cursor_state.pixman_renderer.as_mut()?;

            // Create a pixman image from the source cursor data. This will either be set by the
            // client, or the compositor's choice.
            let cursor_texture = match storage {
                UnderlyingStorage::Wayland(buffer) => pixman_renderer
                    .import_buffer(buffer, None, &[element.src().to_i32_up()])
                    .transpose()
                    .ok()
                    .flatten(),
                UnderlyingStorage::Memory(memory) => {
                    let format = memory.format();
                    let size = memory.size();
                    let Ok(pixman_format) = pixman::FormatCode::try_from(format) else {
                        debug!("No pixman format for {format}");
                        return None;
                    };
                    unsafe {
                        match pixman::Image::from_raw_mut(
                            pixman_format,
                            size.w as usize,
                            size.h as usize,
                            memory.as_ptr() as *mut u32,
                            memory.stride() as usize,
                            false,
                        ) {
                            Ok(image) => Some(PixmanTexture::from(image)),
                            Err(e) => {
                                debug!("pixman cursor: {e}");
                                None
                            }
                        }
                    }
                }
            }?;

            let ret = cursor_buffer
                .map_mut::<_, Result<_, PixmanError>>(
                    0,
                    0,
                    cursor_buffer_size.w as u32,
                    cursor_buffer_size.h as u32,
                    |mbo| {
                        let plane_pixman_format = pixman::FormatCode::try_from(DrmFourcc::Argb8888).unwrap();
                        let mut cursor_dst = unsafe {
                            pixman::Image::from_raw_mut(
                                plane_pixman_format,
                                mbo.width() as usize,
                                mbo.height() as usize,
                                mbo.buffer_mut().as_mut_ptr() as *mut u32,
                                mbo.stride() as usize,
                                false,
                            )
                        }
                        .map_err(|_| PixmanError::ImportFailed)?;
                        let mut framebuffer = pixman_renderer.bind(&mut cursor_dst)?;
                        let mut frame =
                            pixman_renderer.render(&mut framebuffer, cursor_plane_size, output_transform)?;
                        frame.clear(Color32F::TRANSPARENT, &[Rectangle::from_size(cursor_plane_size)])?;
                        let src = element.src();
                        let dst = Rectangle::from_size(element_geometry.size);
                        frame.render_texture_from_to(
                            &cursor_texture,
                            src,
                            dst,
                            &[dst],
                            &[],
                            element.transform(),
                            element.alpha(),
                        )?;
                        let _ = frame.finish()?.wait(); // what can we do?
                        Ok(())
                    },
                )
                .expect("Lost track of cursor device");

            if let Err(err) = ret {
                debug!("{err}");
                return None;
            }
        };

        let src = Rectangle::from_size(cursor_buffer_size).to_f64();
        let dst = Rectangle::new(cursor_plane_location, cursor_plane_size);

        let config = PlaneConfig {
            properties: PlaneProperties {
                src,
                dst,
                alpha: 1.0,
                transform: Transform::Normal,
                format: framebuffer.format(),
            },
            buffer: DrmScanoutBuffer {
                buffer: ScanoutBuffer::Cursor(Arc::new(cursor_buffer)),
                fb: CachedDrmFramebuffer::new(DrmFramebuffer::Gbm(framebuffer)),
            },
            damage_clips: None,
            plane_claim,
            sync: None,
        };
        let is_compatible = previous_state
            .plane_state(plane_info.handle)
            .map(|state| {
                state
                    .config
                    .as_ref()
                    .map(|other| {
                        // Note: We do not use the plane config `is_compatible` test
                        // here as we exclude the destination location from the test
                        other.properties.src == config.properties.src
                            && other.properties.dst.size == config.properties.dst.size
                            && other.properties.alpha == config.properties.alpha
                            && other.properties.transform == config.properties.transform
                            && other.properties.format == config.properties.format
                    })
                    .unwrap_or(false)
            })
            .unwrap_or(false);

        let plane_state = PlaneState {
            skip: false,
            // Note: we assume we only have to test if the plane is
            // not compatible. This should only happen if we either
            // had no cursor plane before or we did direct scan-out
            // on it. A simple re-position without re-render is
            // already handled earlier.
            needs_test: !is_compatible,
            element_state: Some(PlaneElementState {
                id: element.id().clone(),
                commit: element.current_commit(),
                z_index: element_zindex,
                cursor_size: Some(element_size),
            }),
            config: Some(config),
        };

        let res = if is_compatible {
            frame_state.set_state(plane_info.handle, plane_state);
            true
        } else {
            frame_state
                .test_state(
                    &self.surface,
                    self.supports_fencing,
                    plane_info.handle,
                    plane_state,
                    false,
                )
                .is_ok()
        };

        if res {
            cursor_state.previous_output_scale = Some(scale);
            cursor_state.previous_output_transform = Some(output_transform);
            Some(plane_info.into())
        } else {
            info!("failed to test cursor {:?} state", plane_info.handle);
            None
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[instrument(level = "trace", skip_all)]
    #[profiling::function]
    fn element_config<'a, R, E>(
        &mut self,
        renderer: &mut R,
        element: &E,
        element_zindex: usize,
        element_geometry: Rectangle<i32, Physical>,
        element_states: &'a mut IndexMap<Id, ElementState<<F as ExportFramebuffer<A::Buffer>>::Framebuffer>>,
        frame_state: &mut CompositorFrameState<A, F>,
        output_transform: Transform,
        output_geometry: Rectangle<i32, Physical>,
        allow_opaque_fallback: bool,
    ) -> Result<
        ElementPlaneConfig<
            'a,
            <A as Allocator>::Buffer,
            <F as ExportFramebuffer<<A as Allocator>::Buffer>>::Framebuffer,
        >,
        ExportBufferError,
    >
    where
        R: Renderer,
        E: RenderElement<R>,
    {
        let element_id = element.id();

        // We can only try to do direct scan-out for element that provide a underlying storage
        let underlying_storage = element
            .underlying_storage(renderer)
            .ok_or(ExportBufferError::NoUnderlyingStorage)?;

        let export_buffer = ExportBuffer::from_underlying_storage(&underlying_storage)
            .ok_or(ExportBufferError::Unsupported)?;

        if !self.framebuffer_exporter.can_add_framebuffer(&export_buffer) {
            return Err(ExportBufferError::Unsupported);
        }

        // First we try to find a state in our new states, this is important if
        // we got the same id multiple times. If we can't find it we use the previous
        // state if available
        if !element_states.contains_key(element_id) {
            let previous_fb_cache = self
                .previous_element_states
                .get_mut(element_id)
                // Note: We can mem::take the old fb_cache here here as we guarantee that
                // the element state will always overwrite the current state at the end of render_frame
                .map(|state| std::mem::take(&mut state.fb_cache))
                .unwrap_or_default();
            element_states.insert(
                element_id.clone(),
                ElementState {
                    instances: SmallVec::new(),
                    fb_cache: previous_fb_cache,
                },
            );
        }
        let element_fb_cache: &mut ElementFramebufferCache<
            <F as ExportFramebuffer<<A as Allocator>::Buffer>>::Framebuffer,
        > = element_states
            .get_mut(element_id)
            .map(|state| &mut state.fb_cache)
            .unwrap();

        let element_cache_key =
            ElementFramebufferCacheKey::from_underlying_storage(&underlying_storage, allow_opaque_fallback)
                .ok_or(ExportBufferError::Unsupported)?;
        let cached_fb = element_fb_cache.get(&element_cache_key);

        if cached_fb.is_none() {
            trace!(
                "no cached fb, exporting new fb for element {:?} underlying storage {:?}",
                element_id, &underlying_storage
            );

            let fb = self
                .framebuffer_exporter
                .add_framebuffer(self.surface.device_fd(), export_buffer, allow_opaque_fallback)
                .map_err(|err| {
                    trace!("failed to add framebuffer: {:?}", err);
                    ExportBufferError::ExportFailed
                })
                .and_then(|fb| {
                    fb.map(|fb| CachedDrmFramebuffer::new(DrmFramebuffer::Exporter(fb)))
                        .ok_or(ExportBufferError::Unsupported)
                });

            if fb.is_err() {
                trace!(
                    "could not import framebuffer for element {:?} underlying storage {:?}",
                    element_id, &underlying_storage
                );
            }

            element_fb_cache.insert(element_cache_key.clone(), fb);
        } else {
            trace!(
                "using cached fb for element {:?} underlying storage {:?}",
                element_id, &underlying_storage
            );
        }

        let fb: &CachedDrmFramebuffer<<F as ExportFramebuffer<<A as Allocator>::Buffer>>::Framebuffer> =
            element_fb_cache.get(&element_cache_key).unwrap()?;

        let src = element.src();
        let dst = output_transform.transform_rect_in(element_geometry, &output_geometry.size);
        // the output transform we are passed is already inverted to represent CW rotation (this is done to match what the
        // renderer is doing), but drm and the elements actually use/expect CCW rotation. to solve this we just invert
        // the transform again here.
        let transform = apply_output_transform(
            apply_underlying_storage_transform(element.transform(), &underlying_storage),
            output_transform.invert(),
        );
        let alpha = element.alpha();
        let properties = PlaneProperties {
            src,
            dst,
            alpha,
            transform,
            format: fb.format(),
        };
        let buffer: DrmScanoutBuffer<
            <A as Allocator>::Buffer,
            <F as ExportFramebuffer<<A as Allocator>::Buffer>>::Framebuffer,
        > = ScanoutBuffer::from_underlying_storage(underlying_storage)
            .map(|buffer| DrmScanoutBuffer {
                fb: fb.clone(),
                buffer,
            })
            .ok_or(ExportBufferError::Unsupported)?;

        if !element_states
            .get(element_id)
            .unwrap()
            .instances
            .iter()
            .any(|i| i.properties == properties)
        {
            let overlay_bitmask =
                self.planes
                    .overlay
                    .iter()
                    .enumerate()
                    .fold(0u32, |mut acc, (index, plane)| {
                        if frame_state.is_assigned(plane.handle) {
                            acc |= 1 << index;
                        }
                        acc
                    });
            let cursor_bitmask =
                self.planes
                    .cursor
                    .iter()
                    .enumerate()
                    .fold(0u32, |mut acc, (index, plane)| {
                        if frame_state.is_assigned(plane.handle) {
                            acc |= 1 << index;
                        }
                        acc
                    });
            let current_plane_snapshot = PlanesSnapshot {
                primary: frame_state.is_assigned(self.surface.plane()),
                cursor_bitmask,
                overlay_bitmask,
            };

            let element_state = element_states.get_mut(element_id).unwrap();
            element_state.instances.push(ElementInstanceState {
                properties,
                active_planes: current_plane_snapshot,
                failed_planes: Default::default(),
            });

            if let Some(previous_state) = self.previous_element_states.get(element_id) {
                // lets look if we find a previous instance with exactly the same properties.
                // if we find one we can test if nothing changed and re-use the failed tests
                let matching_instance = previous_state
                    .instances
                    .iter()
                    .find(|i| i.properties == properties);

                if let Some(matching_instance) = matching_instance {
                    if current_plane_snapshot == matching_instance.active_planes {
                        let previous_frame_state = self
                            .pending_frame
                            .as_ref()
                            .map(|pending| &pending.frame)
                            .unwrap_or(&self.current_frame);

                        // Note: we ignore the cursor plane here as this would result
                        // in constant re-tests of cursor moves and we do not expect
                        // that to influence the test state of our elements.
                        // Adding or removing cursor can influence the other planes, but
                        // is already covered in the active planes check.
                        let primary_plane_changed = if current_plane_snapshot.primary {
                            frame_state.plane_properties(self.surface.plane())
                                != previous_frame_state.plane_properties(self.surface.plane())
                        } else {
                            false
                        };

                        let overlay_plane_changed =
                            self.planes.overlay.iter().enumerate().any(|(index, plane)| {
                                // we only want to test planes that are currently in use
                                if current_plane_snapshot.overlay_bitmask & (1 << index) == 0 {
                                    return false;
                                }

                                frame_state.plane_properties(plane.handle)
                                    != previous_frame_state.plane_properties(plane.handle)
                            });

                        if !(primary_plane_changed || overlay_plane_changed) {
                            // we now know that nothing changed and we can assume any previously failed
                            // test will again fail
                            let instance_state = element_state
                                .instances
                                .iter_mut()
                                .find(|i| i.properties == properties)
                                .unwrap();
                            instance_state.failed_planes = matching_instance.failed_planes;
                        }
                    }
                }
            }
        }

        let failed_planes = element_states
            .get_mut(element_id)
            .unwrap()
            .instances
            .iter_mut()
            .find_map(|i| {
                if i.properties == properties {
                    Some(&mut i.failed_planes)
                } else {
                    None
                }
            })
            .unwrap();

        Ok(ElementPlaneConfig {
            properties,
            z_index: element_zindex,
            geometry: element_geometry,
            buffer,
            failed_planes,
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[instrument(level = "trace", skip_all)]
    #[profiling::function]
    fn try_assign_overlay_plane<'a, R, E>(
        &mut self,
        renderer: &mut R,
        element: &'a E,
        element_zindex: usize,
        element_geometry: Rectangle<i32, Physical>,
        element_is_opaque: bool,
        element_states: &mut IndexMap<Id, ElementState<<F as ExportFramebuffer<A::Buffer>>::Framebuffer>>,
        primary_plane_elements: &[&'a E],
        scale: Scale<f64>,
        frame_state: &mut CompositorFrameState<A, F>,
        output_transform: Transform,
        output_geometry: Rectangle<i32, Physical>,
        frame_flags: FrameFlags,
    ) -> Result<PlaneAssignment, Option<RenderingReason>>
    where
        R: Renderer,
        E: RenderElement<R>,
    {
        if !frame_flags.contains(FrameFlags::ALLOW_OVERLAY_PLANE_SCANOUT) {
            return Err(None);
        }

        // only try to assign elements on an overlay plane that indicate so
        if element.kind() != Kind::ScanoutCandidate && element.kind() != Kind::Cursor {
            trace!(
                "skipping element {:?} on overlay plane(s), element kind not scanout-candidate/cursor",
                element.id(),
            );
            return Err(None);
        }

        let element_id = element.id();

        // Check if we have a free plane, otherwise we can exit early
        if self
            .planes
            .overlay
            .iter()
            .all(|plane| frame_state.is_assigned(plane.handle))
        {
            trace!(
                "skipping overlay planes for element {:?}, no free planes",
                element_id
            );
            return Err(None);
        }

        let element_config = self.element_config(
            renderer,
            element,
            element_zindex,
            element_geometry,
            element_states,
            frame_state,
            output_transform,
            output_geometry,
            false,
        )?;

        let overlaps_with_primary_plane_element = primary_plane_elements.iter().any(|e| {
            let other_geometry = e.geometry(scale);
            other_geometry.overlaps(element_config.geometry)
        });

        let primary_plane_has_alpha = frame_state
            .plane_buffer(self.surface.plane())
            .map(|state| has_alpha(state.format().code))
            .unwrap_or(false);

        let previous_frame_state = self
            .pending_frame
            .as_ref()
            .map(|pending| &pending.frame)
            .unwrap_or(&self.current_frame);

        // We consider a plane compatible if the z-index of the previous assigned
        // element is or equal to our z-index and the properties (src/dst/format/...)
        // are equal. The reason for the z-index limitation is that we do not want
        // to assign ourself to the same plane if our z-index changed. That could
        // result in assigning the element on a lower plane as necessary and then
        // blocking direct scan-out for some other element
        let is_plane_compatible = |plane: &&PlaneInfo| {
            previous_frame_state
                .plane_state(plane.handle)
                .map(|state| {
                    state
                        .element_state
                        .as_ref()
                        .map(|state| state.z_index <= element_config.z_index)
                        .unwrap_or(false)
                        && state
                            .config
                            .as_ref()
                            .map(|config| config.properties.is_compatible(&element_config.properties))
                            .unwrap_or(false)
                })
                .unwrap_or(false)
        };

        let mut test_overlay_plane = |plane: &PlaneInfo,
                                      element_config: &ElementPlaneConfig<
            '_,
            <A as Allocator>::Buffer,
            <F as ExportFramebuffer<<A as Allocator>::Buffer>>::Framebuffer,
        >| {
            // something is already assigned to our overlay plane
            if frame_state.is_assigned(plane.handle) {
                trace!(
                    "skipping {:?} with zpos {:?} for element {:?}, already has element assigned, skipping",
                    plane.handle, plane.zpos, element_id,
                );
                return Err(None);
            }

            // test if the plane represents an underlay
            let is_underlay =
                self.surface.plane_info().zpos.unwrap_or_default() > plane.zpos.unwrap_or_default();

            if is_underlay && !(element_is_opaque && primary_plane_has_alpha) {
                trace!(
                    "skipping direct scan-out on underlay {:?} with zpos {:?}, element {:?} is not opaque or primary plane has no alpha channel",
                    plane.handle, plane.zpos, element_id
                );
                return Err(None);
            }

            // if the element overlaps with an element on
            // the primary plane and is not an underlay
            // we can not assign it to any overlay plane
            if overlaps_with_primary_plane_element && !is_underlay {
                trace!(
                    "skipping direct scan-out on {:?} with zpos {:?}, element {:?} overlaps with element on primary plane",
                    plane.handle, plane.zpos, element_id,
                );
                return Err(None);
            }

            let overlaps_with_plane_underneath = self
                .planes
                .overlay
                .iter()
                .filter(|info| {
                    info.handle != plane.handle
                        && info.zpos.unwrap_or_default() <= plane.zpos.unwrap_or_default()
                })
                .any(|overlapping_plane| {
                    frame_state.overlaps(overlapping_plane.handle, element_config.geometry)
                });

            // if we overlap we a plane below which already
            // has an element assigned we can not use the
            // plane for direct scan-out
            if overlaps_with_plane_underneath {
                trace!(
                    "skipping direct scan-out on {:?} with zpos {:?}, element {:?} geometry {:?} overlaps with plane underneath",
                    plane.handle, plane.zpos, element_id, element_config.geometry,
                );
                return Err(None);
            }

            self.try_assign_plane(element, element_config, plane, scale, frame_state)
        };

        // First try to assign the element to a compatible plane, this can save us
        // from some atomic testing
        for plane in self.planes.overlay.iter().filter(is_plane_compatible) {
            if let Ok(plane_assignment) = test_overlay_plane(plane, &element_config) {
                trace!(
                    "assigned element {:?} geometry {:?} to compatible {:?} with zpos {:?}",
                    element_id, element_config.geometry, plane.handle, plane.zpos,
                );
                return Ok(plane_assignment);
            }
        }

        // If we found no compatible plane fall back to walk all available planes
        let mut rendering_reason: Option<RenderingReason> = None;
        for (index, plane) in self.planes.overlay.iter().enumerate() {
            // if the tested element state already tells us that this failed skip the test
            if element_config.failed_planes.overlay_bitmask & (1 << index) != 0 {
                trace!(
                    "skipping direct scan-out on {:?} with zpos {:?}, element {:?} geometry {:?}, test already known to fail",
                    plane.handle, plane.zpos, element_id, element_config.geometry,
                );
                rendering_reason = rendering_reason.or(Some(RenderingReason::ScanoutFailed));
                continue;
            }

            match test_overlay_plane(plane, &element_config) {
                Ok(plane) => return Ok(plane),
                Err(err) => {
                    // if the test failed save that in the tested element state
                    if let Some(RenderingReason::ScanoutFailed) = err {
                        element_config.failed_planes.overlay_bitmask |= 1 << index;
                    }

                    rendering_reason = rendering_reason.or(err)
                }
            }
        }

        Err(rendering_reason)
    }

    #[allow(clippy::too_many_arguments)]
    #[instrument(level = "trace", skip_all)]
    #[profiling::function]
    fn try_assign_plane<R, E>(
        &self,
        element: &E,
        element_config: &ElementPlaneConfig<
            '_,
            <A as Allocator>::Buffer,
            <F as ExportFramebuffer<<A as Allocator>::Buffer>>::Framebuffer,
        >,
        plane: &PlaneInfo,
        scale: Scale<f64>,
        frame_state: &mut CompositorFrameState<A, F>,
    ) -> Result<PlaneAssignment, Option<RenderingReason>>
    where
        R: Renderer,
        E: RenderElement<R>,
    {
        let element_id = element.id();

        let plane_claim = match self.surface.claim_plane(plane.handle) {
            Some(claim) => claim,
            None => {
                trace!("failed to claim {:?} for element {:?}", plane.handle, element_id);
                return Err(None);
            }
        };

        // Try to assign the element to a plane
        trace!(
            "testing direct scan-out for element {:?} on {:?} with zpos {:?}: fb: {:?}, element_geometry: {:?}",
            element_id, plane.handle, plane.zpos, &element_config.buffer.fb, element_config.geometry
        );

        if !plane.formats.contains(&element_config.properties.format) {
            trace!(
                "skipping direct scan-out on {:?} with zpos {:?} for element {:?}, format {:?} not supported",
                plane.handle, plane.zpos, element_id, element_config.properties.format,
            );
            return Err(Some(RenderingReason::FormatUnsupported));
        }

        let previous_state = self
            .pending_frame
            .as_ref()
            .map(|pending| &pending.frame)
            .unwrap_or(&self.current_frame);

        let previous_commit = previous_state.plane_state(plane.handle).and_then(|state| {
            state.element_state.as_ref().and_then(|state| {
                if state.id == *element_id {
                    Some(state.commit)
                } else {
                    None
                }
            })
        });

        let element_damage = element.damage_since(scale, previous_commit);
        let has_element_damage = !element_damage.is_empty();

        // Damage were applied buffer transform to be in physical-space. We need to invert it to go
        // back to buffer-coordinate. We'll apply the same transform to the element geometry for
        // scale computation as it's already in physical space.
        let transform = element.transform().invert();
        let damage_clips = if has_element_damage {
            PlaneDamageClips::from_damage(
                self.surface.device_fd(),
                element_config.properties.src,
                element_config.geometry,
                transform,
                transform,
                element_damage,
            )
            .ok()
            .flatten()
        } else {
            None
        };

        let config = PlaneConfig {
            properties: element_config.properties,
            buffer: element_config.buffer.clone(),
            damage_clips,
            plane_claim,
            sync: element_config
                .buffer
                .buffer
                .acquire_point(self.signaled_fence.as_ref()),
        };

        let is_compatible = previous_state
            .plane_state(plane.handle)
            .map(|state| {
                state
                    .config
                    .as_ref()
                    .map(|c| c.is_compatible(&config))
                    .unwrap_or(false)
            })
            .unwrap_or(false);

        // We can only skip the plane update if we have no damage and if
        // the src/dst/alpha properties are unchanged. Also we can not skip if
        // the fb did change (this includes the case where we previously
        // had not assigned anything to the plane)
        let skip = !has_element_damage
            && previous_state
                .plane_state(plane.handle)
                .map(|state| {
                    state
                        .config
                        .as_ref()
                        .map(|c| is_compatible && c.buffer.fb == config.buffer.fb)
                        .unwrap_or(false)
                })
                .unwrap_or(false);

        let plane_state = PlaneState {
            skip,
            needs_test: true,
            element_state: Some(PlaneElementState {
                id: element_id.clone(),
                commit: element.current_commit(),
                z_index: element_config.z_index,
                cursor_size: None,
            }),
            config: Some(config),
        };

        let res = if is_compatible {
            trace!(
                "skipping atomic test for compatible element {:?} on {:?} with zpos {:?}",
                element_id, plane.handle, plane.zpos,
            );
            frame_state.set_state(plane.handle, plane_state);
            true
        } else {
            frame_state
                .test_state(
                    &self.surface,
                    self.supports_fencing,
                    plane.handle,
                    plane_state,
                    false,
                )
                .is_ok()
        };

        if res {
            trace!(
                "successfully assigned element {:?} to {:?} with zpos {:?} for direct scan-out",
                element_id, plane.handle, plane.zpos,
            );

            Ok(plane.into())
        } else {
            trace!(
                "skipping direct scan-out on {:?} with zpos {:?} for element {:?}, test failed",
                plane.handle, plane.zpos, element_id
            );

            Err(Some(RenderingReason::ScanoutFailed))
        }
    }

    /// Clear the surface, setting DPMS state to off, disabling all planes,
    /// and clearing the pending frame.
    ///
    /// Calling [`queue_frame`][Self::queue_frame] will re-enable.
    pub fn clear(&mut self) -> Result<(), DrmError> {
        self.disable_legacy_cursor();
        self.invalidate_legacy_cursor_lifecycle();
        self.surface.clear()?;

        self.current_frame
            .planes
            .iter_mut()
            .for_each(|(_, state)| *state = Default::default());
        self.pending_frame = None;
        self.queued_frame = None;
        self.next_frame = None;

        Ok(())
    }
}

#[inline]
fn apply_underlying_storage_transform(
    element_transform: Transform,
    storage: &UnderlyingStorage<'_>,
) -> Transform {
    match storage {
        UnderlyingStorage::Wayland(buffer) => {
            if buffer_y_inverted(buffer).unwrap_or(false) {
                match element_transform {
                    Transform::Normal => Transform::Flipped,
                    Transform::_90 => Transform::Flipped90,
                    Transform::_180 => Transform::Flipped180,
                    Transform::_270 => Transform::Flipped270,
                    Transform::Flipped => Transform::Normal,
                    Transform::Flipped90 => Transform::_90,
                    Transform::Flipped180 => Transform::_180,
                    Transform::Flipped270 => Transform::_270,
                }
            } else {
                element_transform
            }
        }
        UnderlyingStorage::Memory { .. } => element_transform,
    }
}

#[inline]
fn apply_output_transform(transform: Transform, output_transform: Transform) -> Transform {
    match (transform, output_transform) {
        (Transform::Normal, output_transform) => output_transform,

        (Transform::_90, Transform::Normal) => Transform::_270,
        (Transform::_90, Transform::_90) => Transform::Normal,
        (Transform::_90, Transform::_180) => Transform::_90,
        (Transform::_90, Transform::_270) => Transform::_180,
        (Transform::_90, Transform::Flipped) => Transform::Flipped270,
        (Transform::_90, Transform::Flipped90) => Transform::Flipped,
        (Transform::_90, Transform::Flipped180) => Transform::Flipped90,
        (Transform::_90, Transform::Flipped270) => Transform::Flipped180,

        (Transform::_180, Transform::Normal) => Transform::_180,
        (Transform::_180, Transform::_90) => Transform::_270,
        (Transform::_180, Transform::_180) => Transform::Normal,
        (Transform::_180, Transform::_270) => Transform::_90,
        (Transform::_180, Transform::Flipped) => Transform::Flipped180,
        (Transform::_180, Transform::Flipped90) => Transform::Flipped270,
        (Transform::_180, Transform::Flipped180) => Transform::Flipped,
        (Transform::_180, Transform::Flipped270) => Transform::Flipped90,

        (Transform::_270, Transform::Normal) => Transform::_90,
        (Transform::_270, Transform::_90) => Transform::_180,
        (Transform::_270, Transform::_180) => Transform::_270,
        (Transform::_270, Transform::_270) => Transform::Normal,
        (Transform::_270, Transform::Flipped) => Transform::Flipped90,
        (Transform::_270, Transform::Flipped90) => Transform::Flipped180,
        (Transform::_270, Transform::Flipped180) => Transform::Flipped270,
        (Transform::_270, Transform::Flipped270) => Transform::Flipped,

        (Transform::Flipped, Transform::Normal) => Transform::Flipped,
        (Transform::Flipped, Transform::_90) => Transform::Flipped90,
        (Transform::Flipped, Transform::_180) => Transform::Flipped180,
        (Transform::Flipped, Transform::_270) => Transform::Flipped270,
        (Transform::Flipped, Transform::Flipped) => Transform::Normal,
        (Transform::Flipped, Transform::Flipped90) => Transform::_90,
        (Transform::Flipped, Transform::Flipped180) => Transform::_180,
        (Transform::Flipped, Transform::Flipped270) => Transform::_270,

        (Transform::Flipped90, Transform::Normal) => Transform::Flipped270,
        (Transform::Flipped90, Transform::_90) => Transform::Flipped,
        (Transform::Flipped90, Transform::_180) => Transform::Flipped90,
        (Transform::Flipped90, Transform::_270) => Transform::Flipped180,
        (Transform::Flipped90, Transform::Flipped) => Transform::_270,
        (Transform::Flipped90, Transform::Flipped90) => Transform::Normal,
        (Transform::Flipped90, Transform::Flipped180) => Transform::_90,
        (Transform::Flipped90, Transform::Flipped270) => Transform::_180,

        (Transform::Flipped180, Transform::Normal) => Transform::Flipped180,
        (Transform::Flipped180, Transform::_90) => Transform::Flipped270,
        (Transform::Flipped180, Transform::_180) => Transform::Flipped,
        (Transform::Flipped180, Transform::_270) => Transform::Flipped90,
        (Transform::Flipped180, Transform::Flipped) => Transform::_180,
        (Transform::Flipped180, Transform::Flipped90) => Transform::_270,
        (Transform::Flipped180, Transform::Flipped180) => Transform::Normal,
        (Transform::Flipped180, Transform::Flipped270) => Transform::_90,

        (Transform::Flipped270, Transform::Normal) => Transform::Flipped90,
        (Transform::Flipped270, Transform::_90) => Transform::Flipped180,
        (Transform::Flipped270, Transform::_180) => Transform::Flipped270,
        (Transform::Flipped270, Transform::_270) => Transform::Flipped,
        (Transform::Flipped270, Transform::Flipped) => Transform::_90,
        (Transform::Flipped270, Transform::Flipped90) => Transform::_180,
        (Transform::Flipped270, Transform::Flipped180) => Transform::_270,
        (Transform::Flipped270, Transform::Flipped270) => Transform::Normal,
    }
}

fn render_legacy_cursor_buffer<R, E, G>(
    cursor_state: &mut CursorState<G>,
    renderer: &mut R,
    element: &E,
    element_geometry: Rectangle<i32, Physical>,
    cursor_size: Size<i32, Physical>,
    output_transform: Transform,
) -> Option<GbmBuffer>
where
    R: Renderer,
    E: RenderElement<R>,
    G: AsFd + Clone,
{
    let element_size = output_transform.transform_size(element_geometry.size);
    let mut cursor_buffer = cursor_state
        .allocator
        .create_buffer(
            cursor_size.w as u32,
            cursor_size.h as u32,
            DrmFourcc::Argb8888,
            &[DrmModifier::Linear],
        )
        .ok()?;

    if copy_element_to_cursor_bo(
        renderer,
        element,
        element_size,
        cursor_size,
        output_transform,
        &mut cursor_buffer,
    ) {
        return Some(cursor_buffer);
    }

    #[cfg(not(feature = "renderer_pixman"))]
    return None;

    #[cfg(feature = "renderer_pixman")]
    {
        let storage = element.underlying_storage(renderer)?;
        let pixman_renderer = cursor_state.pixman_renderer.as_mut()?;
        let cursor_texture = match storage {
            UnderlyingStorage::Wayland(buffer) => pixman_renderer
                .import_buffer(buffer, None, &[element.src().to_i32_up()])
                .transpose()
                .ok()
                .flatten(),
            UnderlyingStorage::Memory(memory) => {
                let format = memory.format();
                let size = memory.size();
                let pixman_format = pixman::FormatCode::try_from(format).ok()?;
                unsafe {
                    pixman::Image::from_raw_mut(
                        pixman_format,
                        size.w as usize,
                        size.h as usize,
                        memory.as_ptr() as *mut u32,
                        memory.stride() as usize,
                        false,
                    )
                    .ok()
                    .map(PixmanTexture::from)
                }
            }
        }?;

        let cursor_buffer_size = cursor_size.to_logical(1).to_buffer(1, Transform::Normal);
        let ret = cursor_buffer
            .map_mut::<_, Result<_, PixmanError>>(
                0,
                0,
                cursor_buffer_size.w as u32,
                cursor_buffer_size.h as u32,
                |mbo| {
                    let plane_pixman_format = pixman::FormatCode::try_from(DrmFourcc::Argb8888).unwrap();
                    let mut cursor_dst = unsafe {
                        pixman::Image::from_raw_mut(
                            plane_pixman_format,
                            mbo.width() as usize,
                            mbo.height() as usize,
                            mbo.buffer_mut().as_mut_ptr() as *mut u32,
                            mbo.stride() as usize,
                            false,
                        )
                    }
                    .map_err(|_| PixmanError::ImportFailed)?;
                    let mut framebuffer = pixman_renderer.bind(&mut cursor_dst)?;
                    let mut frame = pixman_renderer.render(&mut framebuffer, cursor_size, output_transform)?;
                    frame.clear(Color32F::TRANSPARENT, &[Rectangle::from_size(cursor_size)])?;
                    let src = element.src();
                    let dst = Rectangle::from_size(element_geometry.size);
                    frame.render_texture_from_to(
                        &cursor_texture,
                        src,
                        dst,
                        &[dst],
                        &[],
                        element.transform(),
                        element.alpha(),
                    )?;
                    let _ = frame.finish()?.wait();
                    Ok(())
                },
            )
            .ok()?;
        ret.ok()?;
        Some(cursor_buffer)
    }
}

#[profiling::function]
fn copy_element_to_cursor_bo<R, E>(
    renderer: &mut R,
    element: &E,
    element_size: Size<i32, Physical>,
    cursor_size: Size<i32, Physical>,
    output_transform: Transform,
    bo: &mut GbmBuffer,
) -> bool
where
    R: Renderer,
    E: RenderElement<R>,
{
    // Without access to the underlying storage we can not copy anything
    let Some(underlying_storage) = element.underlying_storage(renderer) else {
        return false;
    };

    let element_src = element.src();
    let element_scale = element_src.size / element_size.to_f64();

    // We only copy if no crop, scale or transform is active
    if element_src.loc != Point::default()
        || element_scale != Scale::from(1f64)
        || element.transform() != Transform::Normal
        || output_transform != Transform::Normal
    {
        return false;
    }

    let bo_format = bo.format().code;
    let bo_stride = bo.stride();

    let mut copy_to_bo = |src, src_stride, src_height| {
        if src_stride == bo_stride as i32 {
            bo.write(src).is_ok()
        } else {
            let res = bo.map_mut(0, 0, cursor_size.w as u32, cursor_size.h as u32, |mbo| {
                let dst = mbo.buffer_mut();
                for row in 0..src_height {
                    let src_row_start = (row * src_stride) as usize;
                    let src_row_end = src_row_start + src_stride as usize;
                    let src_row = &src[src_row_start..src_row_end];
                    let dst_row_start = (row * bo_stride as i32) as usize;
                    let dst_row_end = dst_row_start + src_stride as usize;
                    let dst_row = &mut dst[dst_row_start..dst_row_end];
                    dst_row.copy_from_slice(src_row);
                }
            });
            res.is_ok()
        }
    };

    match underlying_storage {
        UnderlyingStorage::Wayland(buffer) => {
            // Only shm buffers are supported for copy
            shm::with_buffer_contents(buffer, |ptr, len, data| {
                let Some(format) = shm::shm_format_to_fourcc(data.format) else {
                    return false;
                };

                if format != bo_format {
                    return false;
                };

                let expected_len = (data.stride * data.height) as usize;
                if data.offset as usize + expected_len > len {
                    return false;
                };

                copy_to_bo(
                    unsafe { std::slice::from_raw_parts(ptr.offset(data.offset as isize), expected_len) },
                    data.stride,
                    data.height,
                )
            })
            .unwrap_or(false)
        }
        UnderlyingStorage::Memory(memory) => {
            if memory.format() != bo_format {
                return false;
            };

            copy_to_bo(memory, memory.stride(), memory.size().h)
        }
    }
}

struct CachedDrmFramebuffer<B: Framebuffer>(Arc<DrmFramebuffer<B>>);

impl<B: Framebuffer> PartialEq for CachedDrmFramebuffer<B> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        AsRef::<framebuffer::Handle>::as_ref(&self) == AsRef::<framebuffer::Handle>::as_ref(&other)
    }
}

impl<B: Framebuffer + std::fmt::Debug> std::fmt::Debug for CachedDrmFramebuffer<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("CachedDrmFramebuffer").field(&self.0).finish()
    }
}

impl<B: Framebuffer> CachedDrmFramebuffer<B> {
    #[inline]
    fn new(buffer: DrmFramebuffer<B>) -> Self {
        CachedDrmFramebuffer(Arc::new(buffer))
    }
}

impl<B: Framebuffer> Clone for CachedDrmFramebuffer<B> {
    #[inline]
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<B: Framebuffer> AsRef<framebuffer::Handle> for CachedDrmFramebuffer<B> {
    #[inline]
    fn as_ref(&self) -> &framebuffer::Handle {
        (*self.0).as_ref()
    }
}

impl<B: Framebuffer> Framebuffer for CachedDrmFramebuffer<B> {
    #[inline]
    fn format(&self) -> drm_fourcc::DrmFormat {
        (*self.0).format()
    }
}

/// Errors thrown by a [`DrmCompositor`]
#[derive(Debug, thiserror::Error)]
pub enum FrameError<
    A: std::error::Error + Send + Sync + 'static,
    B: std::error::Error + Send + Sync + 'static,
    F: std::error::Error + Send + Sync + 'static,
> {
    /// Failed to claim the primary plane
    #[error("Failed to claim the primary plane")]
    PrimaryPlaneClaimFailed,
    /// No supported pixel format for the given plane could be determined
    #[error("No supported plane buffer format found")]
    NoSupportedPlaneFormat,
    /// No supported pixel format for the given renderer could be determined
    #[error("No supported renderer buffer format found")]
    NoSupportedRendererFormat,
    /// The swapchain is exhausted, you need to call `frame_submitted`
    #[error("Failed to allocate a new buffer")]
    NoFreeSlotsError,
    /// Error accessing the drm device
    #[error("The underlying drm surface encountered an error: {0}")]
    DrmError(#[from] DrmError),
    /// Error during buffer allocation
    #[error("The underlying allocator encountered an error: {0}")]
    Allocator(#[source] A),
    /// Error during exporting the buffer as dmabuf
    #[error("Failed to export the allocated buffer as dmabuf: {0}")]
    AsDmabufError(#[source] B),
    /// Error during exporting a framebuffer
    #[error("The framebuffer export encountered an error: {0}")]
    FramebufferExport(#[source] F),
    /// No framebuffer available
    #[error("No framebuffer available")]
    NoFramebuffer,
    /// The frame is empty
    ///
    /// Possible reasons include not calling `render_frame` prior to
    /// `queue_frame` or trying to queue a frame without changes.
    #[error("No frame has been prepared or it does not contain any changes")]
    EmptyFrame,
}

/// Error returned from [`DrmCompositor::render_frame`]
#[derive(thiserror::Error)]
pub enum RenderFrameError<
    A: std::error::Error + Send + Sync + 'static,
    B: std::error::Error + Send + Sync + 'static,
    F: std::error::Error + Send + Sync + 'static,
    R: std::error::Error,
> {
    /// Preparing the frame encountered an error
    #[error(transparent)]
    PrepareFrame(#[from] FrameError<A, B, F>),
    /// Rendering the frame encountered en error
    #[error(transparent)]
    RenderFrame(#[from] OutputDamageTrackerError<R>),
}

impl<A, B, F, R> std::fmt::Debug for RenderFrameError<A, B, F, R>
where
    A: std::error::Error + Send + Sync + 'static,
    B: std::error::Error + Send + Sync + 'static,
    F: std::error::Error + Send + Sync + 'static,
    R: std::error::Error,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PrepareFrame(arg0) => f.debug_tuple("PrepareFrame").field(arg0).finish(),
            Self::RenderFrame(arg0) => f.debug_tuple("RenderFrame").field(arg0).finish(),
        }
    }
}

impl<
    A: std::error::Error + Send + Sync + 'static,
    B: std::error::Error + Send + Sync + 'static,
    F: std::error::Error + Send + Sync + 'static,
> From<FrameError<A, B, F>> for SwapBuffersError
{
    #[inline]
    fn from(err: FrameError<A, B, F>) -> SwapBuffersError {
        match err {
            x @ FrameError::NoSupportedPlaneFormat
            | x @ FrameError::NoSupportedRendererFormat
            | x @ FrameError::PrimaryPlaneClaimFailed
            | x @ FrameError::NoFramebuffer => SwapBuffersError::ContextLost(Box::new(x)),
            x @ FrameError::NoFreeSlotsError | x @ FrameError::EmptyFrame => {
                SwapBuffersError::TemporaryFailure(Box::new(x))
            }
            FrameError::DrmError(err) => err.into(),
            FrameError::Allocator(err) => SwapBuffersError::ContextLost(Box::new(err)),
            FrameError::AsDmabufError(err) => SwapBuffersError::ContextLost(Box::new(err)),
            FrameError::FramebufferExport(err) => SwapBuffersError::ContextLost(Box::new(err)),
        }
    }
}

fn nvidia_drm_version() -> Option<(u32, u32, u32)> {
    let ver = std::fs::read_to_string("/sys/module/nvidia_drm/version").ok()?;
    let mut components = ver.trim().split('.');
    let major = u32::from_str(components.next()?).ok()?;
    let minor = u32::from_str(components.next()?).ok()?;
    let patch = u32::from_str(components.next()?).ok()?;
    Some((major, minor, patch))
}

#[test]
fn drm_compositor_is_send() {
    use std::marker::PhantomData;

    use crate::backend::drm::DrmDeviceFd;

    fn is_send<T: Send>() {
        let _ = PhantomData::<T>;
    }

    is_send::<DrmCompositor<GbmAllocator<DrmDeviceFd>, GbmFramebufferExporter<DrmDeviceFd>, (), DrmDeviceFd>>(
    );
}

#[cfg(test)]
mod legacy_cursor_tests {
    use std::cell::RefCell;

    use super::{
        FrameFlags, LegacyCursorInstallResult, LegacyCursorIo, cursor_plane_location,
        install_legacy_cursor,
    };
    use crate::utils::{Physical, Point, Rectangle, Size, Transform};

    #[derive(Default)]
    struct MockIo {
        calls: RefCell<Vec<&'static str>>,
        disable_error: Option<i32>,
        install_error: Option<i32>,
        move_error: Option<i32>,
    }

    impl LegacyCursorIo<()> for MockIo {
        fn disable(&self) -> std::io::Result<()> {
            self.calls.borrow_mut().push("disable");
            self.disable_error
                .map(std::io::Error::from_raw_os_error)
                .map_or(Ok(()), Err)
        }

        fn install(&self, _buffer: &()) -> std::io::Result<()> {
            self.calls.borrow_mut().push("install");
            self.install_error
                .map(std::io::Error::from_raw_os_error)
                .map_or(Ok(()), Err)
        }

        fn move_to(&self, _origin: Point<i32, Physical>) -> std::io::Result<()> {
            self.calls.borrow_mut().push("move");
            self.move_error
                .map(std::io::Error::from_raw_os_error)
                .map_or(Ok(()), Err)
        }
    }

    fn install(io: &MockIo, replacing: bool, succeeded: bool) -> LegacyCursorInstallResult {
        install_legacy_cursor(io, &(), Point::from((10, 20)), replacing, succeeded)
    }

    #[test]
    fn first_install_sets_then_places() {
        let io = MockIo::default();
        assert_eq!(install(&io, false, false), LegacyCursorInstallResult::Installed);
        assert_eq!(&*io.calls.borrow(), &["install", "move"]);
    }

    #[test]
    fn replacement_disables_moves_then_installs() {
        let io = MockIo::default();
        assert_eq!(install(&io, true, true), LegacyCursorInstallResult::Installed);
        assert_eq!(&*io.calls.borrow(), &["disable", "move", "install"]);
    }

    #[test]
    fn failed_disable_preserves_the_active_cursor() {
        let io = MockIo {
            disable_error: Some(16),
            ..Default::default()
        };
        assert_eq!(
            install(&io, true, true),
            LegacyCursorInstallResult::PreserveActive
        );
        assert_eq!(&*io.calls.borrow(), &["disable"]);
    }

    #[test]
    fn only_a_permanent_first_install_releases_the_plane_to_atomic() {
        let permanent = MockIo {
            install_error: Some(22),
            ..Default::default()
        };
        assert_eq!(
            install(&permanent, false, false),
            LegacyCursorInstallResult::Atomic
        );

        let transient = MockIo {
            install_error: Some(16),
            ..Default::default()
        };
        assert_eq!(
            install(&transient, false, false),
            LegacyCursorInstallResult::CandidateSoftware
        );

        let after_success = MockIo {
            install_error: Some(22),
            ..Default::default()
        };
        assert_eq!(
            install(&after_success, false, true),
            LegacyCursorInstallResult::DisabledSoftware
        );
    }

    #[test]
    fn failed_initial_placement_cleans_up_the_installed_cursor() {
        let io = MockIo {
            move_error: Some(16),
            ..Default::default()
        };
        assert_eq!(
            install(&io, false, false),
            LegacyCursorInstallResult::DisabledSoftware
        );
        assert_eq!(&*io.calls.borrow(), &["install", "move", "disable"]);
    }

    #[test]
    fn placement_and_default_flags_share_the_legacy_contract() {
        let output = Rectangle::from_size(Size::from((100, 80)));
        assert_eq!(
            cursor_plane_location(
                Point::from((12, 7)),
                Size::from((64, 64)),
                output,
                Transform::Normal,
            ),
            Point::from((12, 7)),
        );
        assert!(FrameFlags::DEFAULT.contains(FrameFlags::ALLOW_LEGACY_CURSOR));
        assert!(!FrameFlags::empty().contains(FrameFlags::ALLOW_LEGACY_CURSOR));
    }
}

/// DRIFT-1427. An element below a framebuffer effect it overlaps must be composited
/// into the primary framebuffer, because that framebuffer is what the effect samples.
///
/// These drive the two halves the way `render_frame` drives them, rather than
/// restating them: the scan is the function production calls, and the composed cases
/// walk a whole ordered stack. The remaining untested surface is the one `map`
/// closure that adapts `output_elements` and the threading of the result into
/// `try_assign_element`, which cannot be reached without a live `DrmSurface`,
/// allocator and plane set. Faking one would be a fake.
#[cfg(test)]
mod framebuffer_effect_barrier_tests {
    use super::{below_framebuffer_effect, collect_framebuffer_effect_regions};
    use crate::utils::{Physical, Point, Rectangle, Size};

    fn rect(x: i32, y: i32, w: i32, h: i32) -> Rectangle<i32, Physical> {
        Rectangle::new(Point::from((x, y)), Size::from((w, h)))
    }

    /// The whole screen, so a stack built from it overlaps everywhere.
    fn full() -> Rectangle<i32, Physical> {
        rect(0, 0, 100, 100)
    }

    /// Drive the production scan the way `render_frame` does, into a caller-owned
    /// buffer, and hand back what it filled.
    fn scan(
        elements: impl IntoIterator<Item = (bool, Rectangle<i32, Physical>)>,
    ) -> Vec<(usize, Rectangle<i32, Physical>)> {
        let mut out = Vec::new();
        collect_framebuffer_effect_regions(&mut out, elements);
        out
    }

    #[test]
    fn the_scan_keeps_every_effect_in_front_to_back_order() {
        let ordinary = rect(0, 0, 10, 10);
        let effect = rect(1, 1, 20, 20);

        assert_eq!(
            scan([(false, ordinary), (true, effect), (false, ordinary)]),
            [(1, effect)],
            "the single effect is kept at its own index",
        );
        assert!(
            scan([(false, ordinary), (false, ordinary)]).is_empty(),
            "no effect on screen must never suppress anything",
        );
        assert_eq!(
            scan([(true, effect), (false, ordinary), (false, ordinary)]),
            [(0, effect)],
        );
        // The discriminator: keeping only the front-most (`position`) or only the
        // rear-most (`rposition`) would let the element BETWEEN two stacked effects
        // through, because it is below one of them.
        let lower_effect = rect(2, 2, 30, 30);
        assert_eq!(
            scan([(true, effect), (false, ordinary), (true, lower_effect)]),
            [(0, effect), (2, lower_effect)],
            "both stacked effects are kept, not just one",
        );
    }

    #[test]
    fn the_scan_refills_the_callers_buffer_and_never_replaces_it() {
        // The point of taking `&mut Vec` is that the CALLER's allocation survives, so
        // a steady frame does not allocate. Seed a capacity far above what the data
        // needs and assert it is still there afterwards: a body that reassigned
        // (`*out = ...collect()`) would drop it to the collection's own capacity, and
        // would otherwise pass every assertion here, since a two-element fill and a
        // fresh collect both land on the same small capacity.
        let effect = full();
        let stack = [(false, full()), (true, effect), (true, effect)];
        let mut buffer = Vec::with_capacity(64);
        let seeded = buffer.capacity();
        assert!(seeded >= 64, "the seed must exceed what the fill needs");

        for _ in 0..8 {
            collect_framebuffer_effect_regions(&mut buffer, stack);
            assert_eq!(buffer.len(), 2, "each pass refills from scratch");
            assert_eq!(
                buffer.capacity(),
                seeded,
                "the caller's allocation must be reused, not replaced",
            );
        }

        // And a frame with no effects empties it rather than leaving the last one.
        collect_framebuffer_effect_regions(&mut buffer, [(false, full())]);
        assert!(buffer.is_empty(), "the buffer carries no state between frames");
        assert_eq!(buffer.capacity(), seeded, "emptying must not drop the allocation");
    }

    #[test]
    fn below_is_a_greater_index_and_an_overlap() {
        let effect = full();
        let regions = [(1usize, effect)];

        assert!(
            below_framebuffer_effect(2, full(), &regions),
            "an element after the effect is below it",
        );
        assert!(
            !below_framebuffer_effect(0, full(), &regions),
            "an element before the effect is above it",
        );
        assert!(
            !below_framebuffer_effect(1, full(), &regions),
            "the effect's own index is not below itself; it is refused one line earlier",
        );
        assert!(
            !below_framebuffer_effect(2, full(), &[]),
            "with no effect on screen nothing is below one",
        );
    }

    #[test]
    fn a_lower_element_clear_of_the_effect_is_left_alone() {
        let regions = [(0usize, rect(0, 0, 10, 10))];

        assert!(
            !below_framebuffer_effect(1, rect(50, 50, 10, 10), &regions),
            "a lower element that does not overlap the effect is not sampled by it",
        );
        assert!(
            !below_framebuffer_effect(1, rect(10, 0, 10, 10), &regions),
            "edge-adjacent contributes no pixels inside the effect's region",
        );
        assert!(
            below_framebuffer_effect(1, rect(9, 0, 10, 10), &regions),
            "one column of genuine overlap is enough",
        );
    }

    #[test]
    fn the_two_composed_answer_the_ticket() {
        // The ticket's sentence: a lower element that would otherwise be scanned out
        // is composited when an active backdrop sits above it.
        let stack = [(false, full()), (true, full()), (false, full())];
        let regions = scan(stack);
        let verdicts: Vec<bool> = stack
            .iter()
            .enumerate()
            .map(|(index, (_, geometry))| below_framebuffer_effect(index, *geometry, &regions))
            .collect();
        assert_eq!(
            verdicts,
            vec![false, false, true],
            "only the element below the effect is refused a plane",
        );

        // Same stack, bottom element moved clear of the effect.
        let clear = [
            (false, full()),
            (true, rect(0, 0, 10, 10)),
            (false, rect(50, 50, 10, 10)),
        ];
        let regions = scan(clear);
        let verdicts: Vec<bool> = clear
            .iter()
            .enumerate()
            .map(|(index, (_, geometry))| below_framebuffer_effect(index, *geometry, &regions))
            .collect();
        assert_eq!(
            verdicts,
            vec![false, false, false],
            "a lower element clear of the effect keeps its plane",
        );
    }
}
