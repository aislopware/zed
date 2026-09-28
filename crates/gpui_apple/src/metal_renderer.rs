use crate::{metal_atlas::MetalAtlas, presentation_pacing::PresentationPacer};
use anyhow::{Context as _, Result};
use block2::RcBlock;
use core_graphics::geometry::CGSize;
use gpui::{
    AtlasTextureId, Background, Bounds, ContentMask, DevicePixels, PaintSurface, Path, Point,
    PresentedFrame, PresentedFrameSink, PrimitiveBatch, ScaledPixels, Scene, Size, point, size,
};
#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
use image::RgbaImage;
use objc2::runtime::AnyObject;

#[cfg(any(target_os = "macos", target_os = "ios"))]
use core_foundation::base::TCFType;
#[cfg(any(target_os = "macos", target_os = "ios"))]
use core_foundation::string::CFString;
#[cfg(any(target_os = "macos", target_os = "ios"))]
use core_video::{
    buffer::TCVBuffer,
    image_buffer::{
        kCVImageBufferYCbCrMatrix_ITU_R_601_4, kCVImageBufferYCbCrMatrix_ITU_R_2020,
        kCVImageBufferYCbCrMatrixKey,
    },
    metal_texture::{CVMetalTexture, CVMetalTextureGetTexture},
    metal_texture_cache::CVMetalTextureCache,
    pixel_buffer::{
        CVPixelBuffer, kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
        kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
        kCVPixelFormatType_420YpCbCr10BiPlanarFullRange,
        kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange,
        kCVPixelFormatType_422YpCbCr10BiPlanarFullRange,
        kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange,
        kCVPixelFormatType_444YpCbCr10BiPlanarFullRange,
        kCVPixelFormatType_444YpCbCr10BiPlanarVideoRange,
    },
};
use foreign_types::ForeignType;
#[cfg(any(target_os = "macos", target_os = "ios"))]
use foreign_types::ForeignTypeRef;
use metal::{
    CAMetalLayer, CommandQueue, MTLGPUFamily, MTLPixelFormat, MTLResourceOptions, NSRange,
    NSUInteger,
};
use objc::{
    self, msg_send,
    runtime::{NO, YES},
    sel, sel_impl,
};
use parking_lot::Mutex;

use std::{
    cell::Cell,
    ffi::c_void,
    mem,
    mem::MaybeUninit,
    ops::Range,
    ptr, slice,
    sync::Arc,
    time::{Duration, Instant},
};

// Exported to metal
pub(crate) type PointF = gpui::Point<f32>;

#[cfg(not(feature = "runtime_shaders"))]
const SHADERS_METALLIB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders.metallib"));
#[cfg(feature = "runtime_shaders")]
const SHADERS_SOURCE_FILE: &str = include_str!(concat!(env!("OUT_DIR"), "/stitched_shaders.metal"));
// Use 4x MSAA, all devices support it.
// https://developer.apple.com/documentation/metal/mtldevice/1433355-supportstexturesamplecount
const PATH_SAMPLE_COUNT: u32 = 4;
/// Metal requires the offset a buffer is bound at to be 256-byte aligned.
const INSTANCE_BUFFER_ALIGNMENT: usize = 256;
const MAX_INSTANCE_BUFFER_SIZE: usize = 256 * 1024 * 1024;

pub type Context = Arc<Mutex<InstanceBufferPool>>;
pub type Renderer = MetalRenderer;

pub unsafe fn new_renderer(
    context: self::Context,
    _native_window: *mut c_void,
    _native_view: *mut c_void,
    _bounds: gpui::Size<f32>,
    transparent: bool,
) -> Renderer {
    MetalRenderer::new(context, transparent)
}

pub struct InstanceBufferPool {
    buffer_size: usize,
    buffers: Vec<metal::Buffer>,
}

impl Default for InstanceBufferPool {
    fn default() -> Self {
        Self {
            buffer_size: 2 * 1024 * 1024,
            buffers: Vec::new(),
        }
    }
}

pub(crate) struct InstanceBuffer {
    metal_buffer: metal::Buffer,
    size: usize,
}

impl InstanceBufferPool {
    pub(crate) fn reset(&mut self, buffer_size: usize) {
        self.buffer_size = buffer_size;
        self.buffers.clear();
    }

    pub(crate) fn acquire(
        &mut self,
        device: &metal::Device,
        unified_memory: bool,
    ) -> InstanceBuffer {
        let buffer = self.buffers.pop().unwrap_or_else(|| {
            let options = if unified_memory {
                MTLResourceOptions::StorageModeShared
                    // Buffers are write only which can benefit from the combined cache
                    // https://developer.apple.com/documentation/metal/mtlresourceoptions/cpucachemodewritecombined
                    | MTLResourceOptions::CPUCacheModeWriteCombined
            } else {
                MTLResourceOptions::StorageModeManaged
            };

            device.new_buffer(self.buffer_size as u64, options)
        });
        InstanceBuffer {
            metal_buffer: buffer,
            size: self.buffer_size,
        }
    }

    pub(crate) fn release(&mut self, buffer: InstanceBuffer) {
        if buffer.size == self.buffer_size {
            self.buffers.push(buffer.metal_buffer)
        }
    }
}

/// Until the platform names the display's refresh, assume the fastest Apple panel's.
const DEFAULT_REFRESH: Duration = Duration::from_micros(8_333);

pub struct MetalRenderer {
    device: metal::Device,
    layer: Option<metal::MetalLayer>,
    is_apple_gpu: bool,
    is_unified_memory: bool,
    presents_with_transaction: bool,
    /// Receives each drawn frame once the display has shown it; see [`Self::draw`].
    presented_frame_sink: Option<PresentedFrameSink>,
    /// Frames the display has shown or dropped since [`Self::ready_for_vsync_frame`] last
    /// looked, filled from a Metal thread.
    presentations: Arc<Mutex<Vec<PresentedFrame>>>,
    pacer: PresentationPacer,
    /// Whether this layer's drawables report their presentation, learned from the first one.
    /// The iOS simulator's `CAMetalDrawable` implements neither `addPresentedHandler:` nor
    /// `presentedTime`; there no frame is reported, rather than a made-up instant, and the
    /// pacer never sees a queue.
    drawables_report_presentation: Option<bool>,
    /// For headless rendering, tracks whether output should be opaque
    opaque: bool,
    command_queue: CommandQueue,
    paths_rasterization_pipeline_state: metal::RenderPipelineState,
    path_sprites_pipeline_state: metal::RenderPipelineState,
    shadows_pipeline_state: metal::RenderPipelineState,
    quads_pipeline_state: metal::RenderPipelineState,
    underlines_pipeline_state: metal::RenderPipelineState,
    monochrome_sprites_pipeline_state: metal::RenderPipelineState,
    polychrome_sprites_pipeline_state: metal::RenderPipelineState,
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    surfaces_pipeline_state: metal::RenderPipelineState,
    unit_vertices: metal::Buffer,
    #[allow(clippy::arc_with_non_send_sync)]
    instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
    sprite_atlas: Arc<MetalAtlas>,
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    core_video_texture_cache: core_video::metal_texture_cache::CVMetalTextureCache,
    /// The textures the frame being encoded samples. CoreVideo may hand a texture's backing
    /// buffer to a new picture once the `CVMetalTexture` is released, so they ride to the command
    /// buffer's completion handler instead of dying with the draw call.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    surface_textures: Vec<CVMetalTexture>,
    /// A surface was skipped and said so; later ones stay quiet.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    surface_skip_logged: bool,
    path_intermediate_texture: Option<metal::Texture>,
    path_intermediate_msaa_texture: Option<metal::Texture>,
    path_sample_count: u32,
    /// Offscreen render target reused across `render_scene` calls when
    /// rendering headlessly without reading pixels back.
    #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
    headless_render_target: Option<metal::Texture>,
}

/// Records the frame in `drawable` for the pacer, and hands it to `sink`, once the display
/// shows it or learns it was dropped. Must run before the drawable is presented, as
/// `addPresentedHandler:` requires.
fn observe_presentation(
    drawable: &metal::MetalDrawableRef,
    presentations: Arc<Mutex<Vec<PresentedFrame>>>,
    sink: Option<PresentedFrameSink>,
) {
    let submitted_at = Instant::now();
    let handler = RcBlock::new(move |drawable: ptr::NonNull<AnyObject>| {
        // SAFETY: `addPresentedHandler:` calls the block with the `id<MTLDrawable>` it was
        // added to, alive for the duration of the call.
        let drawable = unsafe { metal::DrawableRef::from_ptr(drawable.as_ptr().cast()) };
        let frame = PresentedFrame {
            submitted_at,
            presented_at: host_time_to_instant(drawable.presented_time()),
        };
        presentations.lock().push(frame);
        if let Some(sink) = &sink {
            sink(frame);
        }
    });
    // SAFETY: Both pointee types are opaque views of the same Objective-C block pointer ABI,
    // and `addPresentedHandler:` copies the block before this one is released.
    unsafe {
        drawable.add_presented_handler(&*RcBlock::as_ptr(&handler).cast());
    }
}

/// Converts a Core Animation host time (`CACurrentMediaTime` seconds, as `presentedTime`
/// reports) to an `Instant`. `presentedTime` is zero for a drawable that was never shown.
fn host_time_to_instant(host_time: f64) -> Option<Instant> {
    #[link(name = "QuartzCore", kind = "framework")]
    unsafe extern "C" {
        // QuartzCore/CABase.h: CFTimeInterval CACurrentMediaTime(void).
        fn CACurrentMediaTime() -> f64;
    }
    if host_time <= 0.0 {
        return None;
    }
    let now = Instant::now();
    // SAFETY: A pure function over the host clock, callable from any thread.
    let age = unsafe { CACurrentMediaTime() } - host_time;
    if age >= 0.0 {
        now.checked_sub(Duration::from_secs_f64(age))
    } else {
        now.checked_add(Duration::from_secs_f64(-age))
    }
}

#[repr(C)]
pub struct PathRasterizationVertex {
    pub xy_position: Point<ScaledPixels>,
    pub st_position: Point<f32>,
    pub color: Background,
    pub bounds: Bounds<ScaledPixels>,
}

impl MetalRenderer {
    /// Creates a new MetalRenderer with a CAMetalLayer for window-based rendering.
    pub fn new(instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>, transparent: bool) -> Self {
        let device = Self::create_device();
        let layer = metal::MetalLayer::new();
        Self::configure_layer(&layer, &device, transparent);
        Self::new_internal(device, Some(layer), !transparent, instance_buffer_pool)
    }

    /// Creates a renderer for a CAMetalLayer owned by a platform view.
    ///
    /// # Safety
    ///
    /// `layer` must point to a live CAMetalLayer and must only be used from the
    /// thread on which its owning view may be accessed.
    pub unsafe fn from_layer(
        instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
        layer: *mut CAMetalLayer,
        transparent: bool,
    ) -> Self {
        let device = Self::create_device();
        let retained_layer: *mut CAMetalLayer = unsafe { msg_send![layer, retain] };
        let layer = unsafe { metal::MetalLayer::from_ptr(retained_layer) };
        Self::configure_layer(&layer, &device, transparent);
        Self::new_internal(device, Some(layer), !transparent, instance_buffer_pool)
    }

    fn configure_layer(layer: &metal::MetalLayerRef, device: &metal::DeviceRef, transparent: bool) {
        layer.set_device(device);
        layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        layer.set_opaque(!transparent);
        layer.set_maximum_drawable_count(3);
        #[cfg(any(test, feature = "test-support"))]
        layer.set_framebuffer_only(false);
        unsafe {
            let _: () = msg_send![&*layer, setAllowsNextDrawableTimeout: NO];
            let _: () = msg_send![&*layer, setNeedsDisplayOnBoundsChange: YES];
            #[cfg(target_os = "macos")]
            let _: () = msg_send![&*layer, setAutoresizingMask: 18_u32];
        }
    }

    /// Creates a new headless MetalRenderer for offscreen rendering without a window.
    ///
    /// This renderer can render scenes to images without requiring a CAMetalLayer,
    /// window, or AppKit. Use `render_scene_to_image()` to render scenes.
    #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
    pub fn new_headless(instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>) -> Self {
        let device = Self::create_device();
        Self::new_internal(device, None, true, instance_buffer_pool)
    }

    fn create_device() -> metal::Device {
        #[cfg(target_os = "macos")]
        let device = metal::Device::all()
            .into_iter()
            .min_by_key(|d| (d.is_removable(), !d.is_low_power()))
            .or_else(metal::Device::system_default);
        #[cfg(target_os = "ios")]
        let device = metal::Device::system_default();

        device.unwrap_or_else(|| {
            log::error!("unable to access a compatible Metal device");
            std::process::exit(1);
        })
    }

    fn new_internal(
        device: metal::Device,
        layer: Option<metal::MetalLayer>,
        opaque: bool,
        instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
    ) -> Self {
        #[cfg(feature = "runtime_shaders")]
        let library = device
            .new_library_with_source(&SHADERS_SOURCE_FILE, &metal::CompileOptions::new())
            .expect("error building metal library");
        #[cfg(not(feature = "runtime_shaders"))]
        let library = device
            .new_library_with_data(SHADERS_METALLIB)
            .expect("error building metal library");

        fn to_float2_bits(point: PointF) -> u64 {
            let mut output = point.y.to_bits() as u64;
            output <<= 32;
            output |= point.x.to_bits() as u64;
            output
        }

        // Shared memory can be used only if CPU and GPU share the same memory space.
        // https://developer.apple.com/documentation/metal/setting-resource-storage-modes
        // iOS does not support managed resources. Its simulator may report a
        // non-unified host GPU even though resources must still use shared storage.
        let is_unified_memory = cfg!(target_os = "ios") || device.has_unified_memory();
        // Apple GPU families support memoryless textures, which can significantly reduce
        // memory usage by keeping render targets in on-chip tile memory instead of
        // allocating backing store in system memory.
        // https://developer.apple.com/documentation/metal/mtlgpufamily
        let is_apple_gpu = device.supports_family(MTLGPUFamily::Apple1);

        let unit_vertices = [
            to_float2_bits(point(0., 0.)),
            to_float2_bits(point(1., 0.)),
            to_float2_bits(point(0., 1.)),
            to_float2_bits(point(0., 1.)),
            to_float2_bits(point(1., 0.)),
            to_float2_bits(point(1., 1.)),
        ];
        let unit_vertices = device.new_buffer_with_data(
            unit_vertices.as_ptr() as *const c_void,
            mem::size_of_val(&unit_vertices) as u64,
            if is_unified_memory {
                MTLResourceOptions::StorageModeShared
                    | MTLResourceOptions::CPUCacheModeWriteCombined
            } else {
                MTLResourceOptions::StorageModeManaged
            },
        );

        let paths_rasterization_pipeline_state = build_path_rasterization_pipeline_state(
            &device,
            &library,
            "paths_rasterization",
            "path_rasterization_vertex",
            "path_rasterization_fragment",
            MTLPixelFormat::BGRA8Unorm,
            PATH_SAMPLE_COUNT,
        );
        let path_sprites_pipeline_state = build_path_sprite_pipeline_state(
            &device,
            &library,
            "path_sprites",
            "path_sprite_vertex",
            "path_sprite_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let shadows_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "shadows",
            "shadow_vertex",
            "shadow_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let quads_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "quads",
            "quad_vertex",
            "quad_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let underlines_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "underlines",
            "underline_vertex",
            "underline_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let monochrome_sprites_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "monochrome_sprites",
            "monochrome_sprite_vertex",
            "monochrome_sprite_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let polychrome_sprites_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "polychrome_sprites",
            "polychrome_sprite_vertex",
            "polychrome_sprite_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        let surfaces_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "surfaces",
            "surface_vertex",
            "surface_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );

        let command_queue = device.new_command_queue();
        let supports_shared_storage = cfg!(target_os = "ios") || is_apple_gpu;
        let sprite_atlas = Arc::new(MetalAtlas::new(device.clone(), supports_shared_storage));
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        let core_video_texture_cache =
            CVMetalTextureCache::new(None, device.clone(), None).unwrap();

        Self {
            device,
            layer,
            presents_with_transaction: false,
            presented_frame_sink: None,
            presentations: Arc::default(),
            drawables_report_presentation: None,
            pacer: PresentationPacer::new(DEFAULT_REFRESH),
            is_apple_gpu,
            is_unified_memory,
            opaque,
            command_queue,
            paths_rasterization_pipeline_state,
            path_sprites_pipeline_state,
            shadows_pipeline_state,
            quads_pipeline_state,
            underlines_pipeline_state,
            monochrome_sprites_pipeline_state,
            polychrome_sprites_pipeline_state,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            surfaces_pipeline_state,
            unit_vertices,
            instance_buffer_pool,
            sprite_atlas,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            core_video_texture_cache,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            surface_textures: Vec::new(),
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            surface_skip_logged: false,
            path_intermediate_texture: None,
            path_intermediate_msaa_texture: None,
            path_sample_count: PATH_SAMPLE_COUNT,
            #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
            headless_render_target: None,
        }
    }

    pub fn layer(&self) -> Option<&metal::MetalLayerRef> {
        self.layer.as_ref().map(|l| l.as_ref())
    }

    pub fn layer_ptr(&self) -> *mut CAMetalLayer {
        self.layer
            .as_ref()
            .map(|l| l.as_ptr())
            .unwrap_or(ptr::null_mut())
    }

    pub fn sprite_atlas(&self) -> &Arc<MetalAtlas> {
        &self.sprite_atlas
    }

    pub fn set_presents_with_transaction(&mut self, presents_with_transaction: bool) {
        self.presents_with_transaction = presents_with_transaction;
        if let Some(layer) = &self.layer {
            layer.set_presents_with_transaction(presents_with_transaction);
        }
    }

    /// Reports every frame [`Self::draw`] presents to `sink`, or stops with `None`. The sink
    /// runs on a Metal thread.
    pub fn set_presented_frame_sink(&mut self, sink: Option<PresentedFrameSink>) {
        self.presented_frame_sink = sink;
    }

    /// The display's shortest refresh interval, which paces [`Self::ready_for_vsync_frame`].
    pub fn set_refresh_interval(&mut self, refresh: Duration) {
        self.pacer.set_refresh(refresh);
    }

    /// Whether a display-link tick should draw, or leave its demand to the next tick so the
    /// frames already queued for the display can drain; see `presentation_pacing`.
    pub fn ready_for_vsync_frame(&mut self) -> bool {
        for frame in self.presentations.lock().drain(..) {
            self.pacer.presented(frame);
        }
        self.pacer.should_draw(Instant::now())
    }

    /// Whether no frame began within the last refresh, so one drawn now, off the vsync
    /// tick, cannot land in the same refresh as the previous one.
    pub fn idle_for_a_refresh(&self) -> bool {
        self.pacer.idle_for_a_refresh(Instant::now())
    }

    pub fn update_drawable_size(&mut self, size: Size<DevicePixels>) {
        if let Some(layer) = &self.layer {
            layer.set_drawable_size(CGSize::new(size.width.0 as f64, size.height.0 as f64));
        }
        self.update_path_intermediate_textures(size);
    }

    fn update_path_intermediate_textures(&mut self, size: Size<DevicePixels>) {
        // We are uncertain when this happens, but sometimes size can be 0 here. Most likely before
        // the layout pass on window creation. Zero-sized texture creation causes SIGABRT.
        // https://github.com/zed-industries/zed/issues/36229
        if size.width.0 <= 0 || size.height.0 <= 0 {
            self.path_intermediate_texture = None;
            self.path_intermediate_msaa_texture = None;
            return;
        }

        let texture_descriptor = metal::TextureDescriptor::new();
        texture_descriptor.set_width(size.width.0 as u64);
        texture_descriptor.set_height(size.height.0 as u64);
        texture_descriptor.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
        texture_descriptor.set_storage_mode(metal::MTLStorageMode::Private);
        texture_descriptor
            .set_usage(metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead);
        self.path_intermediate_texture = Some(self.device.new_texture(&texture_descriptor));

        if self.path_sample_count > 1 {
            // https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus
            // Rendering MSAA textures are done in a single pass, so we can use memory-less storage on Apple Silicon
            let storage_mode = if self.is_apple_gpu {
                metal::MTLStorageMode::Memoryless
            } else {
                metal::MTLStorageMode::Private
            };

            let msaa_descriptor = texture_descriptor;
            msaa_descriptor.set_texture_type(metal::MTLTextureType::D2Multisample);
            msaa_descriptor.set_storage_mode(storage_mode);
            msaa_descriptor.set_sample_count(self.path_sample_count as _);
            self.path_intermediate_msaa_texture = Some(self.device.new_texture(&msaa_descriptor));
        } else {
            self.path_intermediate_msaa_texture = None;
        }
    }

    pub fn update_transparency(&mut self, transparent: bool) {
        self.opaque = !transparent;
        if let Some(layer) = &self.layer {
            layer.set_opaque(!transparent);
        }
    }

    pub fn destroy(&self) {
        // nothing to do
    }

    pub fn draw(&mut self, scene: &Scene) {
        let layer = match &self.layer {
            Some(l) => l.clone(),
            None => {
                log::error!(
                    "draw() called on headless renderer - use render_scene_to_image() instead"
                );
                return;
            }
        };
        let viewport_size = layer.drawable_size();
        let viewport_size: Size<DevicePixels> = size(
            (viewport_size.width.ceil() as i32).into(),
            (viewport_size.height.ceil() as i32).into(),
        );
        let drawable = if let Some(drawable) = layer.next_drawable() {
            drawable
        } else {
            log::error!(
                "failed to retrieve next drawable, drawable size: {:?}",
                viewport_size
            );
            return;
        };

        let command_buffer = match self.render_frame(scene, drawable.texture(), viewport_size) {
            Ok(command_buffer) => command_buffer,
            Err(error) => {
                log::error!("failed to render: {error:#}");
                return;
            }
        };

        let reports_presentation = *self.drawables_report_presentation.get_or_insert_with(|| {
            // SAFETY: `respondsToSelector:` is an `NSObject` method every drawable has.
            let responds: objc::runtime::BOOL =
                unsafe { msg_send![drawable, respondsToSelector: sel!(addPresentedHandler:)] };
            responds == YES
        });
        if reports_presentation {
            observe_presentation(
                drawable,
                self.presentations.clone(),
                self.presented_frame_sink.clone(),
            );
        }
        self.pacer.submitted(Instant::now());
        if self.presents_with_transaction {
            command_buffer.commit();
            command_buffer.wait_until_scheduled();
            drawable.present();
        } else {
            command_buffer.present_drawable(drawable);
            command_buffer.commit();
        }
    }

    fn render_frame(
        &mut self,
        scene: &Scene,
        texture: &metal::TextureRef,
        viewport_size: Size<DevicePixels>,
    ) -> Result<metal::CommandBuffer> {
        let mut writer = InstanceBufferWriter::new(
            &self.device,
            &self.instance_buffer_pool,
            self.is_unified_memory,
        );
        let instance_bindings = write_instances(scene, &mut writer).with_context(|| {
            format!(
                "scene too large: {} paths, {} shadows, {} quads, {} underlines, {} mono, {} poly, {} surfaces",
                scene.paths.len(),
                scene.shadows.len(),
                scene.quads.len(),
                scene.underlines.len(),
                scene.monochrome_sprites.len(),
                scene.polychrome_sprites.len(),
                scene.surfaces.len(),
            )
        })?;
        // Housekeeping CoreVideo asks for periodically: textures no command buffer holds any
        // more give their buffers back.
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        self.core_video_texture_cache.flush(0);
        let command_buffer = self.draw_primitives_to_texture(
            scene,
            &instance_bindings,
            &mut writer,
            texture,
            viewport_size,
        );
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        let surface_textures = Cell::new(mem::take(&mut self.surface_textures));
        let command_buffer = command_buffer?;

        let instance_buffer_pool = self.instance_buffer_pool.clone();
        let instance_buffer = Cell::new(Some(writer.finish()));
        let block = RcBlock::new(move |_: ptr::NonNull<AnyObject>| {
            if let Some(instance_buffer) = instance_buffer.take() {
                instance_buffer_pool.lock().release(instance_buffer);
            }
            // The GPU is done sampling this frame's surfaces.
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            drop(surface_textures.take());
        });
        // SAFETY: Both pointee types are opaque views of the same Objective-C block pointer ABI.
        unsafe {
            command_buffer.add_completed_handler(&*RcBlock::as_ptr(&block).cast());
        }

        Ok(command_buffer)
    }

    /// Renders the scene to a texture and returns the pixel data as an RGBA image.
    /// This does not present the frame to screen - useful for visual testing
    /// where we want to capture what would be rendered without displaying it.
    ///
    /// Note: This requires a layer-backed renderer. For headless rendering,
    /// use `render_scene_to_image()` instead.
    #[cfg(any(test, feature = "test-support"))]
    pub fn render_to_image(&mut self, scene: &Scene) -> Result<RgbaImage> {
        let layer = self
            .layer
            .clone()
            .ok_or_else(|| anyhow::anyhow!("render_to_image requires a layer-backed renderer"))?;
        let viewport_size = layer.drawable_size();
        let viewport_size: Size<DevicePixels> = size(
            (viewport_size.width.ceil() as i32).into(),
            (viewport_size.height.ceil() as i32).into(),
        );
        let drawable = layer
            .next_drawable()
            .ok_or_else(|| anyhow::anyhow!("Failed to get drawable for render_to_image"))?;

        let command_buffer = self.render_frame(scene, drawable.texture(), viewport_size)?;

        // Commit and wait for completion without presenting
        command_buffer.commit();
        command_buffer.wait_until_completed();

        read_texture_to_image(drawable.texture())
    }

    /// Renders a scene to an image without requiring a window or CAMetalLayer.
    ///
    /// This is the primary method for headless rendering. It creates an offscreen
    /// texture, renders the scene to it, and returns the pixel data as an RGBA image.
    #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
    pub fn render_scene_to_image(
        &mut self,
        scene: &Scene,
        size: Size<DevicePixels>,
    ) -> Result<RgbaImage> {
        if size.width.0 <= 0 || size.height.0 <= 0 {
            anyhow::bail!("Invalid size for render_scene_to_image: {:?}", size);
        }

        // Headless callers do not have a Cocoa event-loop pool to release
        // autoreleased command buffers and render-pass descriptors.
        objc2::rc::autoreleasepool(|_| {
            // Update path intermediate textures for this size
            self.update_path_intermediate_textures(size);

            // Create an offscreen texture as render target
            let texture_descriptor = metal::TextureDescriptor::new();
            texture_descriptor.set_width(size.width.0 as u64);
            texture_descriptor.set_height(size.height.0 as u64);
            texture_descriptor.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
            texture_descriptor.set_usage(
                metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead,
            );
            texture_descriptor.set_storage_mode(if self.is_unified_memory {
                metal::MTLStorageMode::Shared
            } else {
                metal::MTLStorageMode::Managed
            });
            let target_texture = self.device.new_texture(&texture_descriptor);

            let command_buffer = self.render_frame(scene, &target_texture, size)?;

            // On discrete GPUs (non-unified memory), Managed textures require an
            // explicit blit synchronize before the CPU can read back the rendered
            // data. Without this, get_bytes returns stale zeros.
            if !self.is_unified_memory {
                let blit = command_buffer.new_blit_command_encoder();
                blit.synchronize_resource(&target_texture);
                blit.end_encoding();
            }

            // Commit and wait for completion
            command_buffer.commit();
            command_buffer.wait_until_completed();

            read_texture_to_image(&target_texture)
        })
    }

    /// Renders a scene to a reused offscreen texture without reading pixels
    /// back or blocking on GPU completion.
    ///
    /// This mirrors the CPU cost of presenting a frame to a window (scene
    /// encoding, instance buffer writes, command submission) and is used by
    /// headless benchmark rendering, where the produced pixels are never
    /// inspected.
    #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
    pub fn render_scene(&mut self, scene: &Scene, size: Size<DevicePixels>) -> Result<()> {
        if size.width.0 <= 0 || size.height.0 <= 0 {
            anyhow::bail!("Invalid size for render_scene: {:?}", size);
        }

        objc2::rc::autoreleasepool(|_| {
            self.update_path_intermediate_textures(size);

            let needs_new_target = self.headless_render_target.as_ref().is_none_or(|texture| {
                texture.width() != size.width.0 as u64 || texture.height() != size.height.0 as u64
            });
            if needs_new_target {
                let texture_descriptor = metal::TextureDescriptor::new();
                texture_descriptor.set_width(size.width.0 as u64);
                texture_descriptor.set_height(size.height.0 as u64);
                texture_descriptor.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
                texture_descriptor.set_usage(
                    metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead,
                );
                texture_descriptor.set_storage_mode(metal::MTLStorageMode::Private);
                self.headless_render_target = Some(self.device.new_texture(&texture_descriptor));
            }
            let target_texture = self
                .headless_render_target
                .clone()
                .expect("just ensured the render target exists");

            let command_buffer = self.render_frame(scene, &target_texture, size)?;

            // Commit without waiting, mirroring presentation to a real window where
            // the CPU doesn't block on the GPU.
            command_buffer.commit();
            Ok(())
        })
    }

    fn draw_primitives_to_texture(
        &mut self,
        scene: &Scene,
        instance_bindings: &InstanceBindings,
        writer: &mut InstanceBufferWriter,
        texture: &metal::TextureRef,
        viewport_size: Size<DevicePixels>,
    ) -> Result<metal::CommandBuffer> {
        let command_queue = self.command_queue.clone();
        let command_buffer = command_queue.new_command_buffer();
        let alpha = if self.opaque { 1. } else { 0. };

        let mut command_encoder = new_command_encoder_for_texture(
            command_buffer,
            texture,
            viewport_size,
            Some(metal::MTLClearColor::new(0., 0., 0., alpha)),
        );

        for batch in scene.batches() {
            match batch {
                PrimitiveBatch::Shadows(range) => {
                    self.draw_shadows(range, instance_bindings, viewport_size, command_encoder)
                }
                PrimitiveBatch::Quads(range) => {
                    self.draw_quads(range, instance_bindings, viewport_size, command_encoder)
                }
                PrimitiveBatch::Paths(range) => {
                    let paths = &scene.paths[range];
                    command_encoder.end_encoding();

                    let did_draw = self.draw_paths_to_intermediate(
                        paths,
                        writer,
                        viewport_size,
                        command_buffer,
                    )?;

                    command_encoder = new_command_encoder_for_texture(
                        command_buffer,
                        texture,
                        viewport_size,
                        None,
                    );

                    if did_draw {
                        if let Err(error) = self.draw_paths_from_intermediate(
                            paths,
                            writer,
                            viewport_size,
                            command_encoder,
                        ) {
                            command_encoder.end_encoding();
                            return Err(error);
                        }
                    }
                }
                PrimitiveBatch::Underlines(range) => {
                    self.draw_underlines(range, instance_bindings, viewport_size, command_encoder)
                }
                PrimitiveBatch::MonochromeSprites { texture_id, range } => self
                    .draw_monochrome_sprites(
                        texture_id,
                        range,
                        instance_bindings,
                        viewport_size,
                        command_encoder,
                    ),
                PrimitiveBatch::PolychromeSprites { texture_id, range } => self
                    .draw_polychrome_sprites(
                        texture_id,
                        range,
                        instance_bindings,
                        viewport_size,
                        command_encoder,
                    ),
                PrimitiveBatch::Surfaces(range) => self.draw_surfaces(
                    &scene.surfaces[range.clone()],
                    range.start,
                    instance_bindings,
                    viewport_size,
                    command_encoder,
                ),
                PrimitiveBatch::SubpixelSprites { .. } => unreachable!(),
            }
        }

        command_encoder.end_encoding();

        Ok(command_buffer.to_owned())
    }

    fn draw_paths_to_intermediate(
        &self,
        paths: &[Path<ScaledPixels>],
        writer: &mut InstanceBufferWriter,
        viewport_size: Size<DevicePixels>,
        command_buffer: &metal::CommandBufferRef,
    ) -> Result<bool> {
        if paths.is_empty() {
            return Ok(false);
        }
        let intermediate_texture = self
            .path_intermediate_texture
            .as_ref()
            .context("missing path intermediate texture")?;

        let mut vertices = Vec::new();
        for path in paths {
            vertices.extend(path.vertices.iter().map(|v| PathRasterizationVertex {
                xy_position: v.xy_position,
                st_position: v.st_position,
                color: path.color,
                bounds: path.bounds.intersect(&path.content_mask.bounds),
            }));
        }
        let vertex_instance_bindings = writer.write(&vertices)?;

        let render_pass_descriptor = metal::RenderPassDescriptor::new();
        let color_attachment = render_pass_descriptor
            .color_attachments()
            .object_at(0)
            .unwrap();
        color_attachment.set_load_action(metal::MTLLoadAction::Clear);
        color_attachment.set_clear_color(metal::MTLClearColor::new(0., 0., 0., 0.));

        if let Some(msaa_texture) = &self.path_intermediate_msaa_texture {
            color_attachment.set_texture(Some(msaa_texture));
            color_attachment.set_resolve_texture(Some(intermediate_texture));
            color_attachment.set_store_action(metal::MTLStoreAction::MultisampleResolve);
        } else {
            color_attachment.set_texture(Some(intermediate_texture));
            color_attachment.set_store_action(metal::MTLStoreAction::Store);
        }

        let command_encoder = command_buffer.new_render_command_encoder(render_pass_descriptor);
        command_encoder.set_render_pipeline_state(&self.paths_rasterization_pipeline_state);
        command_encoder.set_vertex_buffer(
            PathRasterizationInputIndex::Vertices as u64,
            Some(&vertex_instance_bindings.buffer),
            vertex_instance_bindings.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            PathRasterizationInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_fragment_buffer(
            PathRasterizationInputIndex::Vertices as u64,
            Some(&vertex_instance_bindings.buffer),
            vertex_instance_bindings.offset as u64,
        );
        command_encoder.draw_primitives(
            metal::MTLPrimitiveType::Triangle,
            0,
            vertices.len() as u64,
        );

        command_encoder.end_encoding();
        Ok(true)
    }

    fn draw_shadows(
        &self,
        shadows: Range<usize>,
        instance_bindings: &InstanceBindings,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        if shadows.is_empty() {
            return;
        }

        command_encoder.set_render_pipeline_state(&self.shadows_pipeline_state);
        command_encoder.set_vertex_buffer(
            ShadowInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            ShadowInputIndex::Shadows as u64,
            Some(&instance_bindings.shadows.buffer),
            instance_bindings.shadows.offset as u64,
        );
        command_encoder.set_fragment_buffer(
            ShadowInputIndex::Shadows as u64,
            Some(&instance_bindings.shadows.buffer),
            instance_bindings.shadows.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            ShadowInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        command_encoder.draw_primitives_instanced_base_instance(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            shadows.len() as u64,
            shadows.start as u64,
        );
    }

    fn draw_quads(
        &self,
        quads: Range<usize>,
        instance_bindings: &InstanceBindings,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        if quads.is_empty() {
            return;
        }

        command_encoder.set_render_pipeline_state(&self.quads_pipeline_state);
        command_encoder.set_vertex_buffer(
            QuadInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            QuadInputIndex::Quads as u64,
            Some(&instance_bindings.quads.buffer),
            instance_bindings.quads.offset as u64,
        );
        command_encoder.set_fragment_buffer(
            QuadInputIndex::Quads as u64,
            Some(&instance_bindings.quads.buffer),
            instance_bindings.quads.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            QuadInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        command_encoder.draw_primitives_instanced_base_instance(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            quads.len() as u64,
            quads.start as u64,
        );
    }

    fn draw_paths_from_intermediate(
        &self,
        paths: &[Path<ScaledPixels>],
        writer: &mut InstanceBufferWriter,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> Result<()> {
        let Some(first_path) = paths.first() else {
            return Ok(());
        };
        let intermediate_texture = self
            .path_intermediate_texture
            .as_ref()
            .context("missing path intermediate texture")?;

        command_encoder.set_render_pipeline_state(&self.path_sprites_pipeline_state);
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        command_encoder.set_fragment_texture(
            SpriteInputIndex::AtlasTexture as u64,
            Some(intermediate_texture),
        );

        // When copying paths from the intermediate texture to the drawable,
        // each pixel must only be copied once, in case of transparent paths.
        //
        // If all paths have the same draw order, then their bounds are all
        // disjoint, so we can copy each path's bounds individually. If this
        // batch combines different draw orders, we perform a single copy
        // for a minimal spanning rect.
        let sprites;
        if paths.last().unwrap().order == first_path.order {
            sprites = paths
                .iter()
                .map(|path| PathSprite {
                    bounds: path.clipped_bounds(),
                })
                .collect();
        } else {
            let mut bounds = first_path.clipped_bounds();
            for path in paths.iter().skip(1) {
                bounds = bounds.union(&path.clipped_bounds());
            }
            sprites = vec![PathSprite { bounds }];
        }

        let sprite_instance_bindings = writer.write(&sprites)?;
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&sprite_instance_bindings.buffer),
            sprite_instance_bindings.offset as u64,
        );

        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            sprites.len() as u64,
        );
        Ok(())
    }

    fn draw_underlines(
        &self,
        underlines: Range<usize>,
        instance_bindings: &InstanceBindings,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        if underlines.is_empty() {
            return;
        }

        command_encoder.set_render_pipeline_state(&self.underlines_pipeline_state);
        command_encoder.set_vertex_buffer(
            UnderlineInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            UnderlineInputIndex::Underlines as u64,
            Some(&instance_bindings.underlines.buffer),
            instance_bindings.underlines.offset as u64,
        );
        command_encoder.set_fragment_buffer(
            UnderlineInputIndex::Underlines as u64,
            Some(&instance_bindings.underlines.buffer),
            instance_bindings.underlines.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            UnderlineInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        command_encoder.draw_primitives_instanced_base_instance(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            underlines.len() as u64,
            underlines.start as u64,
        );
    }

    fn draw_monochrome_sprites(
        &self,
        texture_id: AtlasTextureId,
        sprites: Range<usize>,
        instance_bindings: &InstanceBindings,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        if sprites.is_empty() {
            return;
        }

        let Some(texture) = self.sprite_atlas.metal_texture(texture_id) else {
            return;
        };
        let texture_size = size(
            DevicePixels(texture.width() as i32),
            DevicePixels(texture.height() as i32),
        );
        command_encoder.set_render_pipeline_state(&self.monochrome_sprites_pipeline_state);
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instance_bindings.monochrome_sprites.buffer),
            instance_bindings.monochrome_sprites.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::AtlasTextureSize as u64,
            mem::size_of_val(&texture_size) as u64,
            &texture_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_fragment_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instance_bindings.monochrome_sprites.buffer),
            instance_bindings.monochrome_sprites.offset as u64,
        );
        command_encoder.set_fragment_texture(SpriteInputIndex::AtlasTexture as u64, Some(&texture));

        command_encoder.draw_primitives_instanced_base_instance(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            sprites.len() as u64,
            sprites.start as u64,
        );
    }

    fn draw_polychrome_sprites(
        &self,
        texture_id: AtlasTextureId,
        sprites: Range<usize>,
        instance_bindings: &InstanceBindings,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        if sprites.is_empty() {
            return;
        }

        let Some(texture) = self.sprite_atlas.metal_texture(texture_id) else {
            return;
        };
        let texture_size = size(
            DevicePixels(texture.width() as i32),
            DevicePixels(texture.height() as i32),
        );
        command_encoder.set_render_pipeline_state(&self.polychrome_sprites_pipeline_state);
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instance_bindings.polychrome_sprites.buffer),
            instance_bindings.polychrome_sprites.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::AtlasTextureSize as u64,
            mem::size_of_val(&texture_size) as u64,
            &texture_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_fragment_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instance_bindings.polychrome_sprites.buffer),
            instance_bindings.polychrome_sprites.offset as u64,
        );
        command_encoder.set_fragment_texture(SpriteInputIndex::AtlasTexture as u64, Some(&texture));

        command_encoder.draw_primitives_instanced_base_instance(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            sprites.len() as u64,
            sprites.start as u64,
        );
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn draw_surfaces(
        &mut self,
        surfaces: &[PaintSurface],
        first_surface: usize,
        instance_bindings: &InstanceBindings,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        if surfaces.is_empty() {
            return;
        }

        command_encoder.set_render_pipeline_state(&self.surfaces_pipeline_state);
        command_encoder.set_vertex_buffer(
            SurfaceInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            SurfaceInputIndex::Surfaces as u64,
            Some(&instance_bindings.surfaces.buffer),
            instance_bindings.surfaces.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            SurfaceInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        for (index, surface) in surfaces.iter().enumerate() {
            let texture_size = size(
                DevicePixels::from(surface.image_buffer.get_width() as i32),
                DevicePixels::from(surface.image_buffer.get_height() as i32),
            );

            let format = surface.image_buffer.get_pixel_format();
            let Some(layout) = SurfaceLayout::of(format) else {
                self.skip_surface(format_args!(
                    "pixel format {format:#x} is not an 8- or 10-bit Y′CbCr bi-planar one"
                ));
                continue;
            };
            let ycbcr_to_rgb = ycbcr_to_rgb(surface_matrix(&surface.image_buffer), layout);
            let (luma_format, chroma_format) = layout.depth.plane_formats();

            let plane = |plane: usize, format: MTLPixelFormat| {
                self.core_video_texture_cache.create_texture_from_image(
                    surface.image_buffer.as_concrete_TypeRef(),
                    None,
                    format,
                    surface.image_buffer.get_width_of_plane(plane),
                    surface.image_buffer.get_height_of_plane(plane),
                    plane,
                )
            };
            let (y_texture, cb_cr_texture) = match (plane(0, luma_format), plane(1, chroma_format))
            {
                (Ok(y), Ok(cb_cr)) => (y, cb_cr),
                (Err(status), _) | (_, Err(status)) => {
                    self.skip_surface(format_args!(
                        "CVMetalTextureCacheCreateTextureFromImage failed: {status}"
                    ));
                    continue;
                }
            };

            command_encoder.set_vertex_bytes(
                SurfaceInputIndex::TextureSize as u64,
                mem::size_of_val(&texture_size) as u64,
                &texture_size as *const Size<DevicePixels> as *const _,
            );
            command_encoder.set_fragment_texture(SurfaceInputIndex::YTexture as u64, unsafe {
                let texture = CVMetalTextureGetTexture(y_texture.as_concrete_TypeRef());
                Some(metal::TextureRef::from_ptr(texture as *mut _))
            });
            command_encoder.set_fragment_texture(SurfaceInputIndex::CbCrTexture as u64, unsafe {
                let texture = CVMetalTextureGetTexture(cb_cr_texture.as_concrete_TypeRef());
                Some(metal::TextureRef::from_ptr(texture as *mut _))
            });
            command_encoder.set_fragment_bytes(
                SurfaceInputIndex::YCbCrToRgb as u64,
                mem::size_of_val(&ycbcr_to_rgb) as u64,
                ycbcr_to_rgb.as_ptr() as *const _,
            );
            self.surface_textures.push(y_texture);
            self.surface_textures.push(cb_cr_texture);

            command_encoder.draw_primitives_instanced_base_instance(
                metal::MTLPrimitiveType::Triangle,
                0,
                6,
                1,
                (first_surface + index) as u64,
            );
        }
    }

    /// Leave a surface out of this frame rather than abort the app over it, and say why once.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn skip_surface(&mut self, why: std::fmt::Arguments<'_>) {
        if !self.surface_skip_logged {
            self.surface_skip_logged = true;
            log::error!("skipping a video surface: {why}");
        }
    }
}

/// The Y′CbCr matrix a surface's buffer is tagged with. Untagged is BT.709, the HD default.
#[cfg(any(target_os = "macos", target_os = "ios"))]
fn surface_matrix(buffer: &CVPixelBuffer) -> YCbCrMatrix {
    // SAFETY: the keys and values are CoreVideo's own constant strings, which live forever.
    let tagged = |key| unsafe { CFString::wrap_under_get_rule(key) };
    let Some(matrix) = buffer
        .as_buffer()
        .get_attachment(&tagged(unsafe { kCVImageBufferYCbCrMatrixKey }), None)
        .and_then(|value| value.downcast::<CFString>())
    else {
        return YCbCrMatrix::Bt709;
    };
    if matrix == tagged(unsafe { kCVImageBufferYCbCrMatrix_ITU_R_601_4 }) {
        YCbCrMatrix::Bt601
    } else if matrix == tagged(unsafe { kCVImageBufferYCbCrMatrix_ITU_R_2020 }) {
        YCbCrMatrix::Bt2020
    } else {
        YCbCrMatrix::Bt709
    }
}

/// The Y′CbCr → R′G′B′ matrices a surface may be tagged with.
#[derive(Clone, Copy, Debug, PartialEq)]
enum YCbCrMatrix {
    Bt601,
    Bt709,
    Bt2020,
}

/// How a surface's two planes hold their samples, read off its pixel format. Chroma may be
/// subsampled or not: the shader samples both planes at the same normalised coordinates, so
/// 4:2:0, 4:2:2 and 4:4:4 differ only in the plane sizes CoreVideo reports.
#[derive(Clone, Copy, Debug, PartialEq)]
struct SurfaceLayout {
    depth: SampleDepth,
    full_range: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum SampleDepth {
    Eight,
    /// Ten significant bits at the top of each 16-bit sample.
    Ten,
}

impl SurfaceLayout {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    #[allow(non_upper_case_globals)]
    fn of(format: u32) -> Option<Self> {
        let (depth, full_range) = match format {
            kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange => (SampleDepth::Eight, false),
            kCVPixelFormatType_420YpCbCr8BiPlanarFullRange => (SampleDepth::Eight, true),
            kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange
            | kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange
            | kCVPixelFormatType_444YpCbCr10BiPlanarVideoRange => (SampleDepth::Ten, false),
            kCVPixelFormatType_420YpCbCr10BiPlanarFullRange
            | kCVPixelFormatType_422YpCbCr10BiPlanarFullRange
            | kCVPixelFormatType_444YpCbCr10BiPlanarFullRange => (SampleDepth::Ten, true),
            _ => return None,
        };
        Some(Self { depth, full_range })
    }
}

impl SampleDepth {
    /// The luma and interleaved-chroma texture formats a plane of this depth is viewed as.
    fn plane_formats(self) -> (MTLPixelFormat, MTLPixelFormat) {
        match self {
            Self::Eight => (MTLPixelFormat::R8Unorm, MTLPixelFormat::RG8Unorm),
            Self::Ten => (MTLPixelFormat::R16Unorm, MTLPixelFormat::RG16Unorm),
        }
    }

    fn bits(self) -> i32 {
        match self {
            Self::Eight => 8,
            Self::Ten => 10,
        }
    }

    /// What a unorm texture reads for `code` at this depth.
    fn sampled(self, code: f32) -> f32 {
        match self {
            Self::Eight => code / 255.0,
            Self::Ten => code * 64.0 / 65535.0,
        }
    }
}

/// The `float4x4` (columns, as Metal lays it out) the surface shader multiplies
/// `(y, cb, cr, 1)` by, for `matrix` at the depth and range of `layout`.
fn ycbcr_to_rgb(matrix: YCbCrMatrix, layout: SurfaceLayout) -> [[f32; 4]; 4] {
    let (kr, kb) = match matrix {
        YCbCrMatrix::Bt601 => (0.299, 0.114),
        YCbCrMatrix::Bt709 => (0.2126, 0.0722),
        YCbCrMatrix::Bt2020 => (0.2627, 0.0593),
    };
    let kg = 1.0 - kr - kb;
    let depth = layout.depth;
    // Full range spans every code; video range puts black at 16 and white at 235, chroma
    // between 16 and 240, each scaled by 4 at 10 bits (64, 940, 64 and 960). Chroma is centred
    // on 128 (512) either way.
    let step = 2f32.powi(depth.bits() - 8);
    let max = 2f32.powi(depth.bits()) - 1.0;
    let (y_black, y_span, c_span) = if layout.full_range {
        (0.0, max, max)
    } else {
        (16.0 * step, 219.0 * step, 224.0 * step)
    };
    let y_offset = depth.sampled(y_black);
    let y_scale = 1.0 / depth.sampled(y_span);
    let c_scale = 1.0 / depth.sampled(c_span);
    let c_offset = depth.sampled(128.0 * step);
    let r_cr = 2.0 * (1.0 - kr) * c_scale;
    let g_cb = 2.0 * kb * (1.0 - kb) / kg * c_scale;
    let g_cr = 2.0 * kr * (1.0 - kr) / kg * c_scale;
    let b_cb = 2.0 * (1.0 - kb) * c_scale;
    let y0 = -y_offset * y_scale;
    [
        [y_scale, y_scale, y_scale, 0.0],
        [0.0, -g_cb, b_cb, 0.0],
        [r_cr, -g_cr, 0.0, 0.0],
        [
            y0 - r_cr * c_offset,
            y0 + (g_cb + g_cr) * c_offset,
            y0 - b_cb * c_offset,
            1.0,
        ],
    ]
}

#[cfg(test)]
mod ycbcr_tests {
    use super::{MetalRenderer, SampleDepth, SurfaceLayout, YCbCrMatrix, ycbcr_to_rgb};
    use core_foundation::{base::TCFType, dictionary::CFDictionary, string::CFString};
    use core_video::{
        buffer::{TCVBuffer, kCVAttachmentMode_ShouldPropagate},
        image_buffer::{kCVImageBufferYCbCrMatrix_ITU_R_709_2, kCVImageBufferYCbCrMatrixKey},
        pixel_buffer::{
            CVPixelBuffer, CVPixelBufferGetBaseAddressOfPlane, CVPixelBufferGetBytesPerRowOfPlane,
            kCVPixelBufferIOSurfacePropertiesKey, kCVPixelFormatType_32BGRA,
            kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
            kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
            kCVPixelFormatType_444YpCbCr10BiPlanarFullRange,
            kCVPixelFormatType_444YpCbCr10BiPlanarVideoRange,
        },
    };
    use gpui::{Bounds, ContentMask, DevicePixels, PaintSurface, Scene, point, px, size};
    use image::RgbaImage;

    const SIDE: usize = 32;
    const EIGHT_FULL: SurfaceLayout = SurfaceLayout {
        depth: SampleDepth::Eight,
        full_range: true,
    };
    const EIGHT_VIDEO: SurfaceLayout = SurfaceLayout {
        depth: SampleDepth::Eight,
        full_range: false,
    };
    const TEN_FULL: SurfaceLayout = SurfaceLayout {
        depth: SampleDepth::Ten,
        full_range: true,
    };
    const TEN_VIDEO: SurfaceLayout = SurfaceLayout {
        depth: SampleDepth::Ten,
        full_range: false,
    };

    fn rgb(m: [[f32; 4]; 4], y: f32, cb: f32, cr: f32) -> [f32; 3] {
        let v = [y, cb, cr, 1.0];
        let mut out = [0.0; 3];
        for (row, value) in out.iter_mut().enumerate() {
            *value = (0..4).map(|col| m[col][row] * v[col]).sum();
        }
        out
    }

    fn close(a: [f32; 3], b: [f32; 3]) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < 2e-3)
    }

    /// What an R16Unorm texel reads for a 10-bit code held in the top bits.
    fn ten(code: u16) -> f32 {
        f32::from(code << 6) / 65535.0
    }

    /// An IOSurface-backed buffer, as a decoder hands over, tagged BT.709.
    fn surface(format: u32) -> CVPixelBuffer {
        // SAFETY: the key and values are CoreVideo's own constant strings, which live forever.
        let tag = |key| unsafe { CFString::wrap_under_get_rule(key) };
        let empty = CFDictionary::<CFString, CFString>::from_CFType_pairs(&[]);
        let options = CFDictionary::from_CFType_pairs(&[(
            tag(unsafe { kCVPixelBufferIOSurfacePropertiesKey }),
            empty.as_CFType(),
        )]);
        let buffer = CVPixelBuffer::new(format, SIDE, SIDE, Some(&options)).unwrap();
        buffer.as_buffer().set_attachment(
            &tag(unsafe { kCVImageBufferYCbCrMatrixKey }),
            &tag(unsafe { kCVImageBufferYCbCrMatrix_ITU_R_709_2 }).as_CFType(),
            kCVAttachmentMode_ShouldPropagate,
        );
        buffer
    }

    /// Writes `texel(x)` to every texel of each row of `plane`, as samples of type `T`.
    fn fill<T: Copy>(buffer: &CVPixelBuffer, plane: usize, texel: impl Fn(usize) -> Vec<T>) {
        // SAFETY: the base address is locked, and a plane spans `bytes per row × height`.
        let (base, stride) = unsafe {
            (
                CVPixelBufferGetBaseAddressOfPlane(buffer.as_concrete_TypeRef(), plane),
                CVPixelBufferGetBytesPerRowOfPlane(buffer.as_concrete_TypeRef(), plane),
            )
        };
        let row_len = stride / size_of::<T>();
        let samples = unsafe {
            std::slice::from_raw_parts_mut(
                base.cast::<T>(),
                row_len * buffer.get_height_of_plane(plane),
            )
        };
        for row in samples.chunks_exact_mut(row_len) {
            let mut at = 0;
            for x in 0..buffer.get_width_of_plane(plane) {
                let value = texel(x);
                row[at..at + value.len()].copy_from_slice(&value);
                at += value.len();
            }
        }
    }

    fn render(renderer: &mut MetalRenderer, image_buffer: CVPixelBuffer) -> RgbaImage {
        let bounds = Bounds::new(
            point(px(0.), px(0.)),
            size(px(SIDE as f32), px(SIDE as f32)),
        )
        .scale(1.0);
        let mut scene = Scene::default();
        scene.insert_primitive(PaintSurface {
            order: 0,
            bounds,
            content_mask: ContentMask { bounds },
            image_buffer,
        });
        scene.finish();
        let target = size(DevicePixels(SIDE as i32), DevicePixels(SIDE as i32));
        renderer.render_scene_to_image(&scene, target).unwrap()
    }

    /// Black, white and the primaries come out where BT.709 puts them, at both ranges; the
    /// BT.601 matrix is the one the shader hard-coded before.
    #[test]
    fn the_matrices_map_codes_to_the_right_colours() {
        let c = 128.0 / 255.0;
        let full = ycbcr_to_rgb(YCbCrMatrix::Bt709, EIGHT_FULL);
        assert!(close(rgb(full, 0.0, c, c), [0.0; 3]));
        assert!(close(rgb(full, 1.0, c, c), [1.0; 3]));
        // Pure red in BT.709: Y = Kr, Cb = −Kr / (2 (1 − Kb)), Cr = 1/2.
        let red = rgb(full, 0.2126, c - 0.2126 / (2.0 * (1.0 - 0.0722)), c + 0.5);
        assert!(close(red, [1.0, 0.0, 0.0]), "{red:?}");

        let video = ycbcr_to_rgb(YCbCrMatrix::Bt709, EIGHT_VIDEO);
        assert!(close(rgb(video, 16.0 / 255.0, c, c), [0.0; 3]));
        assert!(close(rgb(video, 235.0 / 255.0, c, c), [1.0; 3]));

        let bt601 = ycbcr_to_rgb(YCbCrMatrix::Bt601, EIGHT_FULL);
        let old = [
            [1.0, 1.0, 1.0, 0.0],
            [0.0, -0.3441, 1.7720, 0.0],
            [1.4020, -0.7141, 0.0, 0.0],
        ];
        for (column, want) in bt601.iter().zip(old) {
            assert!(
                column.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-3),
                "{bt601:?}"
            );
        }
        assert_ne!(ycbcr_to_rgb(YCbCrMatrix::Bt2020, EIGHT_FULL), full);
    }

    /// 10-bit video range puts black at code 64, white at 940 and chroma's extremes at 64 and
    /// 960; full range spans 0 to 1023. The codes sit in the top of 16-bit samples, so they
    /// read differently from the 8-bit ones and need their own offsets.
    #[test]
    fn ten_bit_codes_map_to_the_right_colours() {
        let video = ycbcr_to_rgb(YCbCrMatrix::Bt709, TEN_VIDEO);
        let c = ten(512);
        assert!(close(rgb(video, ten(64), c, c), [0.0; 3]));
        assert!(close(rgb(video, ten(940), c, c), [1.0; 3]));
        let red = rgb(video, ten(250), ten(409), ten(960));
        assert!(close(red, [1.0, 0.0, 0.0]), "{red:?}");

        let full = ycbcr_to_rgb(YCbCrMatrix::Bt709, TEN_FULL);
        assert!(close(rgb(full, 0.0, c, c), [0.0; 3]));
        assert!(close(rgb(full, ten(1023), c, c), [1.0; 3]));
        assert_ne!(video, ycbcr_to_rgb(YCbCrMatrix::Bt709, EIGHT_VIDEO));
    }

    #[test]
    fn a_pixel_format_picks_its_depth_and_range() {
        for (format, layout) in [
            (kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange, EIGHT_VIDEO),
            (kCVPixelFormatType_420YpCbCr8BiPlanarFullRange, EIGHT_FULL),
            (kCVPixelFormatType_444YpCbCr10BiPlanarVideoRange, TEN_VIDEO),
            (kCVPixelFormatType_444YpCbCr10BiPlanarFullRange, TEN_FULL),
        ] {
            assert_eq!(SurfaceLayout::of(format), Some(layout), "{format:#x}");
        }
        assert_eq!(SurfaceLayout::of(kCVPixelFormatType_32BGRA), None);
    }

    /// A full-range BT.709 red surface renders red through the real pipeline, frame after
    /// frame, with its textures held to each command buffer's completion; a surface in a
    /// format the shader cannot sample is left out instead of aborting the renderer.
    #[test]
    fn a_tagged_surface_renders_its_colour_and_a_foreign_one_is_skipped() {
        let red = surface(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange);
        assert_eq!(red.lock_base_address(0), 0);
        // BT.709 full-range red: Y = 0.2126, Cb = 0.5 − 0.2126 / 1.8556, Cr = 1.
        fill::<u8>(&red, 0, |_| vec![54]);
        fill::<u8>(&red, 1, |_| vec![99, 255]);
        assert_eq!(red.unlock_base_address(0), 0);

        let mut renderer = MetalRenderer::new_headless(Default::default());
        for _ in 0..3 {
            let [r, g, b, _] = render(&mut renderer, red.clone()).get_pixel(16, 16).0;
            assert!(
                r >= 250 && g <= 5 && b <= 5,
                "BT.709 red came out {r} {g} {b}"
            );
        }

        let image = render(&mut renderer, surface(kCVPixelFormatType_32BGRA));
        assert_eq!(
            image.get_pixel(16, 16).0[..3],
            [0, 0, 0],
            "skipped, the clear colour shows"
        );
        assert!(renderer.surface_skip_logged);
    }

    /// A 10-bit 4:4:4 surface, at video range and at full, renders columns that alternate red
    /// and blue each keeping its colour: chroma is sampled at full resolution, and the 16-bit
    /// planes are read with the 10-bit offsets.
    #[test]
    fn a_ten_bit_444_surface_keeps_every_column_its_colour() {
        let mut renderer = MetalRenderer::new_headless(Default::default());
        // BT.709 red and blue as 10-bit (Y, Cb, Cr) codes.
        for (format, red, blue) in [
            (
                kCVPixelFormatType_444YpCbCr10BiPlanarVideoRange,
                [250, 409, 960],
                [127, 960, 471],
            ),
            (
                kCVPixelFormatType_444YpCbCr10BiPlanarFullRange,
                [217, 395, 1023],
                [74, 1023, 465],
            ),
        ] {
            let buffer = surface(format);
            assert_eq!(
                buffer.get_width_of_plane(1),
                SIDE,
                "4:4:4 chroma is full width"
            );
            let code = |x: usize| if x.is_multiple_of(2) { red } else { blue };
            assert_eq!(buffer.lock_base_address(0), 0);
            fill::<u16>(&buffer, 0, |x| vec![code(x)[0] << 6]);
            fill::<u16>(&buffer, 1, |x| vec![code(x)[1] << 6, code(x)[2] << 6]);
            assert_eq!(buffer.unlock_base_address(0), 0);

            let image = render(&mut renderer, buffer);
            for x in 0..SIDE as u32 {
                let [r, g, b, _] = image.get_pixel(x, 16).0;
                let want_red = x.is_multiple_of(2);
                let (hi, lo) = if want_red { (r, b) } else { (b, r) };
                assert!(
                    hi >= 250 && g <= 5 && lo <= 5,
                    "{format:#x} column {x} came out {r} {g} {b}"
                );
            }
        }
        assert!(!renderer.surface_skip_logged);
    }
}

fn new_command_encoder_for_texture<'a>(
    command_buffer: &'a metal::CommandBufferRef,
    texture: &'a metal::TextureRef,
    viewport_size: Size<DevicePixels>,
    clear_color: Option<metal::MTLClearColor>,
) -> &'a metal::RenderCommandEncoderRef {
    let render_pass_descriptor = metal::RenderPassDescriptor::new();
    let color_attachment = render_pass_descriptor
        .color_attachments()
        .object_at(0)
        .unwrap();
    color_attachment.set_texture(Some(texture));
    color_attachment.set_store_action(metal::MTLStoreAction::Store);
    if let Some(clear_color) = clear_color {
        color_attachment.set_load_action(metal::MTLLoadAction::Clear);
        color_attachment.set_clear_color(clear_color);
    } else {
        color_attachment.set_load_action(metal::MTLLoadAction::Load);
    }

    let command_encoder = command_buffer.new_render_command_encoder(render_pass_descriptor);
    command_encoder.set_viewport(metal::MTLViewport {
        originX: 0.0,
        originY: 0.0,
        width: i32::from(viewport_size.width) as f64,
        height: i32::from(viewport_size.height) as f64,
        znear: 0.0,
        zfar: 1.0,
    });
    command_encoder
}

#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
fn read_texture_to_image(texture: &metal::TextureRef) -> Result<RgbaImage> {
    let width = texture.width() as u32;
    let height = texture.height() as u32;
    let bytes_per_row = width as usize * 4;
    let mut pixels = vec![0u8; height as usize * bytes_per_row];

    let region = metal::MTLRegion {
        origin: metal::MTLOrigin { x: 0, y: 0, z: 0 },
        size: metal::MTLSize {
            width: width as u64,
            height: height as u64,
            depth: 1,
        },
    };
    texture.get_bytes(
        pixels.as_mut_ptr() as *mut std::ffi::c_void,
        bytes_per_row as u64,
        region,
        0,
    );

    // Convert BGRA to RGBA (swap B and R channels)
    for chunk in pixels.chunks_exact_mut(4) {
        chunk.swap(0, 2);
    }

    RgbaImage::from_raw(width, height, pixels).context("failed to create RgbaImage from pixel data")
}

fn build_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: metal::MTLPixelFormat,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(vertex_fn_name, None)
        .expect("error locating vertex function");
    let fragment_fn = library
        .get_function(fragment_fn_name, None)
        .expect("error locating fragment function");

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(true);
    color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::SourceAlpha);
    color_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::One);

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

fn build_path_sprite_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: metal::MTLPixelFormat,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(vertex_fn_name, None)
        .expect("error locating vertex function");
    let fragment_fn = library
        .get_function(fragment_fn_name, None)
        .expect("error locating fragment function");

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(true);
    color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::One);

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

fn build_path_rasterization_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: metal::MTLPixelFormat,
    path_sample_count: u32,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(vertex_fn_name, None)
        .expect("error locating vertex function");
    let fragment_fn = library
        .get_function(fragment_fn_name, None)
        .expect("error locating fragment function");

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    if path_sample_count > 1 {
        descriptor.set_raster_sample_count(path_sample_count as _);
        descriptor.set_alpha_to_coverage_enabled(false);
    }
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(true);
    color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

#[derive(Clone)]
struct InstanceBinding {
    buffer: metal::Buffer,
    offset: usize,
}

struct InstanceBindings {
    quads: InstanceBinding,
    shadows: InstanceBinding,
    underlines: InstanceBinding,
    monochrome_sprites: InstanceBinding,
    polychrome_sprites: InstanceBinding,
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    surfaces: InstanceBinding,
}

fn write_instances(scene: &Scene, writer: &mut InstanceBufferWriter) -> Result<InstanceBindings> {
    Ok(InstanceBindings {
        quads: writer.write(&scene.quads)?,
        shadows: writer.write(&scene.shadows)?,
        underlines: writer.write(&scene.underlines)?,
        monochrome_sprites: writer.write(&scene.monochrome_sprites)?,
        polychrome_sprites: writer.write(&scene.polychrome_sprites)?,
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        surfaces: writer.write_iter(scene.surfaces.iter().map(|surface| SurfaceBounds {
            bounds: surface.bounds,
            content_mask: surface.content_mask,
        }))?,
    })
}

struct InstanceBufferWriter {
    device: metal::Device,
    pool: Arc<Mutex<InstanceBufferPool>>,
    unified_memory: bool,
    filled: Vec<(InstanceBuffer, usize)>,
    current: InstanceBuffer,
    offset: usize,
}

impl InstanceBufferWriter {
    fn new(
        device: &metal::Device,
        pool: &Arc<Mutex<InstanceBufferPool>>,
        unified_memory: bool,
    ) -> Self {
        let current = pool.lock().acquire(device, unified_memory);
        Self {
            device: device.clone(),
            pool: pool.clone(),
            unified_memory,
            filled: Vec::new(),
            current,
            offset: 0,
        }
    }

    fn allocate<T>(&mut self, count: usize) -> Result<(InstanceBinding, &mut [MaybeUninit<T>])> {
        let size = mem::size_of::<T>() * count;
        let mut offset = self.offset.next_multiple_of(INSTANCE_BUFFER_ALIGNMENT);
        if offset + size > self.current.size {
            self.grow(size)?;
            offset = 0;
        }
        self.offset = offset + size;

        let binding = InstanceBinding {
            buffer: self.current.metal_buffer.clone(),
            offset,
        };
        // Safety: the reservation lies within a buffer this frame owns
        // exclusively, and never overlaps one handed out earlier.
        let values = unsafe {
            let start = (self.current.metal_buffer.contents() as *mut u8).add(offset);
            slice::from_raw_parts_mut(start.cast::<MaybeUninit<T>>(), count)
        };
        Ok((binding, values))
    }

    fn write<T>(&mut self, values: &[T]) -> Result<InstanceBinding> {
        let (binding, destination) = self.allocate::<T>(values.len())?;
        unsafe {
            ptr::copy_nonoverlapping(
                values.as_ptr(),
                destination.as_mut_ptr().cast::<T>(),
                values.len(),
            );
        }
        Ok(binding)
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn write_iter<T>(
        &mut self,
        values: impl ExactSizeIterator<Item = T>,
    ) -> Result<InstanceBinding> {
        let (binding, destination) = self.allocate::<T>(values.len())?;
        for (slot, value) in destination.iter_mut().zip(values) {
            slot.write(value);
        }
        Ok(binding)
    }

    fn grow(&mut self, required: usize) -> Result<()> {
        let mut pool = self.pool.lock();
        let buffer_size = (pool.buffer_size * 2)
            .max(required.next_power_of_two())
            .min(MAX_INSTANCE_BUFFER_SIZE);
        anyhow::ensure!(
            buffer_size >= required,
            "instance buffer needs {required} bytes, above the maximum of {MAX_INSTANCE_BUFFER_SIZE}"
        );
        anyhow::ensure!(
            buffer_size > self.current.size,
            "frame instance data exceeds the {MAX_INSTANCE_BUFFER_SIZE}-byte maximum"
        );
        if buffer_size != pool.buffer_size {
            log::info!("increased instance buffer size to {buffer_size}");
            pool.reset(buffer_size);
        }
        let buffer = pool.acquire(&self.device, self.unified_memory);
        drop(pool);

        let filled = mem::replace(&mut self.current, buffer);
        self.filled.push((filled, self.offset));
        self.offset = 0;
        Ok(())
    }

    fn finish(self) -> InstanceBuffer {
        let Self {
            unified_memory,
            filled,
            current,
            offset,
            ..
        } = self;

        if !unified_memory {
            for (buffer, written) in &filled {
                if *written == 0 {
                    continue;
                }
                buffer.metal_buffer.did_modify_range(NSRange {
                    location: 0,
                    length: *written as NSUInteger,
                });
            }
            if offset > 0 {
                current.metal_buffer.did_modify_range(NSRange {
                    location: 0,
                    length: offset as NSUInteger,
                });
            }
        }

        // Metal retains encoded resources until the command buffer completes.
        // Only the final, largest buffer is worth keeping in the pool.
        drop(filled);
        current
    }
}

#[repr(C)]
enum ShadowInputIndex {
    Vertices = 0,
    Shadows = 1,
    ViewportSize = 2,
}

#[repr(C)]
enum QuadInputIndex {
    Vertices = 0,
    Quads = 1,
    ViewportSize = 2,
}

#[repr(C)]
enum UnderlineInputIndex {
    Vertices = 0,
    Underlines = 1,
    ViewportSize = 2,
}

#[repr(C)]
enum SpriteInputIndex {
    Vertices = 0,
    Sprites = 1,
    ViewportSize = 2,
    AtlasTextureSize = 3,
    AtlasTexture = 4,
}

#[repr(C)]
#[doc(hidden)]
pub enum SurfaceInputIndex {
    Vertices = 0,
    Surfaces = 1,
    ViewportSize = 2,
    TextureSize = 3,
    YTexture = 4,
    CbCrTexture = 5,
    YCbCrToRgb = 6,
}

#[repr(C)]
enum PathRasterizationInputIndex {
    Vertices = 0,
    ViewportSize = 1,
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct PathSprite {
    pub bounds: Bounds<ScaledPixels>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct SurfaceBounds {
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
}

#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
pub struct MetalHeadlessRenderer {
    renderer: MetalRenderer,
}

#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
impl MetalHeadlessRenderer {
    pub fn new() -> Self {
        let instance_buffer_pool = Arc::new(Mutex::new(InstanceBufferPool::default()));
        let renderer = MetalRenderer::new_headless(instance_buffer_pool);
        Self { renderer }
    }
}

#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
impl gpui::PlatformHeadlessRenderer for MetalHeadlessRenderer {
    fn render_scene_to_image(
        &mut self,
        scene: &Scene,
        size: Size<DevicePixels>,
    ) -> anyhow::Result<image::RgbaImage> {
        self.renderer.render_scene_to_image(scene, size)
    }

    fn render_scene(&mut self, scene: &Scene, size: Size<DevicePixels>) -> anyhow::Result<()> {
        self.renderer.render_scene(scene, size)
    }

    fn sprite_atlas(&self) -> Arc<dyn gpui::PlatformAtlas> {
        self.renderer.sprite_atlas().clone()
    }
}
