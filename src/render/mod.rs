//! GPU rendering: camera upload, masked effect pass, and debug overlay.

pub mod overlay;

use anyhow::{anyhow, Context, Result};
use std::borrow::Cow;
use std::sync::Arc;
use winit::window::Window;

use crate::effect::Effect;
use crate::frame::Frame;
use crate::region::Region;
use overlay::{LineInstance, Overlay};

const PRELUDE_HEAD: &str = include_str!("../../assets/shaders/prelude_head.wgsl");
const PRELUDE_TAIL: &str = include_str!("../../assets/shaders/prelude_tail.wgsl");

/// Uniform block shared by every effect. Laid out as vec4s so that WGSL's
/// std140-style alignment rules can't surprise us.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Globals {
    pub quad_a: [f32; 4],
    pub quad_b: [f32; 4],
    /// xy = resolution, z = time, w = region active
    pub view: [f32; 4],
    pub params: [f32; 4],
    /// rgb = outline colour, a = half-width px
    pub outline: [f32; 4],
    /// x = mirror, y = outline enabled
    pub flags: [f32; 4],
}

impl Default for Globals {
    fn default() -> Self {
        Self {
            quad_a: [0.0; 4],
            quad_b: [0.0; 4],
            view: [1.0, 1.0, 0.0, 0.0],
            params: [0.0; 4],
            outline: [1.0, 1.0, 1.0, 1.2],
            flags: [1.0, 1.0, 0.0, 0.0],
        }
    }
}

/// What the renderer needs to draw one frame.
pub struct RenderInput<'a> {
    pub frame: Option<&'a Frame>,
    pub region: Region,
    pub effect_index: usize,
    pub params: [f32; 4],
    pub time: f32,
    pub mirror: bool,
    pub show_outline: bool,
    /// Debug skeleton segments, in normalized screen space.
    pub lines: &'a [LineInstance],
}

pub struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,

    globals: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pipelines: Vec<wgpu::RenderPipeline>,
    overlay: Overlay,

    camera_texture: wgpu::Texture,
    camera_size: (u32, u32),
    /// Last camera frame uploaded, so we skip redundant transfers.
    last_uploaded: u64,
    needs_reconfigure: bool,

    pub adapter_name: String,
    pub backend: String,
}

impl Renderer {
    pub async fn new(
        window: Arc<Window>,
        camera_size: (u32, u32),
        effects: &[Box<dyn Effect>],
    ) -> Result<Self> {
        let size = window.inner_size();
        // `from_env` lets WGPU_BACKEND=dx12|vulkan override the pick at runtime.
        let instance = wgpu::Instance::new(
            wgpu::InstanceDescriptor::new_without_display_handle_from_env(),
        );
        let surface = instance
            .create_surface(window.clone())
            .context("creating window surface")?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                // Ask for the discrete AMD part rather than an integrated one.
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                ..Default::default()
            })
            .await
            .map_err(|e| anyhow!("no suitable GPU adapter: {e}"))?;

        let info = adapter.get_info();
        log::info!("GPU: {} ({:?}, {:?})", info.name, info.backend, info.device_type);

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("pawcontrol"),
                ..Default::default()
            })
            .await
            .map_err(|e| anyhow!("requesting GPU device: {e}"))?;

        let caps = surface.get_capabilities(&adapter);
        // Prefer an sRGB target so the shader can work in linear light and
        // still round-trip the camera image exactly.
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let mut config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .ok_or_else(|| anyhow!("surface is not supported by this adapter"))?;
        config.format = format;
        config.usage = wgpu::TextureUsages::RENDER_ATTACHMENT;
        config.present_mode = wgpu::PresentMode::AutoVsync;
        surface.configure(&device, &config);

        let camera_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("camera"),
            size: wgpu::Extent3d {
                width: camera_size.0,
                height: camera_size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let camera_view = camera_texture.create_view(&Default::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("camera-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let globals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals"),
            size: std::mem::size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("effect-bindings"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("effect-bind-group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: globals.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&camera_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("effect-layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });

        let mut pipelines = Vec::with_capacity(effects.len());
        for effect in effects {
            pipelines.push(build_effect_pipeline(
                &device,
                &pipeline_layout,
                format,
                effect.as_ref(),
            )?);
        }

        let overlay = Overlay::new(&device, &layout, format);

        Ok(Self {
            surface,
            device,
            queue,
            config,
            globals,
            bind_group,
            pipelines,
            overlay,
            camera_texture,
            camera_size,
            last_uploaded: 0,
            needs_reconfigure: false,
            adapter_name: info.name,
            backend: format!("{:?}", info.backend),
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
    }

    fn upload_camera(&mut self, frame: &Frame) {
        if frame.seq == self.last_uploaded {
            return;
        }
        if (frame.width, frame.height) != self.camera_size {
            log::warn!(
                "camera resolution changed to {}x{}; ignoring frame",
                frame.width,
                frame.height
            );
            return;
        }
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.camera_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &frame.rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(frame.width * 4),
                rows_per_image: Some(frame.height),
            },
            wgpu::Extent3d {
                width: frame.width,
                height: frame.height,
                depth_or_array_layers: 1,
            },
        );
        self.last_uploaded = frame.seq;
    }

    pub fn render(&mut self, input: RenderInput<'_>) -> Result<()> {
        if std::mem::take(&mut self.needs_reconfigure) {
            self.surface.configure(&self.device, &self.config);
        }
        if let Some(frame) = input.frame {
            self.upload_camera(frame);
        }

        let corners = input.region.corners().unwrap_or([glam::Vec2::ZERO; 4]);
        let globals = Globals {
            quad_a: [corners[0].x, corners[0].y, corners[1].x, corners[1].y],
            quad_b: [corners[2].x, corners[2].y, corners[3].x, corners[3].y],
            view: [
                self.config.width as f32,
                self.config.height as f32,
                input.time,
                if input.region.is_active() { 1.0 } else { 0.0 },
            ],
            params: input.params,
            outline: [1.0, 1.0, 1.0, 1.2],
            flags: [
                if input.mirror { 1.0 } else { 0.0 },
                if input.show_outline { 1.0 } else { 0.0 },
                0.0,
                0.0,
            ],
        };
        self.queue
            .write_buffer(&self.globals, 0, bytemuck::bytes_of(&globals));
        self.overlay.upload(&self.queue, &self.device, input.lines);

        let surface_texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) => t,
            // Suboptimal still gives us a usable texture; reconfigure so the
            // next frame is back on the fast path.
            wgpu::CurrentSurfaceTexture::Suboptimal(t) => {
                self.needs_reconfigure = true;
                t
            }
            // Transient states: skip this frame rather than treating it as an
            // error. Outdated/Lost additionally need a reconfigure.
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return Ok(());
            }
            other => return Err(anyhow!("acquiring swapchain image: {other:?}")),
        };

        let view = surface_texture.texture.create_view(&Default::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("effect"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            let index = input.effect_index.min(self.pipelines.len().saturating_sub(1));
            if let Some(pipeline) = self.pipelines.get(index) {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.draw(0..3, 0..1);
            }

            self.overlay.draw(&mut pass, &self.bind_group);
        }

        self.queue.submit(Some(encoder.finish()));
        self.queue.present(surface_texture);
        Ok(())
    }
}

/// Assemble an effect's full shader and compile it into a pipeline.
fn build_effect_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    format: wgpu::TextureFormat,
    effect: &dyn Effect,
) -> Result<wgpu::RenderPipeline> {
    // Order matters: WGSL has no forward declarations, so the effect body must
    // sit between the bindings it uses and the entry points that call it.
    let source = format!("{PRELUDE_HEAD}\n{}\n{PRELUDE_TAIL}", effect.shader());

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(effect.name()),
        source: wgpu::ShaderSource::Wgsl(Cow::Owned(source)),
    });

    Ok(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(effect.name()),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(format.into())],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compile every effect shader on a real device.
    ///
    /// Shaders are assembled and compiled at runtime, so a WGSL error would
    /// otherwise only surface when the window opens. This catches it in
    /// `cargo test` instead. Skips if the machine has no usable GPU adapter.
    #[test]
    fn every_effect_shader_compiles() {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let Ok(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let (device, _queue) = pollster::block_on(
            adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("shader-test"),
                ..Default::default()
            }),
        )
        .expect("requesting a device");

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });

        for effect in crate::effect::registry() {
            let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
            let _pipeline = build_effect_pipeline(
                &device,
                &pipeline_layout,
                wgpu::TextureFormat::Rgba8UnormSrgb,
                effect.as_ref(),
            )
            .expect("building the pipeline");
            if let Some(err) = pollster::block_on(scope.pop()) {
                panic!("effect '{}' failed to compile:\n{err}", effect.name());
            }
        }

        // The overlay shader ships separately from the effect prelude.
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _overlay = Overlay::new(&device, &layout, wgpu::TextureFormat::Rgba8UnormSrgb);
        if let Some(err) = pollster::block_on(scope.pop()) {
            panic!("overlay shader failed to compile:\n{err}");
        }
    }
}
