//! Instanced line renderer, used for the debug hand skeleton.
//!
//! Each segment is one instance; the vertex shader expands it into a quad in
//! pixel space. That keeps line width constant on screen regardless of how the
//! window is sized.

use glam::Vec2;

/// One line segment in normalized screen space.
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LineInstance {
    pub a: [f32; 2],
    pub b: [f32; 2],
    pub color: [f32; 4],
    /// Half-width in pixels.
    pub width: f32,
    pub _pad: [f32; 3],
}

impl LineInstance {
    pub fn new(a: Vec2, b: Vec2, color: [f32; 4], width: f32) -> Self {
        Self {
            a: a.to_array(),
            b: b.to_array(),
            color,
            width,
            _pad: [0.0; 3],
        }
    }
}

const SHADER: &str = r#"
struct Globals {
    quad_a: vec4<f32>,
    quad_b: vec4<f32>,
    view: vec4<f32>,
    params: vec4<f32>,
    outline: vec4<f32>,
    flags: vec4<f32>,
};

@group(0) @binding(0) var<uniform> g: Globals;

struct Instance {
    @location(0) a: vec2<f32>,
    @location(1) b: vec2<f32>,
    @location(2) color: vec4<f32>,
    @location(3) width: f32,
};

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, inst: Instance) -> VsOut {
    let res = g.view.xy;
    let a = inst.a * res;
    let b = inst.b * res;

    let dir = b - a;
    let len = max(length(dir), 1.0e-6);
    let t = dir / len;
    let n = vec2<f32>(-t.y, t.x) * max(inst.width, 0.5);

    // Two triangles: 0,1,2 and 2,1,3.
    var offsets = array<vec2<f32>, 6>(
        vec2<f32>(0.0, -1.0), vec2<f32>(0.0, 1.0), vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, -1.0), vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 1.0),
    );
    let o = offsets[vi];
    let p = mix(a, b, o.x) + n * o.y;

    // Pixel space -> NDC, flipping y back.
    let ndc = vec2<f32>(p.x / res.x * 2.0 - 1.0, 1.0 - p.y / res.y * 2.0);

    var out: VsOut;
    out.pos = vec4<f32>(ndc, 0.0, 1.0);
    out.color = inst.color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return in.color;
}
"#;

pub struct Overlay {
    pipeline: wgpu::RenderPipeline,
    buffer: Option<wgpu::Buffer>,
    capacity: usize,
    count: u32,
}

impl Overlay {
    pub fn new(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        format: wgpu::TextureFormat,
    ) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("overlay"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("overlay-layout"),
            bind_group_layouts: &[Some(bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("overlay"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<LineInstance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![
                        0 => Float32x2,
                        1 => Float32x2,
                        2 => Float32x4,
                        3 => Float32
                    ],
                })],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            pipeline,
            buffer: None,
            capacity: 0,
            count: 0,
        }
    }

    pub fn upload(&mut self, queue: &wgpu::Queue, device: &wgpu::Device, lines: &[LineInstance]) {
        self.count = lines.len() as u32;
        if lines.is_empty() {
            return;
        }
        // Grow geometrically so a varying hand count doesn't reallocate often.
        if self.capacity < lines.len() {
            let capacity = lines.len().next_power_of_two().max(64);
            self.buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("overlay-lines"),
                size: (capacity * std::mem::size_of::<LineInstance>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.capacity = capacity;
        }
        if let Some(buffer) = &self.buffer {
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(lines));
        }
    }

    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>, bind_group: &wgpu::BindGroup) {
        let (Some(buffer), true) = (&self.buffer, self.count > 0) else {
            return;
        };
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.set_vertex_buffer(0, buffer.slice(..));
        pass.draw(0..6, 0..self.count);
    }
}
