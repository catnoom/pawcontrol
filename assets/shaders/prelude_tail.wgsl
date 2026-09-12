// Entry points, appended after the effect body.

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    // Oversized triangle covering the viewport; cheaper than a quad.
    var verts = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    let p = verts[vi];
    var out: VsOut;
    out.pos = vec4<f32>(p, 0.0, 1.0);
    // Flip y: texture row 0 is the top of the image, NDC +y is the top.
    out.uv = p * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let base = cam(in.uv);
    var color = base;

    let mask = region_mask(in.uv);
    if (mask > 0.0) {
        color = mix(base, effect(in.uv), mask);
    }

    // Thin outline traced on the region boundary.
    if (g.view.w > 0.5 && g.flags.y > 0.5) {
        let d = abs(region_sdf(in.uv * resolution()));
        let w = max(g.outline.a, 0.5);
        let edge = 1.0 - smoothstep(w - 1.0, w + 1.0, d);
        color = mix(color, g.outline.rgb, edge);
    }

    return vec4<f32>(color, 1.0);
}
