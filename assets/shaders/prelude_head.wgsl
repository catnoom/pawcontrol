// Shared header for every effect shader.
//
// An effect only has to define:
//     fn effect(uv: vec2<f32>) -> vec3<f32>
// and may use anything declared here. `prelude_tail.wgsl` supplies the entry
// points; WGSL requires declaration before use, which is why the prelude is
// split around the effect body.

struct Globals {
    // Region corners 0,1 (xy, zw) and 2,3 (xy, zw), normalized.
    quad_a: vec4<f32>,
    quad_b: vec4<f32>,
    // xy = resolution in px, z = seconds, w = 1 when a region is active.
    view: vec4<f32>,
    // Per-effect tunables.
    params: vec4<f32>,
    // rgb = outline colour, a = outline half-width in px.
    outline: vec4<f32>,
    // x = mirror camera, y = outline enabled.
    flags: vec4<f32>,
};

@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(1) var cam_tex: texture_2d<f32>;
@group(0) @binding(2) var cam_smp: sampler;

fn resolution() -> vec2<f32> { return g.view.xy; }
fn time() -> f32 { return g.view.z; }

/// Sample the camera. Mirroring lives here so every other coordinate in the
/// pipeline (landmarks, region, mask) stays in one consistent screen space.
fn cam(uv: vec2<f32>) -> vec3<f32> {
    var c = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0));
    if (g.flags.x > 0.5) {
        c.x = 1.0 - c.x;
    }
    return textureSampleLevel(cam_tex, cam_smp, c, 0.0).rgb;
}

fn corner(i: i32) -> vec2<f32> {
    if (i == 0) { return g.quad_a.xy; }
    if (i == 1) { return g.quad_a.zw; }
    if (i == 2) { return g.quad_b.xy; }
    return g.quad_b.zw;
}

fn region_center() -> vec2<f32> {
    return (corner(0) + corner(1) + corner(2) + corner(3)) * 0.25;
}

fn cross2(a: vec2<f32>, b: vec2<f32>) -> f32 {
    return a.x * b.y - a.y * b.x;
}

/// Signed distance (in pixels) from `p` to the region quad: negative inside.
///
/// Works for either winding order by deriving the orientation from the
/// polygon's signed area, so the effect survives the user crossing their hands.
fn region_sdf(p_px: vec2<f32>) -> f32 {
    let res = resolution();
    var area = 0.0;
    for (var i = 0; i < 4; i = i + 1) {
        area = area + cross2(corner(i) * res, corner((i + 1) % 4) * res);
    }
    let s = select(-1.0, 1.0, area > 0.0);

    var d = -1.0e9;
    for (var i = 0; i < 4; i = i + 1) {
        let a = corner(i) * res;
        let b = corner((i + 1) % 4) * res;
        let e = b - a;
        let len = length(e);
        // A zero-length edge (coincident corners) has no normal; skipping it
        // avoids poisoning the max with a bogus zero distance.
        if (len < 1.0e-4) {
            continue;
        }
        // Outward normal for this winding.
        let n = (vec2<f32>(e.y, -e.x) / len) * s;
        d = max(d, dot(p_px - a, n));
    }
    return d;
}

/// 1 inside the region, 0 outside, antialiased across one pixel.
fn region_mask(uv: vec2<f32>) -> f32 {
    if (g.view.w < 0.5) { return 0.0; }
    return 1.0 - smoothstep(-1.0, 1.0, region_sdf(uv * resolution()));
}

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}
