// Chromatic aberration radiating from the region centre.
// params.x = displacement in pixels.
fn effect(uv: vec2<f32>) -> vec3<f32> {
    let dir = normalize(uv - region_center() + vec2<f32>(1.0e-5));
    let off = dir * (g.params.x / resolution());
    return vec3<f32>(
        cam(uv + off).r,
        cam(uv).g,
        cam(uv - off).b,
    );
}
