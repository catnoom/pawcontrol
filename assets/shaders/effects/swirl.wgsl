// Twists the image around the region centre.
// params.x = strength in radians, params.y = falloff radius (normalized).
fn effect(uv: vec2<f32>) -> vec3<f32> {
    let res = resolution();
    let aspect = vec2<f32>(res.x / res.y, 1.0);
    let c = region_center();
    let d = (uv - c) * aspect;
    let r = length(d);
    let falloff = max(g.params.y, 1.0e-4);
    // Rotation decays to zero at the falloff radius, so the effect blends into
    // the untouched image instead of tearing at the region edge.
    let amount = g.params.x * max(1.0 - r / falloff, 0.0);
    let s = sin(amount);
    let co = cos(amount);
    let rotated = vec2<f32>(d.x * co - d.y * s, d.x * s + d.y * co);
    return cam(c + rotated / aspect);
}
