// Sobel edge detection tinted into a neon glow.
// params.x = gain, params.y = hue cycling speed.
fn effect(uv: vec2<f32>) -> vec3<f32> {
    let texel = 1.0 / resolution();
    var gx = 0.0;
    var gy = 0.0;
    let kx = array<f32, 9>(-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0);
    let ky = array<f32, 9>(-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0);
    for (var i = 0; i < 9; i = i + 1) {
        let o = vec2<f32>(f32(i % 3) - 1.0, f32(i / 3) - 1.0) * texel;
        let l = luma(cam(uv + o));
        gx = gx + l * kx[i];
        gy = gy + l * ky[i];
    }
    let edge = clamp(sqrt(gx * gx + gy * gy) * g.params.x, 0.0, 1.0);
    let t = time() * g.params.y;
    let tint = 0.5 + 0.5 * cos(vec3<f32>(0.0, 2.094, 4.188) + t);
    return tint * edge;
}
