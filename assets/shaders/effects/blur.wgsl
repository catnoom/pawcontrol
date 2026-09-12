// Box blur. params.x = radius in pixels.
fn effect(uv: vec2<f32>) -> vec3<f32> {
    let texel = g.params.x / resolution();
    var sum = vec3<f32>(0.0);
    for (var y = -2; y <= 2; y = y + 1) {
        for (var x = -2; x <= 2; x = x + 1) {
            sum = sum + cam(uv + vec2<f32>(f32(x), f32(y)) * texel);
        }
    }
    return sum / 25.0;
}
