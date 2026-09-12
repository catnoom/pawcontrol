// Mosaic. params.x = block size in pixels.
//
// The grid is anchored to the screen, not the region, so blocks stay put while
// the hands move — matching the reference clip, where the mosaic reads as a
// fixed grid revealed through a moving window.
fn effect(uv: vec2<f32>) -> vec3<f32> {
    let res = resolution();
    let block = max(g.params.x, 1.0);
    let cell = (floor(uv * res / block) + 0.5) * block;
    return cam(cell / res);
}
