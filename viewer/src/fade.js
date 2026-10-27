// Opacity multiplier on a whole SplatMesh. It stays visible even at 0: Spark composes the visible SplatMeshes
// into one accumulator, and adding or removing one changes its "mapping". While a mapping change is pending
// Spark skips regeneration until the current sort finishes and keeps drawing the old accumulator, so every part
// froze for a whole sort (~0.5 s on a 450k-splat scene) when its reveal/vanish splats toggled, then jumped.
// A constant set of visible meshes keeps the mapping fixed, so a moving part is redrawn every frame.
export function setFade(mesh,v){
 const o=Math.min(1,Math.max(0,v));
 if(mesh.opacity!==o)mesh.opacity=o;
}
