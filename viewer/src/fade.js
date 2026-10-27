// Opacity multiplier on a whole SplatMesh. It stays visible even at 0: Spark composes the visible SplatMeshes
// into one accumulator, and adding or removing one changes its "mapping". While a mapping change is pending
// Spark skips regeneration until the current sort finishes and keeps drawing the old accumulator, so every part
// froze for a whole sort (~0.5 s on a 450k-splat scene) when its reveal/vanish splats toggled, then jumped.
// A constant set of visible meshes keeps the mapping fixed, so a moving part is redrawn every frame.
export function setFade(mesh,v){
 const o=Math.min(1,Math.max(0,v));
 if(mesh.opacity!==o)mesh.opacity=o;
}

// How reveal/vanish splats follow the part: a steep smoothstep over a short stretch of the travel, not a linear
// ramp. Linear meant halfway open the interior was half-faded and the closed door's leftover splats half-visible,
// i.e. ghosts fading in and out while the door was still moving. `reveal` is the stretch (fraction of travel) over
// which the uncovered content appears, `vanish` the stretch over which the closed-only content disappears; scene.json
// joint.fade {reveal:[a,b], vanish:[a,b], panel:[a,b]} overrides them per part.
// `panel` is the same for partReveal: the panel's own open-pose copy (inner face, glass, edge). It is a child of the
// part's group, so it moves with the panel and can come in earlier than a world-fixed reveal ever could; a copy
// left fixed at the open pose would show the panel's end position while the real one is still near closed.
export const FADE_DEFAULTS={reveal:[.04,.2],vanish:[0,.12],panel:[.15,.5]};
export function smoothstep(a,b,x){
 if(!(b>a))return x>=b?1:0;
 const t=Math.min(1,Math.max(0,(x-a)/(b-a)));return t*t*(3-2*t);
}
export function fadeLevels(fraction,fade){
 const r=fade?.reveal??FADE_DEFAULTS.reveal,v=fade?.vanish??FADE_DEFAULTS.vanish,p=fade?.panel??FADE_DEFAULTS.panel;
 return {reveal:smoothstep(r[0],r[1],fraction),vanish:1-smoothstep(v[0],v[1],fraction),panel:smoothstep(p[0],p[1],fraction)};
}

// One call per part and frame: reveal, vanish and the part-attached partReveal follow the joint fraction. None of
// them is ever hidden (see setFade). partReveal (the `panel` curve) sits inside the part's group, so it moves with the panel.
export function applyFades(part,fraction){
 const l=fadeLevels(fraction,part.fade);
 if(part.reveal)setFade(part.reveal,l.reveal);
 if(part.vanish)setFade(part.vanish,l.vanish);
 if(part.partReveal)setFade(part.partReveal,l.panel);
 return l;
}
