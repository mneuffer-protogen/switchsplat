// Critically damped spring motion, copied from Alyx-Physics src/spring.js. Pure: no three, no DOM, so
// node tests can import it. `response` is the settling time in seconds; `damping` 1 never overshoots.
// The viewer uses `smoothing` for hover highlights and `springStep` for the XR placement ease.
export const SPRING={
 response:.35,        // settling time in seconds
 damping:1,           // critically damped
 maxStep:1/30,        // dt clamp; a frame hitch must not detonate the integrator
 settleDistance:.02,  // close enough to the target to call it arrived
};

// Frame-rate independent exponential smoothing, for a fading opacity or a plain follow factor.
export const smoothing=(rate,dt)=>1-Math.exp(-rate*Math.max(dt,0));

// One axis. Semi-implicit Euler is only conditionally stable, so a stiff spring is subdivided into
// steps that satisfy the stability condition (this also makes the result frame-rate independent).
const SAFE_STEP=.5,MAX_SUBSTEPS=16;
export function springStep(current,velocity,target,dt,opts=SPRING){
 const step=Math.min(Math.max(dt,0),opts.maxStep);
 if(step===0)return {value:current,velocity};
 const w=2*Math.PI/opts.response;
 const substeps=Math.min(MAX_SUBSTEPS,Math.max(1,Math.ceil(step/(SAFE_STEP/w))));
 const h=step/substeps;
 let value=current,v=velocity;
 for(let i=0;i<substeps;i++){
  const accel=-(w*w)*(value-target)-(2*opts.damping*w)*v;
  v+=accel*h;value+=v*h;
 }
 return {value,velocity:v};
}
