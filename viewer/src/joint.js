// A door, cabinet door or drawer as one rigid part on a 1-DOF joint: a hinge's turn about its axis or a
// slider's travel along it. Ported from Alyx-Physics src/physics/joints.js (JointedPart) without
// cannon-es: there are no rigid bodies or contacts here, only the one coordinate q (radians for a
// hinge, metres for a slider) and its speed qd, so friction, springs, limits and the latch all act on
// that single number and the part's pose is set exactly from it (nothing can drift).
//
// Dropped from the original: contact projection, pushSpeed, writeVelocity, gravity. (holdContacts lives
// on as `pushJoint` in push.js: a hand is the only thing that touches the part.)
// Added: `animateTo` (a critically damped drive for click-to-open, optionally with an initial kick), a
// `locked` state with a handle jiggle (Alyx doorLatch), `applyPose` onto an Object3D.
import * as THREE from 'three';
import {DEG,clamp,inverseLerp} from './math.js';

export const HINGE='hinge',SLIDER='slider';
// The joint is integrated at a fixed rate, whatever the frame rate; slow frames cap at MAX_SUBSTEPS
// so a hitch cannot snowball into a stall.
export const STEP=1/90,MAX_SUBSTEPS=4;
// The handle rattle of a locked part: amplitude, frequency, decay and length.
const JIGGLE_DEG=1.5,JIGGLE_M=.005,JIGGLE_HZ=9,JIGGLE_TAU=.13,JIGGLE_TIME=.45;

export class FixedStepper{
 constructor({step=STEP,maxSubsteps=MAX_SUBSTEPS}={}){this.step=step;this.maxSubsteps=maxSubsteps;this.acc=0}
 // Calls fn(step) as many whole steps as `dt` holds; returns the leftover time (for pose extrapolation).
 run(dt,fn){
  this.acc+=Math.max(0,dt);let n=0;
  while(this.acc>=this.step&&n<this.maxSubsteps){fn(this.step);this.acc-=this.step;n++}
  if(n===this.maxSubsteps)this.acc=Math.min(this.acc,this.step);
  return this.acc;
 }
}

export class JointedPart{
 // `box` is the part's bounding box at rest in the joint's frame ({center,size,quaternion}); it only
 // sizes the generalised mass. `object` (optional) is the Object3D posed by applyPose; its pose when
 // given is the rest pose (q = 0).
 constructor({object=null,kind=HINGE,axis,pivot=new THREE.Vector3(),min=0,max=1,open=max,friction=0,frictionTorque=0,spring=0,damper=0,rest=0,latch=0,maxSpeed=0,mass=2,box=null,activateAt=.9,deactivateAt=.1,locked=false,latchReleaseDelay=.25}={}){
  Object.assign(this,{object,kind,min,max,open,friction,frictionTorque,spring,damper,rest,latch,maxSpeed,bodyMass:mass,activateAt,deactivateAt,latchReleaseDelay});
  this.axis=new THREE.Vector3().copy(axis).normalize();this.pivot=new THREE.Vector3().copy(pivot);
  this.box=box?{center:box.center.clone(),size:box.size.clone(),quaternion:(box.quaternion??new THREE.Quaternion()).clone()}:{center:this.pivot.clone(),size:new THREE.Vector3(.4,.4,.4),quaternion:new THREE.Quaternion()};
  this.restPosition=object?object.position.clone():new THREE.Vector3();
  this.restQuaternion=object?object.quaternion.clone():new THREE.Quaternion();
  this.q=0;this.qd=0;this.lo=min;this.hi=max;this.latched=false;this.locked=!!locked;this.active=false;this.lastValue=-1;
  this.holders=0;this.holdTime=0;this.hitSpeed=0;this.jig=null;this.anim=null;this.listeners={};this.canUnlatch=null;this._drawn=NaN;
  this.mass=this.generalizedMass();this.reach=this.computeReach();
  this.refreshLimits();
 }
 on(name,fn){(this.listeners[name]??=new Set()).add(fn);return ()=>this.listeners[name].delete(fn)}
 emit(name,value){for(const fn of this.listeners[name]??[])fn(value)}
 // Inertia about the hinge axis through the pivot, or the mass along the slide. A box of mass m turns
 // about its own centre with the principal inertias below; the parallel-axis theorem moves that to
 // the pivot line.
 generalizedMass(){
  const m=this.bodyMass;if(this.kind===SLIDER)return Math.max(1e-3,m);
  const {center,size,quaternion}=this.box;
  const Ix=m/12*(size.y*size.y+size.z*size.z),Iy=m/12*(size.x*size.x+size.z*size.z),Iz=m/12*(size.x*size.x+size.y*size.y);
  const la=this.axis.clone().applyQuaternion(quaternion.clone().invert());
  const d=center.clone().sub(this.pivot),dPerp=d.addScaledVector(this.axis,-d.dot(this.axis));
  return Math.max(1e-4,la.x*la.x*Ix+la.y*la.y*Iy+la.z*la.z*Iz+m*dPerp.lengthSq());
 }
 // The part's farthest point from the hinge line (metres): how far its free edge travels per radian.
 computeReach(){
  if(this.kind===SLIDER)return 0;
  const {center,size,quaternion}=this.box,c=new THREE.Vector3();let far=0;
  for(const sx of [-.5,.5])for(const sy of [-.5,.5])for(const sz of [-.5,.5]){
   c.set(size.x*sx,size.y*sy,size.z*sz).applyQuaternion(quaternion).add(center).sub(this.pivot);
   far=Math.max(far,c.addScaledVector(this.axis,-c.dot(this.axis)).length());
  }
  return far;
 }
 // Change the joint's settings (the edit panel); the axis, mass and limits are rebuilt.
 configure(opts){
  for(const k of ['kind','min','max','open','friction','frictionTorque','spring','damper','rest','latch','maxSpeed','activateAt','deactivateAt','latchReleaseDelay'])if(opts[k]!==undefined)this[k]=opts[k];
  if(opts.mass!==undefined)this.bodyMass=opts.mass;
  if(opts.axis)this.axis.copy(opts.axis).normalize();
  if(opts.pivot)this.pivot.copy(opts.pivot);
  if(opts.box)this.box={center:opts.box.center.clone(),size:opts.box.size.clone(),quaternion:(opts.box.quaternion??new THREE.Quaternion()).clone()};
  this.mass=this.generalizedMass();this.reach=this.computeReach();
  this.latched=false;this.q=clamp(this.q,this.min,this.max);this.refreshLimits();this._drawn=NaN;
 }
 get angle(){return this.q}
 // 0..1 across the whole travel.
 get value(){return inverseLerp(this.min,this.max,this.q)}
 // 0 closed .. 1 open, which is what the reveal/vanish fades and the click toggle follow.
 get fraction(){return this.open>1e-9?clamp(this.q/this.open,0,1):0}
 // The coordinate a click closes to / opens to.
 get closedQ(){return clamp(0,this.min,this.max)}
 get openQ(){return clamp(this.open,this.min,this.max)}
 get tolerance(){return this.kind===HINGE?.02*DEG:1e-4}
 setLimits(lo,hi){this.lo=lo;this.hi=hi}
 // How far past the latch's end stop it is held; a hair, so the part cannot be pushed out.
 get latchHold(){return this.kind===HINGE?.01*DEG:1e-4}
 // The limits the state implies: pinned where it stands when locked, at the closed stop when latched.
 refreshLimits(){
  if(this.locked)this.setLimits(this.q,this.q+this.latchHold);
  else if(this.latched)this.setLimits(this.min,this.min+this.latchHold);
  else this.setLimits(this.min,this.max);
 }
 unlatch(){
  if(!this.latched)return false;
  this.latched=false;this.refreshLimits();this.emit('unlatched');return true;
 }
 // Lock (or unlock) the part where it stands: a locked part ignores pushes, grabs and clicks (they only
 // jiggle it) until unlocked. A latched part stays latched underneath.
 setLocked(v){
  v=!!v;if(v===this.locked)return;
  this.locked=v;
  if(v){this.qd=0;this.anim=null}
  this.refreshLimits();this.emit('locked',v);
 }
 // The handle rattle of a locked part (Alyx doorLatch): a small decaying spring-back oscillation about
 // the pinned pose, 1.5 degrees (a hinge) or 5 mm (a slide). Only the drawn pose moves, never `q`.
 jiggle(){
  if(this.jig)return false;
  this.jig={t:0};this.emit('jiggle');return true;
 }
 jiggleOffset(){
  const j=this.jig;if(!j)return 0;
  return (this.kind===HINGE?JIGGLE_DEG*DEG:JIGGLE_M)*Math.sin(2*Math.PI*JIGGLE_HZ*j.t)*Math.exp(-j.t/JIGGLE_TAU);
 }
 // The closing speed above which a part latching at its stop counts as a slam.
 get slamSpeed(){return this.kind===HINGE?1.5:.5}
 // Whatever was driving the part (a click animation) lets go of it; a push or a grab does this.
 interrupt(){this.anim=null}
 // Settling time (s) of a click animation for this part: a car door takes about 0.9 s, a cabinet door 0.6,
 // a drawer 0.45; sized by how far the free edge travels and a little by the mass.
 responseFor(target){
  const d=Math.abs(clamp(target,this.min,this.max)-this.q),arc=this.kind===HINGE?this.reach*d:d;
  return clamp(.15+.55*arc+.02*Math.sqrt(this.bodyMass),.3,1.4);
 }

 // ---------------------------------------------------------------- pose

 _rot(q,out=_q){return out.setFromAxisAngle(this.axis,q)}
 // A point fixed to the part (given at rest, q = 0) as it stands at coordinate q.
 pointAt(local,q=this.q,out=new THREE.Vector3()){
  if(this.kind===HINGE)return out.copy(local).sub(this.pivot).applyQuaternion(this._rot(q)).add(this.pivot);
  return out.copy(local).addScaledVector(this.axis,q);
 }
 // Inverse of pointAt: where a point currently on the part sits on the part at rest.
 toRest(point,q=this.q,out=new THREE.Vector3()){
  if(this.kind===HINGE)return out.copy(point).sub(this.pivot).applyQuaternion(this._rot(-q)).add(this.pivot);
  return out.copy(point).addScaledVector(this.axis,-q);
 }
 // Write the part's transform from its rest pose: turn about the axis through the pivot, or slide.
 // `dq` extrapolates the drawn pose by a fraction of a step so a 144 Hz display does not stutter on
 // a 90 Hz joint.
 applyPose(dq=0){
  const o=this.object;if(!o)return;
  let q=dq?clamp(this.q+this.qd*dq,this.lo,this.hi):this.q;
  if(this.jig)q=clamp(q+this.jiggleOffset(),this.min,this.max);
  if(q===this._drawn)return;// nothing moved: leave the transform alone so Spark has nothing to redo
  this._drawn=q;
  if(this.kind===HINGE){
   const rot=this._rot(q);
   o.position.copy(this.restPosition).sub(this.pivot).applyQuaternion(rot).add(this.pivot);
   o.quaternion.copy(this.restQuaternion).premultiply(rot);
  }else{o.position.copy(this.restPosition).addScaledVector(this.axis,q);o.quaternion.copy(this.restQuaternion)}
  o.updateMatrix();o.matrixWorldNeedsUpdate=true;
 }
 // Place the part at a coordinate immediately (scene setup, restoring a saved pose, the scrub slider).
 setPosition(q){this.anim=null;this.latched=false;this.q=clamp(q,this.min,this.max);this.qd=0;this.refreshLimits();this.applyPose()}

 // ---------------------------------------------------------------- impulses and holds

 applyImpulse(p){this.qd+=p/this.mass}
 // A hand or the mouse takes hold: whatever was animating the part lets go of it. Two hands can hold one part.
 get held(){return this.holders>0}
 grab(){this.holders++;this.anim=null;if(this.locked)this.jiggle()}
 release(){this.holders=Math.max(0,this.holders-1)}
 // A part pushed (not held) goes no faster than `maxSpeed` (m/s along a slide, rad/s about a hinge; 0 = no limit).
 limitSpeed(qd){
  if(this.held||!(this.maxSpeed>0))return qd;
  return clamp(qd,-this.maxSpeed,this.maxSpeed);
 }
 // A critically damped drive to coordinate `q` (click to open/close). `response` is its settling time in
 // seconds (default .5; `style:'swing'` sizes it by the part, see responseFor). It is the same implicit
 // spring/damper as `preStep`, so it is stable at any step and a new target (or a grab, or a push) just
 // takes over from wherever the part is, with whatever speed it has.
 // `style:'swing'` also gives the part a shove first: it starts at a fraction (`kick`) of the speed that
 // would just reach the target, so a door leaves the latch briskly and eases into the end instead of
 // creeping out of rest. A critically damped spring only overshoots when it starts faster than w*distance,
 // so the kick (< 1) never does. A locked part only jiggles.
 animateTo(q,{response,style='drive',kick=.3}={}){
  const target=clamp(q,this.min,this.max),delta=target-this.q;
  if(this.locked){this.anim=null;if(Math.abs(delta)>this.tolerance)this.jiggle();return}
  // A latched door would hold itself shut against the drive.
  if(this.latched&&target>this.q+this.tolerance)this.unlatch();
  const swing=style==='swing';
  const r=Math.max(.05,response??(swing?this.responseFor(target):.5));
  this.anim={target,response:r,time:0};
  if(swing&&Math.abs(delta)>this.tolerance*4){
   const dir=Math.sign(delta),w=2*Math.PI/r,v=dir*Math.min(kick*w*Math.abs(delta),this.maxSpeed>0?this.maxSpeed:Infinity);
   if(v*dir>this.qd*dir)this.qd=v;
  }
 }
 get animating(){return !!this.anim}
 // The controller pulls the grip point a fraction of the way to the hand each step, sized by the
 // mass the part really has along its one motion, so it cannot overshoot (Unity ConstrainedImpulse:
 // a spring sized by total mass made light doors oscillate at tens of rad/s).
 drag(anchor,target,dt,{constrainedFollow=.35,constrainedMaxSpeed=4,constrainedDriveMaxForce=450}={}){
  const maxImpulse=constrainedDriveMaxForce*dt;
  const vWant=target.clone().sub(anchor).multiplyScalar(constrainedFollow/dt);
  if(vWant.length()>constrainedMaxSpeed)vWant.setLength(constrainedMaxSpeed);
  if(this.kind===HINGE){
   const r=anchor.clone().sub(this.pivot),rPerp=r.addScaledVector(this.axis,-r.dot(this.axis)),lever=rPerp.length();
   if(lever<.01)return;// held on the axis itself: pulling cannot turn it
   const tangent=new THREE.Vector3().crossVectors(this.axis,rPerp).divideScalar(lever);
   const impulse=this.mass*(vWant.dot(tangent)/lever-this.qd);
   this.applyImpulse(clamp(impulse,-maxImpulse*lever,maxImpulse*lever));
  }else this.applyImpulse(clamp(this.mass*(vWant.dot(this.axis)-this.qd),-maxImpulse,maxImpulse));
 }

 // ---------------------------------------------------------------- step

 // One fixed step: latch, events, then the dynamics and the integration.
 step(dt){this.fixedStep(dt);this.preStep(dt);this.postStep(dt)}
 fixedStep(dt){
  this.holdTime=this.held?this.holdTime+dt:0;
  // The latch waits while a click animation is still driving the part, so it never snaps the last degrees.
  // A part that arrives fast (a slam) has stopped dead against its end stop by the next step and latches then.
  if(this.latch>0&&!this.locked){
   if(!this.latched&&!this.held&&!this.anim&&this.q-this.min<this.latch&&Math.abs(this.qd)<1.5){
    const hit=this.hitSpeed;this.hitSpeed=0;
    this.latched=true;this.refreshLimits();this.emit('latched');
    if(hit>this.slamSpeed)this.emit('slam',hit);
   }
   // Only a hand that has hold of the part for a moment (a hand on the handle) lets the latch go. A push never does.
   else if(this.latched&&this.held&&this.holdTime>=this.latchReleaseDelay&&(!this.canUnlatch||this.canUnlatch()))this.unlatch();
  }
  if(this.q-this.min>=this.latch)this.hitSpeed=0;
  const v=this.value;
  if(Math.abs(v-this.lastValue)>1e-5){
   this.emit('value',v);
   if(!this.active&&v>=this.activateAt){this.active=true;this.emit('activated')}
   else if(this.active&&v<=this.deactivateAt){this.active=false;this.emit('deactivated')}
   this.lastValue=v;
  }
 }
 // The forces on the one coordinate: spring and damper, friction, the speed cap and the end stops.
 preStep(dt){
  const M=this.mass,a=this.anim;let qd=this.qd;
  if(a){
   // Implicit spring: stiffness w^2 and damping 2w per unit mass is critically damped (settles in `response`).
   const w=2*Math.PI/a.response;
   qd=(qd-dt*w*w*(this.q-a.target))/(1+2*dt*w+dt*dt*w*w);
  }else{
   // Spring and damper, implicit so a stiff spring on a light part stays stable at any step size.
   if(this.spring>0)qd=(qd-dt*this.spring/M*(this.q-this.rest))/(1+dt*this.damper/M+dt*dt*this.spring/M);
   if(this.friction>0)qd/=1+dt*this.friction;
   // Coulomb friction at the joint (a zero-speed motor with a torque cap): stops a swinging frame.
   if(this.frictionTorque>0){const dq=this.frictionTorque*dt/M;qd=Math.abs(qd)<=dq?0:qd-Math.sign(qd)*dq}
   qd=this.limitSpeed(qd);
  }
  const next=this.q+qd*dt;
  if(next<this.lo)qd=(this.lo-this.q)/dt;else if(next>this.hi)qd=(this.hi-this.q)/dt;
  const atEnd=(this.q<=this.lo+1e-6&&qd<=0)||(this.q>=this.hi-1e-6&&qd>=0);
  if(atEnd&&Math.abs(this.qd)>.05)this.emit('limit',Math.abs(this.qd));
  // How hard it struck the closed stop, for the latch's slam.
  if(atEnd&&!this.locked&&!this.latched&&this.q<=this.min+1e-6&&this.qd<0)this.hitSpeed=Math.max(this.hitSpeed,-this.qd);
  this.qd=qd;
 }
 postStep(dt){
  // qd was chosen to land inside the limits, so this clamp only catches rounding.
  this.q=clamp(this.q+this.qd*dt,this.lo,this.hi);
  if(this.jig&&(this.jig.t+=dt)>=JIGGLE_TIME)this.jig=null;
  const a=this.anim;
  if(a){
   a.time+=dt;
   const tol=this.tolerance;
   // Arrived (or the limits will never let it): land exactly on the target and hand the part back.
   if((Math.abs(this.q-a.target)<tol&&Math.abs(this.qd)<tol*4)||a.time>a.response*4){
    this.q=clamp(a.target,this.lo,this.hi);this.qd=0;this.anim=null;this.emit('settled');
   }
  }
 }
}
const _q=new THREE.Quaternion();
