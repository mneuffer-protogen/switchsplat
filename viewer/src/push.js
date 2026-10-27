// Pushing a part with a hand (a controller, or the mouse). Alyx-Physics gives its hands real colliders and
// lets JointedPart.holdContacts trade momentum with each body touching the part; here the hand is a
// sphere that is not simulated at all (it is the tracked controller), so it is a body of infinite mass:
//
//   every fixed step, for each part the hand does not hold:
//     1. test the hand sphere against the part's oriented box at the part's current pose
//     2. contact point = closest point on the box, normal n = from the surface toward the hand
//     3. jn = how fast the surface moves along n per unit of joint speed
//          hinge  (axis x (p - pivot)) . n      slider  axis . n
//     4. sP = qd * jn is the surface's speed along n, sB = hand velocity . n. If sB < sP the hand is
//        closing on the surface faster than the surface gives way: after an impulse the two normal speeds
//        are equal, and with an immovable hand that is qd = sB / jn (Alyx: p = (sP - sB) / (invMass +
//        jn^2/M) with invMass = 0). The new speed is capped, then cut to what the limits allow, so a latched
//        or locked part (limits pinned at its pose) or a part at its stop does not move: the hand just stops.
//
// There is no penetration correction beyond the speed match. A push never unlatches (that takes a grab).
// Positions and the hand's velocity are in the joint's frame (the export's coordinates and units), like drive.js, so
// surface and hand speeds are compared in the same space. Real-world sizes and speeds (the hand radius, the speed caps,
// the contact margin: metres) are divided by joint.unit (metres per export unit) here, once.
import * as THREE from 'three';
import {HINGE} from './joint.js';
import {clamp} from './math.js';
import {smoothing} from './spring.js';
import {CONTACT_MARGIN} from './contact.js';

export const HAND_RADIUS=.045;
// A pushed part goes no faster than this, however hard it is shoved (rad/s about a hinge, m/s along a slide).
export const PUSH_MAX_SPEED={hinge:6,slider:3};
// Below this the contact is too close to the hinge line (m), or too far across a slide (cosine), to move the part.
const MIN_JN={hinge:.03,slider:.15};

const _hr=new THREE.Vector3(),_hp=new THREE.Vector3(),_hn=new THREE.Vector3();
const _k4=new THREE.Quaternion(),_contact={point:new THREE.Vector3(),normal:new THREE.Vector3(),depth:0};
const _c=new THREE.Vector3(),_q=new THREE.Quaternion(),_l=new THREE.Vector3(),_k=new THREE.Vector3(),_n=new THREE.Vector3(),_r=new THREE.Vector3(),_rot=new THREE.Quaternion();

// The part's box (given at rest in joint.box) as it stands at coordinate q.
export function boxPoseAt(joint,q,center=new THREE.Vector3(),quaternion=new THREE.Quaternion()){
 joint.pointAt(joint.box.center,q,center);
 quaternion.copy(joint.box.quaternion);
 if(joint.kind===HINGE)quaternion.premultiply(_rot.setFromAxisAngle(joint.axis,q));
 return {center,quaternion};
}

// Sphere against the part's current box. Returns {point,normal,depth} (point on the box surface, normal from the
// box toward the sphere, depth how far the sphere reaches into the box) or null when they do not touch.
export function sphereObbContact(joint,center,radius,out={point:new THREE.Vector3(),normal:new THREE.Vector3(),depth:0}){
 const pose=boxPoseAt(joint,joint.q,_c,_q),half=joint.box.size,hx=half.x/2,hy=half.y/2,hz=half.z/2;
 _l.copy(center).sub(pose.center).applyQuaternion(_k4.copy(pose.quaternion).invert());
 _k.set(clamp(_l.x,-hx,hx),clamp(_l.y,-hy,hy),clamp(_l.z,-hz,hz));
 _n.copy(_l).sub(_k);
 const dist=_n.length();
 if(dist>radius)return null;
 if(dist>1e-9){_n.divideScalar(dist);out.depth=radius-dist}
 else{
  // The centre is inside the box: leave through the nearest face.
  const dx=hx-Math.abs(_l.x),dy=hy-Math.abs(_l.y),dz=hz-Math.abs(_l.z),m=Math.min(dx,dy,dz);
  _n.set(0,0,0);
  if(m===dx){const s=_l.x<0?-1:1;_n.x=s;_k.x=s*hx}
  else if(m===dy){const s=_l.y<0?-1:1;_n.y=s;_k.y=s*hy}
  else{const s=_l.z<0?-1:1;_n.z=s;_k.z=s*hz}
  out.depth=radius+m;
 }
 out.normal.copy(_n).applyQuaternion(pose.quaternion);
 out.point.copy(_k).applyQuaternion(pose.quaternion).add(pose.center);
 return out;
}

// Sphere against the part's surface splat centres (joint.contacts, in the part's rest space): touching when the hand
// is within radius + CONTACT_MARGIN of a centre. Contact point = that centre as it stands now. The normal is the part's
// thin PCA axis turned with the part, signed to point toward the hand; for a part with no clear thin axis it is the
// direction from the centre to the hand.
// `radius` is in export units (the caller converts a hand size in metres); the margin is converted here.
export function contactSetContact(joint,set,center,radius,out={point:new THREE.Vector3(),normal:new THREE.Vector3(),depth:0}){
 const reach=radius+CONTACT_MARGIN/joint.unit;
 const hit=set.nearest(joint.toRest(center,joint.q,_hr),reach);
 if(!hit)return null;
 joint.pointAt(set.point(hit.index,_hp),joint.q,out.point);
 _hn.copy(center).sub(out.point);
 const away=_hn.length();
 if(set.thinReliable){
  out.normal.copy(set.thinAxis);
  if(joint.kind===HINGE)out.normal.applyQuaternion(_rot.setFromAxisAngle(joint.axis,joint.q));
  if(out.normal.dot(_hn)<0)out.normal.negate();
 }else if(away>1e-6)out.normal.copy(_hn).divideScalar(away);
 else out.normal.copy(set.thinAxis);
 out.depth=reach-hit.distance;
 return out;
}

// One hand-versus-part exchange for one fixed step. `hand` = {position, velocity, radius}: position and velocity
// in the joint's frame (export units, smoothed), radius in metres. Returns null when the sphere is not touching the part, else
//   {closing, pushed, blocked, depth}: closing speed (m/s, real) between hand and surface (0 when separating),
//   pushed = the part's speed was changed, blocked = the part could not give way (latched, locked, at a stop,
//   or pushed across its path).
export function pushJoint(joint,hand,dt,{maxSpeed}={}){
 if(!joint.box&&!joint.contacts)return null;
 const radius=(hand.radius??HAND_RADIUS)/joint.unit;// metres to export units
 // A part with a contact set is touched where its splats are; the box is only for parts that have none.
 const c=joint.contacts?contactSetContact(joint,joint.contacts,hand.position,radius,_contact):sphereObbContact(joint,hand.position,radius,_contact);
 if(!c)return null;
 const hinge=joint.kind===HINGE,n=c.normal;
 const jn=hinge?_r.crossVectors(joint.axis,_r.copy(c.point).sub(joint.pivot)).dot(n):joint.axis.dot(n);
 const sP=joint.qd*jn,sB=hand.velocity.dot(n);
 const closing=sP-sB;
 if(closing<=1e-6)return {closing:0,pushed:false,blocked:false,depth:c.depth};
 // Pressed across the way the part can move (or on the hinge line): the surface cannot be pushed along n.
 if(Math.abs(jn)<(hinge?MIN_JN.hinge/joint.unit:MIN_JN.slider))return {closing:closing*joint.unit,pushed:false,blocked:true,depth:c.depth};
 const cap=joint.qSpeed(maxSpeed??(hinge?PUSH_MAX_SPEED.hinge:PUSH_MAX_SPEED.slider));
 const want=clamp(sB/jn,-cap,cap);
 let qd=want;
 const next=joint.q+qd*dt;
 if(next<joint.lo)qd=(joint.lo-joint.q)/dt;else if(next>joint.hi)qd=(joint.hi-joint.q)/dt;
 joint.qd=qd;joint.interrupt();
 // Gave way by less than half of what the hand asked: something is holding it.
 const blocked=Math.abs(want)>1e-6&&Math.abs(qd)<Math.abs(want)*.5;
 return {closing:closing*joint.unit,pushed:true,blocked,depth:c.depth};
}

// A hand's velocity from its pose history: the raw difference, smoothed (the tracked pose is noisy and a
// frame is long), and thrown away across a jump (tracking loss, a snap turn, the start of a session).
export class HandMotion{
 constructor({rate=28,maxSpeed=8}={}){this.rate=rate;this.maxSpeed=maxSpeed;this.prev=new THREE.Vector3();this.has=false;this.velocity=new THREE.Vector3();this._raw=new THREE.Vector3()}
 reset(){this.has=false;this.velocity.set(0,0,0)}
 update(position,dt){
  if(!this.has||!(dt>0)){this.prev.copy(position);this.has=true;this.velocity.set(0,0,0);return this.velocity}
  const raw=this._raw.copy(position).sub(this.prev).divideScalar(dt);
  if(raw.length()>this.maxSpeed)this.velocity.set(0,0,0);
  else this.velocity.lerp(raw,smoothing(this.rate,dt));
  this.prev.copy(position);
  return this.velocity;
 }
}
