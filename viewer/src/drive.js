// Grab drives. After Alyx-Physics src/physics/grabber.js `createDrive`: the grip point is stored on
// the part at the moment of the grab; every physics step the part's real position of that point is the
// anchor, wherever the hand or the mouse wants it is the target, and the joint pulls the anchor toward
// the target by its own `drag` (so the part only ever moves as its joint allows).
//
// All positions here are in the joint's frame (the export's coordinates), not world space.
import * as THREE from 'three';
import {HINGE} from './joint.js';

export class Drive{
 // `target(anchor, out, dt)` fills `out` with where the grip wants to be and returns false to skip a step
 // (a mouse ray running parallel to its drag plane, a controller that lost tracking).
 constructor(joint,{anchor,target,tune}){
  this.joint=joint;this.tune=tune;this.target=target;
  // The anchor on the part at rest, so it follows the part wherever it has turned.
  this.gripOnObject=joint.toRest(anchor,joint.q,new THREE.Vector3());
  this.anchorNow=new THREE.Vector3();this.targetNow=new THREE.Vector3();
  joint.grab();
 }
 get anchor(){return this.joint.pointAt(this.gripOnObject,this.joint.q,this.anchorNow)}
 step(dt){
  const anchor=this.anchor;
  if(!this.target(anchor,this.targetNow,dt))return;
  this.joint.drag(anchor,this.targetNow,dt,this.tune);
 }
 // The part keeps its speed: the grab simply lets go and friction does the rest.
 release(){this.joint.release()}
}

// Where a mouse ray (origin/dir in the joint's frame) lands on the plane the anchor drags in: for a hinge,
// the plane through the anchor perpendicular to the hinge axis (the circle the anchor swings on lies in
// it); for a slider, the plane through the anchor that contains the slide axis and faces `eye` most.
export function dragPlaneTarget(joint,anchor,origin,dir,eye,out){
 const n=_n;
 if(joint.kind===HINGE)n.copy(joint.axis);
 else{
  n.copy(eye).sub(anchor);n.addScaledVector(joint.axis,-n.dot(joint.axis));
  if(n.lengthSq()<1e-8){n.crossVectors(joint.axis,Math.abs(joint.axis.y)<.9?_up:_x)}
  n.normalize();
 }
 const denom=n.dot(dir);
 // A ray skimming the plane would throw the target miles away.
 if(Math.abs(denom)<.04)return false;
 const t=n.dot(_d.copy(anchor).sub(origin))/denom;
 if(t<0||t>100)return false;
 out.copy(origin).addScaledVector(dir,t);return true;
}
const _n=new THREE.Vector3(),_d=new THREE.Vector3(),_up=new THREE.Vector3(0,1,0),_x=new THREE.Vector3(1,0,0);
