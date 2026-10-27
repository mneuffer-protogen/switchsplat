// Small math helpers, copied from Alyx-Physics src/physics/math.js (the Pose type and scalar helpers
// the joint and grab code use; Unity-coordinate conversion dropped).
import * as THREE from 'three';

export const DEG=Math.PI/180;
export const clamp=(v,lo,hi)=>v<lo?lo:v>hi?hi:v;
export const clamp01=v=>clamp(v,0,1);
export const lerp=(a,b,t)=>a+(b-a)*t;
export const inverseLerp=(a,b,v)=>a===b?0:clamp01((v-a)/(b-a));

export class Pose{
 constructor(position=new THREE.Vector3(),quaternion=new THREE.Quaternion()){this.position=position;this.quaternion=quaternion}
 set(p,q){this.position.copy(p);this.quaternion.copy(q);return this}
 copy(o){return this.set(o.position,o.quaternion)}
 clone(){return new Pose(this.position.clone(),this.quaternion.clone())}
 // A point given in this pose's local space, as a point in the parent space.
 apply(local,out=new THREE.Vector3()){return out.copy(local).applyQuaternion(this.quaternion).add(this.position)}
 // A parent-space point expressed in this pose's local space.
 localize(world,out=new THREE.Vector3()){return out.copy(world).sub(this.position).applyQuaternion(_q.copy(this.quaternion).invert())}
 static of(object,out=new Pose()){object.updateWorldMatrix(true,false);object.getWorldPosition(out.position);object.getWorldQuaternion(out.quaternion);return out}
}
const _q=new THREE.Quaternion();
