// Pure placement maths for real scenes (articulation-spec.md, Addendum v1.1). No DOM, so node tests can import it.
//
// The scene root is turned so `up` maps to +Y (rootQuaternion). `floor` is any point on the ground plane and
// `view` a camera pose, both in export coordinates. In VR the reference space is local-floor: its origin is
// on the floor where the user stands, and -Z is "ahead". The `world` group that holds the root is moved so
//   - the floor plane is at y = 0,
//   - view.position, dropped straight down onto the floor, is at the origin (the user stands there),
//   - the horizontal direction from view.position to view.target is -Z (the user faces it).
import * as THREE from 'three';

const Y=new THREE.Vector3(0,1,0);

// A point of the export as it stands in the root's frame (scene up on +Y).
export const toRootFrame=(p,rootQuaternion)=>new THREE.Vector3(...p).applyQuaternion(rootQuaternion);

// The yaw (radians about +Y) that turns the horizontal direction (dx, dz) onto -Z.
export const yawToFace=(dx,dz)=>Math.atan2(dx,-dz);

// {position: Vector3, quaternion: Quaternion} for the world group, or null when `floor` or `view` is missing.
export function computeSpawn({rootQuaternion,floor,view}){
 if(!floor||!view?.position||!view?.target)return null;
 const f=toRootFrame(floor,rootQuaternion),eye=toRootFrame(view.position,rootQuaternion),target=toRootFrame(view.target,rootQuaternion);
 const dx=target.x-eye.x,dz=target.z-eye.z;
 // Looking straight up or down gives no heading: keep the scene's own -Z.
 const yaw=Math.hypot(dx,dz)>1e-6?yawToFace(dx,dz):0;
 const quaternion=new THREE.Quaternion().setFromAxisAngle(Y,yaw);
 const stand=new THREE.Vector3(eye.x,f.y,eye.z);
 const position=stand.applyQuaternion(quaternion).negate();
 return {position,quaternion,yaw};
}

// The desktop camera: view.position / view.target in the root's frame, or null.
export function viewInRootFrame(view,rootQuaternion){
 if(!view?.position||!view?.target)return null;
 return {position:toRootFrame(view.position,rootQuaternion),target:toRootFrame(view.target,rootQuaternion)};
}

// Height of the floor plane in the root's frame (for a scene with `floor` but no `view`).
export const floorHeight=(floor,rootQuaternion)=>floor?toRootFrame(floor,rootQuaternion).y:null;
