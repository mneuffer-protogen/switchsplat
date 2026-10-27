// VR locomotion in the floor plane: left thumbstick moves, right thumbstick snap-turns. The controllers and the
// head live in the XR reference space, so the user is moved by moving the `world` group (the scene) the other
// way, and turned by turning the scene about the user's feet.
import * as THREE from 'three';

export const MOVE_SPEED=1.6,SNAP_DEG=30,DEADZONE=.2,SNAP_ON=.7,SNAP_OFF=.35;
const Y=new THREE.Vector3(0,1,0);

// Dead-zoned stick: the vector scaled so the dead zone's edge is 0 and the rim is 1.
export function stickVector(x,y,deadzone=DEADZONE,out=new THREE.Vector2()){
 const m=Math.hypot(x,y);
 if(m<deadzone)return out.set(0,0);
 const k=Math.min(1,(m-deadzone)/(1-deadzone))/m;
 return out.set(x*k,y*k);
}

// Move the user by `delta` (a world-space vector in the floor plane) by moving the world group the other way.
export function moveUser(world,delta){world.position.x-=delta.x;world.position.z-=delta.z;return world}

// Turn the user by `angle` radians to the right (clockwise seen from above) about `pivot` (the head, in scene
// coordinates): the scene turns the other way, counter-clockwise, about that point.
export function turnUser(world,pivot,angle){
 const turn=new THREE.Quaternion().setFromAxisAngle(Y,angle);
 world.position.sub(_p.set(pivot.x,world.position.y,pivot.z)).applyQuaternion(turn).add(_p);
 world.quaternion.premultiply(turn);
 return world;
}
const _p=new THREE.Vector3();

// The direction the head faces and its right, flattened to the floor plane.
export function headingFrame(forward,out={forward:new THREE.Vector3(),right:new THREE.Vector3()}){
 out.forward.set(forward.x,0,forward.z);
 if(out.forward.lengthSq()<1e-8)out.forward.set(0,0,-1);else out.forward.normalize();
 out.right.set(-out.forward.z,0,out.forward.x);
 return out;
}

// The stick-to-motion mapping: `stick` = (x right, y down) as a gamepad reports it.
export function stickToDelta(stick,frame,speed,dt,out=new THREE.Vector3()){
 return out.set(0,0,0).addScaledVector(frame.right,stick.x*speed*dt).addScaledVector(frame.forward,-stick.y*speed*dt);
}

// Reads the two thumbsticks each frame. `onTurn` is told when the view was turned (hand velocities across a turn
// are meaningless). Quest controllers report the thumbstick as axes[2], axes[3]; a pad with only two axes as 0, 1.
export function createLocomotion({renderer,world,onTurn=()=>{},speed=MOVE_SPEED,snap=SNAP_DEG}){
 const head=new THREE.Vector3(),dir=new THREE.Vector3(),frame={forward:new THREE.Vector3(),right:new THREE.Vector3()},stick=new THREE.Vector2(),delta=new THREE.Vector3();
 let armed=true;
 const axesOf=src=>{const a=src.gamepad?.axes;if(!a||a.length<2)return null;return a.length>=4?[a[2],a[3]]:[a[0],a[1]]};
 return {
  enabled:true,
  update(dt,{blocked=false}={}){
   const session=renderer.xr.getSession?.();if(!session||!this.enabled||blocked)return;
   const cam=renderer.xr.getCamera();cam.getWorldPosition(head);cam.getWorldDirection(dir);
   headingFrame(dir,frame);
   for(const src of session.inputSources){
    const a=axesOf(src);if(!a)continue;
    if(src.handedness==='left'){
     stickVector(a[0],a[1],DEADZONE,stick);
     if(stick.lengthSq()>0){moveUser(world,stickToDelta(stick,frame,speed,dt,delta))}
    }else if(src.handedness==='right'){
     const x=a[0];
     if(armed&&Math.abs(x)>SNAP_ON){armed=false;turnUser(world,head,Math.sign(x)*snap*Math.PI/180);onTurn()}
     else if(!armed&&Math.abs(x)<SNAP_OFF)armed=true;
    }
   }
  },
 };
}
