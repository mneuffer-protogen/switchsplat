// Spawn placement from scene.json view/floor/up, VR locomotion maths, the persisted lock, and the HUD's merged label.
import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {computeSpawn,viewInRootFrame,floorHeight,yawToFace} from '../src/spawn.js';
import {moveUser,turnUser,stickVector,stickToDelta,headingFrame} from '../src/locomotion.js';
import {createPersistence} from '../src/persist.js';
import {JointedPart,HINGE} from '../src/joint.js';
import {mergedLabel} from '../src/hud.js';

const V=(x,y,z)=>new THREE.Vector3(x,y,z);
const rootFor=up=>new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(...up).normalize(),V(0,1,0));
// Where a point of the export ends up in the XR reference space, given the root rotation and the spawn.
function toXr(p,rootQ,spawn){
 return V(...p).applyQuaternion(rootQ).applyQuaternion(spawn.quaternion).add(spawn.position);
}

test('spawn, y-up scene: the user stands at the view, on the floor, facing the target',()=>{
 const q=rootFor([0,1,0]);
 const spawn=computeSpawn({rootQuaternion:q,floor:[0,0.3,0],view:{position:[2,1.7,3],target:[2,1.2,-1]}});
 const feet=toXr([2,.3,3],q,spawn);assert.ok(feet.length()<1e-9,`feet at ${feet.toArray()}`);
 const floorPoint=toXr([-5,.3,7],q,spawn);assert.ok(Math.abs(floorPoint.y)<1e-9,'the floor plane is y = 0');
 // Looking along -Z already: no turn.
 assert.ok(Math.abs(spawn.yaw)<1e-9);
 const ahead=toXr([2,.3,-1],q,spawn);assert.ok(Math.abs(ahead.x)<1e-9&&ahead.z<0,'the target is straight ahead');
 // The eye height above the floor is kept (the headset adds the user's own height).
 assert.ok(Math.abs(toXr([2,1.7,3],q,spawn).y-1.4)<1e-9);
});

test('spawn: facing +X, -X and +Z in the export turns them onto -Z',()=>{
 const q=rootFor([0,1,0]);
 for(const [target,expect] of [[[5,0,0],V(0,0,-1)],[[-5,0,0],V(0,0,-1)],[[0,0,5],V(0,0,-1)],[[3,0,-3],V(0,0,-1)]]){
  const spawn=computeSpawn({rootQuaternion:q,floor:[0,0,0],view:{position:[0,1.5,0],target}});
  const dir=V(...target).applyQuaternion(q).applyQuaternion(spawn.quaternion).add(spawn.position);
  dir.y=0;dir.normalize();
  assert.ok(dir.distanceTo(V(0,0,-1))<1e-9,`target ${target} -> ${dir.toArray()}`);
 }
 assert.ok(Math.abs(yawToFace(1,0)-Math.PI/2)<1e-12);assert.ok(Math.abs(Math.abs(yawToFace(0,1))-Math.PI)<1e-12);
});

test('spawn, COLMAP scene with up = [0,-1,0]: the root turns it upright first, floor and heading follow',()=>{
 const up=[0,-1,0],q=rootFor(up);
 // In export coordinates "up" is -Y, so the floor sits at some y and the eye is at a smaller y (higher up).
 const floor=[0.5,2,0.2],view={position:[1,.4,-2],target:[1.5,.5,1]};
 const spawn=computeSpawn({rootQuaternion:q,floor,view});
 const feet=toXr([1,2,-2],q,spawn);// the view position dropped to the floor along "up"
 assert.ok(feet.length()<1e-9,`feet at ${feet.toArray()}`);
 assert.ok(Math.abs(toXr(floor,q,spawn).y)<1e-9);
 const eye=toXr(view.position,q,spawn);assert.ok(eye.y>1.5&&eye.y<1.7,`eye ${eye.y} above the floor`);
 // Whatever the rotation, the target is straight ahead horizontally.
 const t=toXr(view.target,q,spawn);assert.ok(Math.abs(t.x)<1e-9&&t.z<0,`target ${t.toArray()}`);
});

test('spawn: tilted up vector; and a missing view or floor gives no spawn',()=>{
 const up=[0.1,-0.97,0.2],q=rootFor(up);
 const f=V(1,2,3),eyeW=V(0,0,0);
 // A floor point and the eye 1.5 m above it along -up.
 const upv=new THREE.Vector3(...up).normalize();eyeW.copy(f).addScaledVector(upv,1.5);
 const spawn=computeSpawn({rootQuaternion:q,floor:f.toArray(),view:{position:eyeW.toArray(),target:[0,0,0]}});
 const eye=toXr(eyeW.toArray(),q,spawn);
 assert.ok(Math.abs(eye.y-1.5)<1e-9&&Math.hypot(eye.x,eye.z)<1e-9,`eye ${eye.toArray()}`);
 assert.equal(computeSpawn({rootQuaternion:q,floor:null,view:{position:[0,0,0],target:[1,0,0]}}),null);
 assert.equal(computeSpawn({rootQuaternion:q,floor:[0,0,0],view:null}),null);
 assert.ok(Math.abs(floorHeight([0,3,0],rootFor([0,1,0]))-3)<1e-9);assert.equal(floorHeight(null,q),null);
 // Looking straight down gives no heading but still stands on the floor.
 const down=computeSpawn({rootQuaternion:rootFor([0,1,0]),floor:[0,0,0],view:{position:[1,2,1],target:[1,0,1]}});
 assert.equal(down.yaw,0);
});

test('desktop view is expressed in the root frame',()=>{
 const q=rootFor([0,-1,0]);
 const v=viewInRootFrame({position:[1,2,3],target:[0,0,0]},q);
 assert.ok(v.position.distanceTo(V(1,2,3).applyQuaternion(q))<1e-12);
 assert.equal(viewInRootFrame(null,q),null);assert.equal(viewInRootFrame({position:[0,0,0]},q),null);
});

test('locomotion: stick dead zone, move along the head heading, snap turn about the feet',()=>{
 assert.equal(stickVector(.1,.1).length(),0);
 const full=stickVector(0,-1);assert.ok(Math.abs(full.y+1)<1e-9);
 const half=stickVector(0,-.6);assert.ok(half.length()>0&&half.length()<1);
 const world=new THREE.Group();
 // Facing -Z, stick pushed forward: the user goes -Z, so the world comes toward +Z.
 const frame=headingFrame(V(0,.3,-1));
 moveUser(world,stickToDelta(V(0,-1,0),frame,1.6,.5));
 assert.ok(world.position.distanceTo(V(0,0,.8))<1e-9,`world at ${world.position.toArray()}`);
 // Facing +X, stick right: the user's right is +Z.
 const w2=new THREE.Group();const f2=headingFrame(V(1,0,0));
 assert.ok(f2.right.distanceTo(V(0,0,1))<1e-9);
 moveUser(w2,stickToDelta(V(1,0,0),f2,2,.5));assert.ok(w2.position.distanceTo(V(0,0,-1))<1e-9);
 // Looking straight up: heading falls back to -Z, not NaN.
 assert.ok(headingFrame(V(0,1,0)).forward.distanceTo(V(0,0,-1))<1e-9);
 // A snap turn to the right by 30 degrees about the user's feet: what is under the feet in the scene stays under
 // them, and the scene turns counter-clockwise (its -Z axis, straight ahead, now points to the user's left).
 const w=new THREE.Group();w.position.set(1,0,-3);
 const feet=V(.5,0,.5),underBefore=feet.clone().sub(w.position);
 turnUser(w,feet,30*Math.PI/180);
 const underAfter=feet.clone().sub(w.position).applyQuaternion(w.quaternion.clone().invert());
 assert.ok(underAfter.distanceTo(underBefore)<1e-9,'the same scene point stays under the feet');
 const ahead=V(0,0,-1).applyQuaternion(w.quaternion);
 assert.ok(ahead.x<0&&Math.abs(Math.atan2(-ahead.x,-ahead.z)-30*Math.PI/180)<1e-9);
});

test('persistence keeps position and lock; the lock is only stored when it differs from scene.json',()=>{
 const mem=new Map(),storage={getItem:k=>mem.get(k)??null,setItem:(k,v)=>mem.set(k,v),key:i=>[...mem.keys()][i],removeItem:k=>mem.delete(k),get length(){return mem.size}};
 const make=(opts={})=>new JointedPart({kind:HINGE,axis:V(0,1,0),min:0,max:1.5,open:1.5,latch:0,...opts});
 let p=createPersistence('scene-a',{delay:0,storage:()=>storage});
 let j=make();p.watch('slot01',j);
 j.setPosition(.7);j.emit('value',j.value);j.setLocked(true);p.flush();
 assert.deepEqual(JSON.parse(mem.get('switchsplat:v1:scene-a:slot01')),{q:.7,locked:true});
 p=createPersistence('scene-a',{delay:0,storage:()=>storage});
 j=make();p.watch('slot01',j);
 assert.ok(Math.abs(j.q-.7)<1e-12&&j.locked,'position and lock restored');
 // A joint that scene.json already locks does not save a redundant lock.
 const j2=make({locked:true});p.watch('slot02',j2);j2.setPosition(.2);j2.emit('value',j2.value);p.flush();
 assert.deepEqual(JSON.parse(mem.get('switchsplat:v1:scene-a:slot02')),{q:.2});
 // An old entry (position only) still loads, and the lock keeps scene.json's value.
 mem.set('switchsplat:v1:scene-a:old',JSON.stringify({q:.4}));
 const j3=make({locked:true});createPersistence('scene-a',{storage:()=>storage}).watch('old',j3);
 assert.ok(Math.abs(j3.q-.4)<1e-12&&j3.locked);
 assert.equal(p.restore('slot01'),.7);
});

test('a part merged from several slots says so',()=>{
 assert.equal(mergedLabel({merged:['slot02','slot03']}),'merged from slot02, slot03');
 assert.equal(mergedLabel({merged:null}),'');assert.equal(mergedLabel({}),'');
});
