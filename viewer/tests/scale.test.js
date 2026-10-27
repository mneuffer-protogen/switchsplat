// metresPerUnit: the joint keeps the export's units and converts its metric tuning with `unit`. The same physical
// hand motion on the same physical door must give the same result whether the scene is in metres (unit 1) or in
// COLMAP-style units (the car: 0.586 m per unit, geometry 1/0.586 times bigger).
import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {JointedPart,HINGE,SLIDER,STEP} from '../src/joint.js';
import {Drive} from '../src/drive.js';
import {ContactSet,nearestPart,pickParts,contactDistance,GRAB_RADIUS,PICK_RADIUS} from '../src/contact.js';
import {pushJoint,contactSetContact,HAND_RADIUS} from '../src/push.js';
import {computeSpawn,viewInRootFrame,floorHeight,toRootFrame} from '../src/spawn.js';
import {distanceToProxy} from '../src/obb.js';
import {DEG} from '../src/math.js';

const S=.586,SCALES=[1,S];
const V=(x,y,z,k=1)=>new THREE.Vector3(x/k,y/k,z/k);// a point given in metres, as export units for scale k

// A slab "scan" of a 0.9 x 2 m door leaf in metres, spacing ~4.5 x 5 cm, as export units.
function slab(k){const p=[];for(let ix=0;ix<20;ix++)for(let iy=0;iy<40;iy++)p.push((.05+ix*.045)/k,(iy*.05)/k,0);return new Float32Array(p)}
function door(k,{contacts=true,...opts}={}){
 const object=new THREE.Object3D();
 const joint=new JointedPart({object,unit:k,kind:HINGE,axis:V(0,1,0),pivot:V(0,0,0),min:0,max:100*DEG,open:100*DEG,friction:1.5,frictionTorque:5,mass:25,latch:0,
  box:{center:V(.45,1,0,k),size:V(.9,2,.04,k)},...opts});
 if(contacts)joint.contacts=new ContactSet(slab(k));
 return joint;
}
function drawer(k,opts={}){
 const object=new THREE.Object3D();
 const joint=new JointedPart({object,unit:k,kind:SLIDER,axis:V(0,0,1),min:0,max:.43/k,open:.43/k,friction:3,frictionTorque:2,mass:2.5,latch:0,maxSpeed:2,
  box:{center:V(0,0,0),size:V(.5,.2,.4,k)},...opts});
 joint.contacts=new ContactSet(new Float32Array([...Array(10).keys()].flatMap(i=>[(-.2+i*.045)/k,0,-.2/k])));
 return joint;
}

test('generalised mass is the same physical inertia at any scale; a slider carries unit^2',()=>{
 const [a,b]=SCALES.map(k=>door(k));
 assert.ok(Math.abs(a.mass-b.mass)/a.mass<1e-9,`${a.mass} vs ${b.mass}`);
 const [da,db]=SCALES.map(k=>drawer(k));
 assert.ok(Math.abs(db.mass-da.mass*S*S)<1e-9);
 const resp=SCALES.map(k=>{const j=door(k);j.step(STEP);return j.responseFor(j.openQ)});
 assert.ok(Math.abs(resp[0]-resp[1])<1e-9,'a click animation takes as long');
 const r2=SCALES.map(k=>drawer(k).responseFor(.43/k));assert.ok(Math.abs(r2[0]-r2[1])<1e-9);
});

test('the same physical drag gives the same door angle at metresPerUnit 1 and 0.586',()=>{
 const qs=SCALES.map(k=>{
  const joint=door(k),start=V(.85,1,0,k);let t=0;
  const drive=new Drive(joint,{anchor:start,target:(a,out)=>{out.copy(start).applyAxisAngle(new THREE.Vector3(0,1,0),Math.min(45*DEG,t*45*DEG));return true}});
  for(let i=0;i<Math.round(2/STEP);i++){t=i*STEP;drive.step(STEP);joint.step(STEP)}
  const q=joint.q;drive.release();for(let i=0;i<Math.round(.5/STEP);i++)joint.step(STEP);
  return [q,joint.q];
 });
 assert.ok(Math.abs(qs[0][0]-qs[1][0])<.05*DEG,`${qs[0][0]/DEG} vs ${qs[1][0]/DEG} deg`);
 assert.ok(Math.abs(qs[0][1]-qs[1][1])<.05*DEG,'and the coast after release');
 assert.ok(Math.abs(qs[0][0]-45*DEG)<6*DEG,'and it really pulled the door');
});

test('the same physical drag gives the same drawer travel in metres',()=>{
 const out=SCALES.map(k=>{
  const joint=drawer(k),start=V(.1,0,.2,k);let z=0;
  const drive=new Drive(joint,{anchor:start,target:(a,o)=>{o.set(.1/k,0,(.2+z)/k);return true}});
  for(let i=0;i<Math.round(1.5/STEP);i++){z=Math.min(.2,i*STEP*.4);drive.step(STEP);joint.step(STEP)}
  return joint.q*k;
 });
 assert.ok(Math.abs(out[0]-out[1])<1e-4,`${out} m`);
 assert.ok(Math.abs(out[0]-.2)<.03);
});

test('the same physical push gives the same angle (door) and travel (drawer); speed caps are metric',()=>{
 const door_=SCALES.map(k=>{
  const joint=door(k),pos=V(.8,1,.07,k),vel=V(0,0,-.5,k);
  for(let i=0;i<Math.round(.6/STEP);i++){pushJoint(joint,{position:pos,velocity:vel,radius:HAND_RADIUS},STEP);joint.step(STEP);pos.addScaledVector(vel,STEP)}
  return [joint.q,joint.qd];
 });
 assert.ok(door_[0][0]>10*DEG&&Math.abs(door_[0][0]-door_[1][0])<.05*DEG,`${door_.map(d=>d[0]/DEG)} deg`);
 assert.ok(Math.abs(door_[0][1]-door_[1][1])<1e-3);
 const slide=SCALES.map(k=>{
  const joint=drawer(k),pos=V(0,0,-.3,k),vel=V(0,0,.5,k);let peak=0,touched=0;
  for(let i=0;i<Math.round(.8/STEP);i++){if(pushJoint(joint,{position:pos,velocity:vel,radius:HAND_RADIUS},STEP))touched++;joint.step(STEP);pos.addScaledVector(vel,STEP);peak=Math.max(peak,Math.abs(joint.qd*k))}
  return {m:joint.q*k,peak,touched};
 });
 assert.ok(slide[0].touched>0&&slide[0].m>.05,`pushed ${slide[0].m} m`);
 assert.ok(Math.abs(slide[0].m-slide[1].m)<1e-4,`${slide.map(s=>s.m)} m`);
 // A violent shove goes no faster than the cap in metres per second, in either scale.
 for(const k of SCALES){
  const j=drawer(k,{friction:0,maxSpeed:0});
  pushJoint(j,{position:V(0,0,-.26,k),velocity:V(0,0,20,k),radius:HAND_RADIUS},STEP);
  assert.ok(j.qd*k<=3+1e-9&&j.qd*k>2.9,`scale ${k}: ${j.qd*k} m/s`);
 }
 // Sub-cap: the speed limit of a pushed (not held) part is in m/s too.
 const capped=SCALES.map(k=>{const j=drawer(k);j.qd=10/k;j.step(STEP);return Math.abs(j.qd*k)});
 assert.ok(Math.abs(capped[0]-capped[1])<1e-6&&capped[0]<=2+1e-6,`${capped}`);
});

test('contact radii are real sizes: 7 cm grab, 4 cm pick, 4.5 + 2.5 cm push, whatever the unit',()=>{
 for(const k of SCALES){
  const joint=door(k),part={id:'d',joint,contacts:joint.contacts,interactive:true};
  // The slab lies in z = 0; a hand 6 cm off its face grabs, 8 cm does not.
  assert.ok(Math.abs(contactDistance(part,V(.5,1,.06,k),GRAB_RADIUS)-.06)<.003,`scale ${k}`);
  assert.equal(contactDistance(part,V(.5,1,.085,k),GRAB_RADIUS),Infinity);
  assert.ok(nearestPart([part],V(.5,1,.06,k)));assert.equal(nearestPart([part],V(.5,1,.085,k)),null);
  // Push: touching at 6.5 cm, not at 7.5 cm.
  const out={point:new THREE.Vector3(),normal:new THREE.Vector3(),depth:0};
  assert.ok(contactSetContact(joint,joint.contacts,V(.5,1,.065,k),HAND_RADIUS/k,out));
  assert.equal(contactSetContact(joint,joint.contacts,V(.5,1,.075,k),HAND_RADIUS/k,out),null);
  assert.ok(pushJoint(joint,{position:V(.5,1,.065,k),velocity:new THREE.Vector3(),radius:HAND_RADIUS},STEP));
  assert.equal(pushJoint(joint,{position:V(.5,1,.075,k),velocity:new THREE.Vector3(),radius:HAND_RADIUS},STEP),null);
  // Pick: a ray 3 cm beside a splat hits (the lattice is 4.5 cm wide), the hit distance is in metres.
  const hit=pickParts([part],V(.5,1.01,3,k),new THREE.Vector3(0,0,-1),PICK_RADIUS);
  assert.ok(hit&&Math.abs(hit.distance-(3-PICK_RADIUS))<.03,`distance ${hit?.distance} m at scale ${k}`);
  assert.equal(pickParts([part],V(1.1,1,3,k),new THREE.Vector3(0,0,-1),PICK_RADIUS),null);
  assert.equal(pickParts([part],V(.5,1,3,k),new THREE.Vector3(0,0,-1),PICK_RADIUS,2),null,'beyond maxDist metres');
 }
});

test('latch, slam and jiggle sizes are metric for a slider',()=>{
 const j=drawer(S,{latch:.02/S});
 assert.ok(Math.abs(j.slamSpeed*S-.5)<1e-9);assert.ok(Math.abs(j.tolerance*S-1e-4)<1e-12);
 j.jiggle();let peak=0;for(let i=0;i<30;i++){j.step(STEP);peak=Math.max(peak,Math.abs(j.jiggleOffset()))}
 assert.ok(peak*S<=.005+1e-9&&peak*S>.001,`rattle ${peak*S} m`);
});

test('spawn with a metric scale: floor at y = 0, the user at the view, the car at its real size',()=>{
 const up=[0.119,-0.988,-0.102],q=new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(...up).normalize(),new THREE.Vector3(0,1,0));
 const floor=[-0.908,2.301,0.869],view={position:[1.92,1.067,5.554],target:[0.041,2.582,0.514]};
 const spawn=computeSpawn({rootQuaternion:q,rootScale:S,floor,view});
 const toXr=p=>new THREE.Vector3(...p).applyQuaternion(q).multiplyScalar(S).applyQuaternion(spawn.quaternion).add(spawn.position);
 assert.ok(Math.abs(toXr(floor).y)<1e-9,'floor at y = 0');
 const eye=toXr(view.position),feet=eye.clone().setY(0);
 assert.ok(Math.hypot(eye.x,eye.z)<1e-9,'standing at the view');
 // The eye is as far above the floor as the view was (in metres): the up-component of (view - floor) times the scale.
 const upv=new THREE.Vector3(...up).normalize(),h=-(new THREE.Vector3(...view.position).sub(new THREE.Vector3(...floor)).dot(upv))*-1;
 assert.ok(Math.abs(eye.y-Math.abs(h)*S)<1e-9&&feet.y===0,`eye ${eye.y} m`);
 const t=toXr(view.target);assert.ok(Math.abs(t.x)<1e-9&&t.z<0,'facing the target');
 // A 1.7 export-unit tall thing above the floor is 1.0 m tall.
 const roof=toXr(new THREE.Vector3(...floor).addScaledVector(upv,1.7).toArray());assert.ok(Math.abs(roof.y-1.7*S)<1e-9);
 const v=viewInRootFrame(view,q,S);assert.ok(Math.abs(v.position.length()-new THREE.Vector3(...view.position).length()*S)<1e-9);
 assert.ok(Math.abs(floorHeight(floor,q,S)-toRootFrame(floor,q,S).y)<1e-12);
 // Default scale 1 as before.
 assert.ok(computeSpawn({rootQuaternion:q,floor,view}).position.distanceTo(computeSpawn({rootQuaternion:q,rootScale:1,floor,view}).position)<1e-12);
});

test('box distance (fallback grab) reads the proxy in metres through a scaled root',()=>{
 const root=new THREE.Group();root.scale.setScalar(S);
 const proxy=new THREE.Mesh(new THREE.BoxGeometry(1,1,1));proxy.scale.set(1/S,2/S,.04/S);proxy.position.set(0,1/S,0);root.add(proxy);
 root.updateMatrixWorld(true);
 // A 1 m x 2 m x 4 cm box centred 1 m up: 10 cm off its face is 10 cm - 2 cm = 8 cm away.
 assert.ok(Math.abs(distanceToProxy(proxy,new THREE.Vector3(0,1,.1))-.08)<1e-9);
 assert.equal(distanceToProxy(proxy,new THREE.Vector3(0,1,0)),0);
});
