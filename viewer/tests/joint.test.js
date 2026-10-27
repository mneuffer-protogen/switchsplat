// The joint, ported from Alyx-Physics tests/physics-joints.test.js: no physics world, just the one
// coordinate stepped at a fixed rate, driven by a grab or by animateTo.
import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {JointedPart,HINGE,SLIDER,STEP,FixedStepper} from '../src/joint.js';
import {Drive,dragPlaneTarget} from '../src/drive.js';
import {DEG} from '../src/math.js';

const V=(x,y,z)=>new THREE.Vector3(x,y,z);
// A 0.9 m door leaf in the XY plane, hinged on its x=0 edge about +Y; open (+q) swings its free edge toward -Z.
function door(opts={}){
 const object=new THREE.Object3D();
 const joint=new JointedPart({object,kind:HINGE,axis:V(0,1,0),pivot:V(0,0,0),min:0,max:100*DEG,open:100*DEG,friction:1.5,mass:25,
  box:{center:V(.45,1,0),size:V(.9,2,.04)},...opts});
 return {object,joint};
}
function drawer(opts={}){
 const object=new THREE.Object3D();
 const joint=new JointedPart({object,kind:SLIDER,axis:V(0,0,1),min:0,max:.43,open:.43,friction:3,mass:2.5,...opts});
 return {object,joint};
}
// Steps like the app's loop: the grab (if any) pulls, then the joint steps.
const run=(seconds,joint,each,drive)=>{const n=Math.round(seconds/STEP);for(let i=0;i<n;i++){each?.(i*STEP,i);drive?.step(STEP);joint.step(STEP)}};
const Y=V(0,1,0);

test('generalised mass: a door about its hinge edge, a drawer is its mass',()=>{
 const {joint}=door({mass:25});
 // m/12 (w^2 + t^2) about the centre + m d^2 for the 0.45 m lever.
 const expected=25/12*(.9*.9+.04*.04)+25*.45*.45;
 assert.ok(Math.abs(joint.mass-expected)<1e-9,`I = ${joint.mass}`);
 assert.equal(drawer({mass:2.5}).joint.mass,2.5);
});

test('applyPose writes the hinge turn about the pivot line from the rest pose',()=>{
 const object=new THREE.Object3D();object.position.set(1,0,0);
 const joint=new JointedPart({object,kind:HINGE,axis:V(0,1,0),pivot:V(0,0,0),max:Math.PI,open:Math.PI});
 joint.setPosition(Math.PI/2);
 // A quarter turn about +Y maps +X to -Z.
 assert.ok(object.position.distanceTo(V(0,0,-1))<1e-9);
 const p=joint.pointAt(V(1,0,0),joint.q);assert.ok(p.distanceTo(V(0,0,-1))<1e-9);
 assert.ok(joint.toRest(p,joint.q).distanceTo(V(1,0,0))<1e-9,'toRest inverts pointAt');
 const {object:o2,joint:s}=drawer();s.setPosition(.2);assert.ok(Math.abs(o2.position.z-.2)<1e-9);
});

test('dragging the free edge of a door opens it by the angle the hand moved',()=>{
 const {joint}=door();
 const start=V(.8,1,0);
 // The hand sweeps the grip point on a 40 degree arc about the hinge at 40 deg/s.
 let t=0;
 const drive=new Drive(joint,{anchor:start,target:(a,out)=>{out.copy(start).applyAxisAngle(Y,Math.min(40*DEG,t*40*DEG));return true}});
 run(1.6,joint,tt=>{t=tt},drive);
 assert.ok(Math.abs(joint.q-40*DEG)<3*DEG,`door at ${(joint.q/DEG).toFixed(1)} deg for a 40 deg pull`);
 drive.release();
});

test('a drawer follows a hand that grabs its front and pulls, and the end stop holds',()=>{
 const {joint}=drawer();
 const start=V(.2,.2,0);let z=0;
 const drive=new Drive(joint,{anchor:start,target:(a,out)=>{out.set(.2,.2,z);return true}});
 run(1.5,joint,t=>{z=Math.min(.2,t*.4)},drive);
 assert.ok(Math.abs(joint.q-.2)<.02,`drawer moved ${joint.q.toFixed(3)} m for a 0.2 m pull`);
 let worst=0;
 run(1.5,joint,t=>{z=.2+Math.min(.6,t);worst=Math.max(worst,joint.q)},drive);
 assert.ok(worst<=.43+1e-9&&joint.q<=.43+1e-9,`stays inside its travel (${worst})`);
 assert.ok(joint.q>.42,'and is pushed right up to the stop');
 drive.release();run(2,joint);
 assert.ok(Math.abs(joint.qd)<.01,'comes to rest after release');
});

test('a hinge never leaves its limits, even thrown at them',()=>{
 const {joint}=door({friction:0});
 joint.qd=8;run(1,joint);
 assert.ok(joint.q<=joint.hi+1e-9,`q=${joint.q/DEG}`);
 assert.ok(Math.abs(joint.q-100*DEG)<1e-6,'stops dead at the open stop');
 let limit=0;joint.on('limit',s=>limit=Math.max(limit,s));
 joint.qd=-8;run(.6,joint);
 assert.ok(joint.q>=-1e-9,`q=${joint.q/DEG}`);
 assert.ok(limit>1,`limit event at speed ${limit}`);
});

test('release coasts on the joint momentum, then friction stops it',()=>{
 const {joint}=door();
 const start=V(.8,1,0);let t=0;
 const drive=new Drive(joint,{anchor:start,target:(a,out)=>{out.copy(start).applyAxisAngle(Y,Math.min(60*DEG,t*90*DEG));return true}});
 run(.5,joint,tt=>{t=tt},drive);
 const q0=joint.q,qd0=joint.qd;
 assert.ok(qd0>.3,`swinging at ${qd0.toFixed(2)} rad/s when let go`);
 drive.release();run(.4,joint);
 assert.ok(joint.q>q0+5*DEG,`coasted ${((joint.q-q0)/DEG).toFixed(1)} deg after release`);
 run(8,joint);assert.ok(Math.abs(joint.qd)<1e-3,'friction brings it to rest');
});

test('latch engages near min and blocks until grabbed',()=>{
 const {joint}=door({latch:3*DEG,friction:1.5});
 let latched=0,unlatched=0;joint.on('latched',()=>latched++);joint.on('unlatched',()=>unlatched++);
 joint.setPosition(2*DEG);run(.05,joint);
 assert.ok(joint.latched&&latched===1,'a nearly shut door latches');
 assert.ok(joint.q<.02*DEG,'and is pulled shut');
 // Pushed (not held), the latch holds.
 for(let i=0;i<90;i++){joint.applyImpulse(30);joint.step(STEP)}
 assert.ok(joint.q<.05*DEG,`latched door turned ${(joint.q/DEG).toFixed(2)} deg`);
 // A grab releases it, and the same pull now opens it.
 const start=V(.8,1,0);let t=0;
 const drive=new Drive(joint,{anchor:start,target:(a,out)=>{out.copy(start).applyAxisAngle(Y,Math.min(30*DEG,t*60*DEG));return true}});
 run(1,joint,tt=>{t=tt},drive);
 assert.ok(!joint.latched&&unlatched===1,'grabbing unlatches');
 assert.ok(joint.q>20*DEG,`door opened to ${(joint.q/DEG).toFixed(1)} deg`);
 // Swung shut again after release, it latches.
 drive.release();joint.qd=-2;run(3,joint);
 assert.ok(joint.latched,'closing re-latches');
});

test('animateTo reaches the target without crossing the limits',()=>{
 const {joint}=door({latch:3*DEG});
 joint.step(STEP);assert.ok(joint.latched);
 let unlatched=0;joint.on('unlatched',()=>unlatched++);
 joint.animateTo(joint.openQ);
 assert.ok(!joint.latched&&unlatched===1,'animating open unlatches');
 let max=0,min=1,settled=0;joint.on('settled',()=>settled++);
 let prev=0,monotonic=true;
 run(1.2,joint,()=>{max=Math.max(max,joint.q);min=Math.min(min,joint.q);if(joint.q<prev-1e-9)monotonic=false;prev=joint.q});
 assert.ok(max<=joint.hi+1e-9&&min>=-1e-9,'within limits');
 assert.ok(monotonic,'smooth, critically damped: no overshoot or backing up');
 assert.equal(joint.q,joint.openQ);assert.equal(joint.qd,0);assert.ok(!joint.animating&&settled===1);
 // And back: it closes and latches only once it has arrived.
 joint.animateTo(joint.closedQ);
 let latchedWhileMoving=false;
 run(1.5,joint,()=>{if(joint.latched&&joint.animating)latchedWhileMoving=true});
 assert.ok(!latchedWhileMoving);assert.ok(Math.abs(joint.q)<1e-9);
 run(.05,joint);assert.ok(joint.latched,'latched again once shut');
});

test('animateTo is not a snap: it takes about its response time',()=>{
 const {joint}=door({latch:0});
 joint.animateTo(joint.openQ,{response:.5});
 let t=0;
 while(t<.1){joint.step(STEP);t+=STEP}
 const early=joint.q/joint.openQ;
 while(t<.25){joint.step(STEP);t+=STEP}
 assert.ok(early<.5,`only ${(early*100).toFixed(0)}% open after 0.1 s`);
 assert.ok(joint.q/joint.openQ<.9,'still arriving at 0.25 s');
 run(1.5,joint);assert.equal(joint.q,joint.openQ);
});

test('animateTo works for a slider and can be retargeted mid-flight',()=>{
 const {joint}=drawer();
 joint.animateTo(.43);run(.2,joint);
 const mid=joint.q;assert.ok(mid>0&&mid<.43);
 joint.animateTo(0);run(1.5,joint);
 assert.ok(Math.abs(joint.q)<1e-9&&joint.qd===0,'the new target takes over from the current speed');
});

test('a grab interrupts an animation and the part obeys the hand',()=>{
 const {joint}=door({latch:0});
 joint.animateTo(joint.openQ);run(.2,joint);
 const q=joint.q;assert.ok(joint.animating&&q>0);
 const start=joint.pointAt(V(.8,1,0));
 const drive=new Drive(joint,{anchor:start,target:(a,out)=>{out.copy(start);return true}});
 assert.ok(!joint.animating,'grab clears the animation');
 run(1.5,joint,null,drive);
 // Held where it was grabbed: it does not carry on to the open stop.
 assert.ok(Math.abs(joint.q-q)<8*DEG,`held near ${(q/DEG).toFixed(1)} deg, at ${(joint.q/DEG).toFixed(1)}`);
 drive.release();
});

test('fraction and the click toggle follow q over open',()=>{
 const {joint}=door({open:90*DEG});
 joint.setPosition(45*DEG);assert.ok(Math.abs(joint.fraction-.5)<1e-9);
 joint.setPosition(100*DEG);assert.equal(joint.fraction,1);
});

test('the fixed stepper takes whole steps and caps a hitch',()=>{
 const s=new FixedStepper();let n=0;
 s.run(1/60,()=>n++);assert.equal(n,1);
 n=0;s.run(2,()=>n++);assert.equal(n,4,'a 2 s hitch is 4 substeps, not 180');
 n=0;s.run(.004,()=>n++);assert.ok(n<=1);
});

test('drag plane: perpendicular to a hinge axis, facing the camera for a slider',()=>{
 const {joint}=door();const out=V(0,0,0);
 // A hinge about +Y: the horizontal plane through the anchor is hit below the eye.
 assert.ok(dragPlaneTarget(joint,V(.8,1,0),V(.8,2,3),V(0,-.2,-1).normalize(),V(.8,2,3),out));
 assert.ok(Math.abs(out.y-1)<1e-9);
 // A slider along Z seen from +X/+Z: the plane contains Z and faces the eye, so it is x = anchor.x.
 const {joint:s}=drawer();
 assert.ok(dragPlaneTarget(s,V(0,0,0),V(2,0,5),V(-.3,0,-1).normalize(),V(2,0,5),out));
 assert.ok(Math.abs(out.x)<1e-9,'on the plane through the anchor');
 assert.equal(dragPlaneTarget(joint,V(.8,1,0),V(0,0,3),V(1,0,0),V(0,0,3),out),false,'a ray parallel to the plane misses');
});
