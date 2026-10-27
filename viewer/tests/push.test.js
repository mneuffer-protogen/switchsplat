// Pushing with a hand sphere (push.js), the lock and jiggle, and the swing-open animation.
import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {JointedPart,HINGE,SLIDER,STEP} from '../src/joint.js';
import {Drive} from '../src/drive.js';
import {pushJoint,sphereObbContact,boxPoseAt,HandMotion,HAND_RADIUS} from '../src/push.js';
import {DEG} from '../src/math.js';

const V=(x,y,z)=>new THREE.Vector3(x,y,z);
// A 0.9 m door leaf in the XY plane (4 cm thick), hinged on its x=0 edge about +Y; opening (+q) swings its free
// edge toward -Z. Car-door-scale defaults: 25 kg, a little joint friction torque.
function door(opts={}){
 const object=new THREE.Object3D();
 const joint=new JointedPart({object,kind:HINGE,axis:V(0,1,0),pivot:V(0,0,0),min:0,max:100*DEG,open:100*DEG,friction:1.5,frictionTorque:5,mass:25,
  box:{center:V(.45,1,0),size:V(.9,2,.04)},...opts});
 return {object,joint};
}
function drawer(opts={}){
 const object=new THREE.Object3D();
 const joint=new JointedPart({object,kind:SLIDER,axis:V(0,0,1),min:0,max:.43,open:.43,friction:3,mass:2.5,box:{center:V(0,0,0),size:V(.5,.2,.4)},...opts});
 return {object,joint};
}
const hand=(position,velocity,radius=HAND_RADIUS)=>({position:position.clone(),velocity:velocity.clone(),radius});
// Steps like the app: the hand (moving at its velocity) pushes, then the joint steps.
function run(seconds,joint,h,{each}={}){
 const n=Math.round(seconds/STEP);let touched=0;
 for(let i=0;i<n;i++){
  each?.(i*STEP,i);
  if(pushJoint(joint,h,STEP))touched++;
  joint.step(STEP);
  h.position.addScaledVector(h.velocity,STEP);
 }
 return touched;
}

test('sphere against the box: closest point, normal toward the sphere, depth, and inside',()=>{
 const {joint}=door();
 const c=sphereObbContact(joint,V(.8,1,.05),.045);
 assert.ok(c,'touching');assert.ok(c.normal.distanceTo(V(0,0,1))<1e-9);assert.ok(c.point.distanceTo(V(.8,1,.02))<1e-9);
 assert.ok(Math.abs(c.depth-(.045-.03))<1e-9);
 assert.equal(sphereObbContact(joint,V(.8,1,.2),.045),null,'too far');
 const back=sphereObbContact(joint,V(.8,1,-.05),.045);assert.ok(back.normal.distanceTo(V(0,0,-1))<1e-9,'other face, other normal');
 const inside=sphereObbContact(joint,V(.8,1,.01),.045);assert.ok(inside.normal.z>.99,'centre inside leaves through the nearest face');
 // The box follows the joint: at 90 degrees the free edge is at -Z.
 joint.setPosition(90*DEG);
 const pose=boxPoseAt(joint,joint.q);assert.ok(pose.center.distanceTo(V(0,1,-.45))<1e-9);
 assert.ok(sphereObbContact(joint,V(.05,1,-.8),.045),'touches the door where it is now');
 assert.equal(sphereObbContact(joint,V(.8,1,.05),.045),null,'and not where it was');
});

test('a hand pushing the free edge of a door opens it, with the sign of the push',()=>{
 const {joint}=door({latch:0});
 // From the +Z side, moving -Z at 0.5 m/s: the door's open direction.
 const h=hand(V(.8,1,.07),V(0,0,-.5));
 const touched=run(.6,joint,h);
 assert.ok(touched>0);
 assert.ok(joint.q>10*DEG,`opened to ${(joint.q/DEG).toFixed(1)} deg`);
 // Right after the first contact the door's edge goes at the hand's speed: qd = 0.5 / 0.8 m.
 const {joint:j2}=door({latch:0});
 const h2=hand(V(.8,1,.06),V(0,0,-.5));pushJoint(j2,h2,STEP);
 assert.ok(Math.abs(j2.qd-.5/.8)<.02,`qd ${j2.qd}`);
 // From the other side, moving +Z: it turns the other way (toward its closed stop, which holds).
 const {joint:j3}=door({latch:0});j3.setPosition(30*DEG);
 const h3=hand(V(.8,1,-.06).applyAxisAngle(V(0,1,0),30*DEG),V(0,0,.5).applyAxisAngle(V(0,1,0),30*DEG));
 pushJoint(j3,h3,STEP);assert.ok(j3.qd<0,'pushing it shut');
 assert.ok(!j3.animating);
});

test('a hand moving at the door keeps the door no faster than the hand, then the door coasts',()=>{
 const {joint}=door({latch:0});
 const h=hand(V(.8,1,.07),V(0,0,-.5));
 let worst=0;
 run(.6,joint,h,{each:()=>{worst=Math.max(worst,joint.qd*.8)}});
 assert.ok(worst<=.5+.05,`edge speed peaked at ${worst.toFixed(2)} m/s for a 0.5 m/s hand`);
 // The hand stops: the door is moving away from it, so it is not held back and friction carries it on.
 const q0=joint.q;h.velocity.set(0,0,0);run(.5,joint,h);
 assert.ok(joint.q>q0+.5*DEG,'coasts after the hand stops');
});

test('a hand that is not moving stops a swinging door (it pushes back like a wall)',()=>{
 const {joint}=door({latch:0});
 joint.qd=1.5;// swinging open
 // A still hand just in front of where the free edge is going: the door arrives, the hand holds it.
 const h=hand(V(.8,1,-.2),V(0,0,0));
 const worstBefore=joint.q;run(1,joint,h);
 assert.ok(Math.abs(joint.qd)<.05,`qd ${joint.qd}`);
 assert.ok(joint.q>worstBefore&&joint.q<40*DEG);
});

test('a hand pushing a latched door does not move it, and never unlatches it',()=>{
 const {joint}=door({latch:3*DEG});
 joint.step(STEP);assert.ok(joint.latched);
 let unlatched=0;joint.on('unlatched',()=>unlatched++);
 const h=hand(V(.8,1,.07),V(0,0,-1.2));
 const r=[];
 for(let i=0;i<180;i++){const c=pushJoint(joint,h,STEP);if(c)r.push(c);joint.step(STEP);h.position.addScaledVector(h.velocity,STEP);if(i>40)h.position.z=.04}
 assert.ok(r.length>0&&r.some(c=>c.blocked),'the hand felt it blocked');
 assert.ok(joint.q<.05*DEG,`latched door turned ${(joint.q/DEG).toFixed(3)} deg`);
 assert.ok(joint.latched&&unlatched===0);
});

test('a locked door does not move for a push either, and stays where it is',()=>{
 const {joint}=door({latch:0});joint.setPosition(20*DEG);joint.setLocked(true);
 const p=joint.pointAt(V(.8,1,0)).add(V(0,0,.07));
 const h=hand(p,V(0,0,-1));
 for(let i=0;i<90;i++){pushJoint(joint,h,STEP);joint.step(STEP);h.position.addScaledVector(h.velocity,STEP);}
 assert.ok(Math.abs(joint.q-20*DEG)<.05*DEG,`locked door at ${(joint.q/DEG).toFixed(3)} deg`);
});

test('pushing a door shut at speed swings it, hits the stop and latches, with a slam',()=>{
 const {joint}=door({latch:3*DEG});joint.setPosition(40*DEG);
 let latched=0,slam=0,limit=0;joint.on('latched',()=>latched++);joint.on('slam',s=>slam=Math.max(slam,s));joint.on('limit',()=>limit++);
 // Closing turns the free edge along -tangent (q decreases); the hand sits behind the door on that side's
 // opposite face and drives the edge along it at 3 m/s.
 const edge=joint.pointAt(V(.8,1,0));
 const closing=new THREE.Vector3().crossVectors(joint.axis,edge.clone().setY(0)).normalize().negate();
 const h=hand(edge.clone().addScaledVector(closing,-.06),closing.clone().multiplyScalar(3));
 run(1.2,joint,h);
 assert.ok(Math.abs(joint.q)<.05*DEG,`shut at ${(joint.q/DEG).toFixed(3)} deg`);
 assert.ok(joint.latched&&latched===1,'latched');
 assert.ok(slam>1.5,`slam at ${slam.toFixed(2)} rad/s`);
 assert.ok(limit>0,'it struck the stop');
});

test('pushing a door shut slowly also latches, without a slam',()=>{
 const {joint}=door({latch:3*DEG});joint.setPosition(25*DEG);
 let slam=0;joint.on('slam',()=>slam++);
 const edge=joint.pointAt(V(.8,1,0));
 const t=new THREE.Vector3().crossVectors(joint.axis,edge.clone().setY(0)).normalize().negate();
 const h=hand(edge.clone().addScaledVector(t,-.06),t.clone().multiplyScalar(.4));
 run(2.5,joint,h);
 assert.ok(joint.latched&&joint.q<.05*DEG,'latched');
 assert.equal(slam,0);
});

test('a drawer is pushed along its rail by the front, not by its side',()=>{
 const {joint}=drawer({latch:0});
 // Front face at z = +0.2 (box z size .4): a hand moving -Z... the drawer slides +Z when opened, so push from -Z side moving +Z.
 const h=hand(V(0,0,-.2-.04),V(0,0,.4));
 run(.4,joint,h);
 assert.ok(joint.q>.05,`drawer slid ${joint.q.toFixed(3)} m`);
 const {joint:s}=drawer({latch:0});
 const side=hand(V(.25+.04,0,0),V(-.5,0,0));// pressing the side across the slide
 run(.4,s,side);
 assert.ok(Math.abs(s.q)<1e-6,'a hand pressed on the side moves nothing');
});

test('a pushed part stays inside its limits, and the hand speed cap holds',()=>{
 const {joint}=door({latch:0,friction:0});
 const h=hand(V(.05,1,.07),V(0,0,-8));// right next to the hinge, a violent shove
 let worst=0;
 run(.3,joint,h,{each:()=>{worst=Math.max(worst,Math.abs(joint.qd))}});
 assert.ok(worst<=6+1e-9,`peak ${worst} rad/s`);
 const {joint:j2}=door({latch:0});
 j2.qd=0;const h2=hand(V(.8,1,.07),V(0,0,-5));
 run(2,j2,h2);assert.ok(j2.q<=j2.hi+1e-9&&j2.q>=-1e-9);
});

test('a push takes a click animation away from the part',()=>{
 const {joint}=door({latch:0});
 joint.animateTo(joint.openQ,{style:'swing'});assert.ok(joint.animating);
 for(let i=0;i<10;i++)joint.step(STEP);
 const h=hand(V(.8,1,.07).applyAxisAngle(V(0,1,0),joint.q),V(0,0,0).applyAxisAngle(V(0,1,0),joint.q));
 // The hand is behind the door, still: the door runs into it.
 const edge=joint.pointAt(V(.8,1,0));h.position.copy(edge).add(V(0,0,-.05));
 let c=null;for(let i=0;i<60&&!c;i++){c=pushJoint(joint,h,STEP);if(!c)joint.step(STEP)}
 assert.ok(c&&c.pushed);assert.ok(!joint.animating,'the push interrupts the animation');
});

test('hand motion: smoothed velocity from poses, no speed across a jump',()=>{
 const m=new HandMotion({rate:1000});
 m.update(V(0,0,0),1/72);assert.equal(m.velocity.length(),0,'first sample has no velocity');
 for(let i=1;i<=10;i++)m.update(V(i*.01,0,0),1/72);
 assert.ok(Math.abs(m.velocity.x-.72)<.01,`vx ${m.velocity.x}`);
 m.update(V(50,0,0),1/72);assert.equal(m.velocity.length(),0,'a jump (tracking loss, snap turn) is not a speed');
 m.reset();assert.ok(!m.has);
});

// ------------------------------------------------------------------ lock and jiggle

test('a locked door only jiggles: a grab and a pull do not open it',()=>{
 const {object,joint}=door({latch:3*DEG,locked:true});
 let jiggles=0,latched=0;joint.on('jiggle',()=>jiggles++);joint.on('latched',()=>latched++);
 joint.step(STEP);assert.ok(!joint.latched&&latched===0,'a locked door does not latch (it is not free to)');
 const start=V(.8,1,0);let t=0;
 const drive=new Drive(joint,{anchor:start,target:(a,out)=>{out.copy(start).applyAxisAngle(V(0,1,0),Math.min(40*DEG,t*60*DEG));return true}});
 assert.equal(jiggles,1,'the grab rattles the handle');
 let maxDrawn=0,seenJig=false;
 for(let i=0;i<180;i++){t=i*STEP;drive.step(STEP);joint.step(STEP);joint.applyPose();
  const drawn=new THREE.Euler().setFromQuaternion(object.quaternion,'YXZ').y;
  maxDrawn=Math.max(maxDrawn,Math.abs(drawn));if(joint.jig)seenJig=true}
 assert.ok(joint.q<.05*DEG,`locked door opened to ${(joint.q/DEG).toFixed(3)} deg`);
 assert.ok(seenJig&&maxDrawn>.3*DEG&&maxDrawn<=1.5*DEG+1e-6,`jiggle amplitude ${(maxDrawn/DEG).toFixed(2)} deg, at most 1.5`);
 drive.release();
});

test('animating a locked door open jiggles it instead; the jiggle fades and q never changes',()=>{
 const {joint}=door({latch:0,locked:true});
 let jiggles=0;joint.on('jiggle',()=>jiggles++);
 joint.animateTo(joint.openQ,{style:'swing'});
 assert.equal(jiggles,1);assert.ok(!joint.animating);
 for(let i=0;i<20;i++)joint.step(STEP);
 assert.equal(joint.q,0);
 joint.animateTo(joint.openQ);assert.equal(jiggles,1,'still rattling: not retriggered');
 for(let i=0;i<60;i++)joint.step(STEP);
 assert.equal(joint.jig,null,'the rattle ends');
 joint.animateTo(joint.openQ);assert.equal(jiggles,2);
 // A slider rattles by 5 mm.
 const {joint:d}=drawer({locked:true});d.jiggle();
 let peak=0;for(let i=0;i<40;i++){d.step(STEP);peak=Math.max(peak,Math.abs(d.jiggleOffset()))}
 assert.ok(peak>.001&&peak<=.005+1e-9,`drawer rattle ${peak}`);
});

test('unlocking frees the part; locking an open door freezes it where it is',()=>{
 const {joint}=door({latch:3*DEG,locked:true});
 let events=[];joint.on('locked',v=>events.push(v));
 joint.setLocked(false);assert.deepEqual(events,[false]);
 joint.step(STEP);assert.ok(joint.latched,'free again: shut, it latches');
 joint.animateTo(joint.openQ);for(let i=0;i<180;i++)joint.step(STEP);
 assert.equal(joint.q,joint.openQ);
 joint.setLocked(true);joint.qd=3;for(let i=0;i<30;i++)joint.step(STEP);
 assert.ok(Math.abs(joint.q-joint.openQ)<.02*DEG,'a locked open door stays put (a hair of hold at most)');
 joint.setLocked(false);joint.animateTo(0);for(let i=0;i<180;i++)joint.step(STEP);
 assert.ok(joint.q<.05*DEG);
});

test('a latched door needs a grab held for a moment to let go; a quick tap does not',()=>{
 const {joint}=door({latch:3*DEG,latchReleaseDelay:.25});
 joint.step(STEP);assert.ok(joint.latched);
 joint.grab();for(let i=0;i<10;i++)joint.step(STEP);
 assert.ok(joint.latched,'0.11 s on the handle: still latched');
 for(let i=0;i<20;i++)joint.step(STEP);
 assert.ok(!joint.latched,'0.33 s: released');
 joint.release();
 const {joint:j2}=door({latch:3*DEG,latchReleaseDelay:.25});j2.step(STEP);
 j2.grab();for(let i=0;i<10;i++)j2.step(STEP);j2.release();for(let i=0;i<60;i++)j2.step(STEP);
 assert.ok(j2.latched,'a quick tap never unlatched it');
});

// ------------------------------------------------------------------ swing open

function trace(joint,seconds,each){
 const n=Math.round(seconds/STEP),out={max:-1,min:1e9,monotonic:true,peakSpeed:0,qs:[]};let prev=joint.q;
 for(let i=0;i<n;i++){joint.step(STEP);each?.(i);out.max=Math.max(out.max,joint.q);out.min=Math.min(out.min,joint.q);if(joint.q<prev-1e-9)out.monotonic=false;prev=joint.q;out.peakSpeed=Math.max(out.peakSpeed,Math.abs(joint.qd));out.qs.push(joint.q)}
 return out;
}

test('swing open: leaves the latch with a shove, lands on the target, never crosses a limit',()=>{
 const {joint}=door({latch:3*DEG,max:60*DEG,open:60*DEG});
 joint.step(STEP);assert.ok(joint.latched);
 let unlatched=0,settled=0;joint.on('unlatched',()=>unlatched++);joint.on('settled',()=>settled++);
 joint.animateTo(joint.openQ,{style:'swing'});
 assert.ok(unlatched===1&&!joint.latched,'unlatches first');
 assert.ok(joint.qd>0,'with an initial speed');
 const resp=joint.anim.response;
 assert.ok(resp>.7&&resp<1.1,`a car door settles in about 0.9 s (${resp.toFixed(2)})`);
 const t=trace(joint,2);
 assert.ok(t.max<=joint.hi+1e-9&&t.min>=-1e-9);
 assert.ok(t.monotonic,'no overshoot, no backing up');
 assert.equal(joint.q,joint.openQ);assert.equal(joint.qd,0);assert.equal(settled,1);
 // A cabinet door and a drawer are quicker.
 const cab=door({box:{center:V(.2,.4,0),size:V(.4,.8,.02)},mass:2,max:95*DEG,open:95*DEG,latch:0}).joint;cab.animateTo(cab.openQ,{style:'swing'});
 assert.ok(cab.anim.response<resp&&cab.anim.response>.3,`cabinet ${cab.anim.response.toFixed(2)} s`);
 const dr=drawer({latch:0}).joint;dr.animateTo(dr.openQ,{style:'swing'});assert.ok(dr.anim.response<resp);
 // It starts faster than the plain critically damped drive, which starts at rest.
 const plain=door({latch:0,max:60*DEG,open:60*DEG}).joint,kicked=door({latch:0,max:60*DEG,open:60*DEG}).joint;
 plain.animateTo(plain.openQ,{response:resp});kicked.animateTo(kicked.openQ,{response:resp,style:'swing'});
 for(let i=0;i<9;i++){plain.step(STEP);kicked.step(STEP)}
 assert.ok(kicked.q>plain.q*1.5,`${(kicked.q/DEG).toFixed(1)} vs ${(plain.q/DEG).toFixed(1)} deg after 0.1 s`);
});

test('swing close arrives, then latches; a swing to the middle stops in the middle',()=>{
 const {joint}=door({latch:3*DEG});
 joint.animateTo(joint.openQ,{style:'swing'});trace(joint,2);
 joint.animateTo(joint.closedQ,{style:'swing'});
 let latchedWhileMoving=false;const t=trace(joint,2,()=>{if(joint.latched&&joint.animating)latchedWhileMoving=true});
 assert.ok(!latchedWhileMoving&&t.min>=-1e-9);
 trace(joint,.1);
 assert.ok(joint.latched&&Math.abs(joint.q)<1e-9,'shut and latched');
 joint.animateTo(50*DEG,{style:'swing'});const t2=trace(joint,2);
 assert.ok(t2.max<=50*DEG+1e-9,`peak ${(t2.max/DEG).toFixed(2)} deg`);assert.equal(joint.q,50*DEG);
});

test('a swing is interruptible by a grab and by a push',()=>{
 const a=door({latch:0}).joint;
 a.animateTo(a.openQ,{style:'swing'});trace(a,.3);
 const q=a.q;assert.ok(a.animating&&q>5*DEG);
 const start=a.pointAt(V(.8,1,0));
 const drive=new Drive(a,{anchor:start,target:(p,out)=>{out.copy(start);return true}});
 assert.ok(!a.animating,'the grab takes it');
 for(let i=0;i<90;i++){drive.step(STEP);a.step(STEP)}
 assert.ok(Math.abs(a.q-q)<10*DEG,`held near ${(q/DEG).toFixed(1)}, at ${(a.q/DEG).toFixed(1)}`);drive.release();
 // A hand in the way stops it from opening further.
 const b=door({latch:0}).joint;
 b.animateTo(b.openQ,{style:'swing'});
 const h=hand(V(.8,1,-.3),V(0,0,0));
 for(let i=0;i<180;i++){pushJoint(b,h,STEP);b.step(STEP)}
 assert.ok(b.q<45*DEG&&Math.abs(b.qd)<.05,`stopped by the hand at ${(b.q/DEG).toFixed(1)} deg`);
 assert.ok(!b.animating);
});

test('the drive still feels right at 25 kg: a grab pulls a car door open, and release coasts',()=>{
 const {joint}=door({latch:3*DEG});
 joint.step(STEP);
 const start=V(.85,1,0);let t=0;
 const drive=new Drive(joint,{anchor:start,target:(a,out)=>{out.copy(start).applyAxisAngle(V(0,1,0),Math.min(45*DEG,t*45*DEG));return true}});
 for(let i=0;i<Math.round(2/STEP);i++){t=i*STEP;drive.step(STEP);joint.step(STEP)}
 assert.ok(Math.abs(joint.q-45*DEG)<6*DEG,`pulled to ${(joint.q/DEG).toFixed(1)} deg`);
 drive.release();for(let i=0;i<Math.round(8/STEP);i++)joint.step(STEP);
 assert.ok(Math.abs(joint.qd)<1e-3,'comes to rest');
});
