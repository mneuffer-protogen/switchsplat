// Contact sets: grabbing, pushing and picking against a part's real surface splat centres instead of its box.
import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {JointedPart,HINGE,STEP} from '../src/joint.js';
import {filterContactCentres,buildContacts,ContactSet,nearestPart,pickParts,contactDistance,GRAB_RADIUS,PICK_RADIUS} from '../src/contact.js';
import {pushJoint,contactSetContact,HAND_RADIUS} from '../src/push.js';
import {plyPoints} from '../src/ply-centres.js';
import {DEG} from '../src/math.js';

const V=(x,y,z)=>new THREE.Vector3(x,y,z);
function lcg(seed){let s=seed;return ()=>((s=(s*1664525+1013904223)>>>0)/4294967296)}
const logit=p=>Math.log(p/(1-p));

// A sparse door leaf scan, 0.9 x 2 m in the XY plane, 4.5 x 5 cm apart with a few mm of z noise, plus junk:
// isolated floaters, faint splats and huge splats. Returns the arrays a PLY would give.
function scan({floaters=30,faint=20,huge=20}={}){
 const rnd=lcg(11),c=[],op=[],sc=[];
 for(let ix=0;ix<20;ix++)for(let iy=0;iy<40;iy++){c.push(.05+ix*.045,iy*.05,(rnd()-.5)*.01);op.push(logit(.8));sc.push(Math.log(.02))}
 const surface=c.length/3;
 for(let i=0;i<floaters;i++){c.push((rnd()-.5)*6,(rnd()-.5)*6,(rnd()-.5)*6);op.push(logit(.9));sc.push(Math.log(.03))}
 for(let i=0;i<faint;i++){c.push(.1+rnd()*.7,rnd()*2,0);op.push(logit(.05));sc.push(Math.log(.02))}
 for(let i=0;i<huge;i++){c.push(.1+rnd()*.7,rnd()*2,0);op.push(logit(.9));sc.push(Math.log(.6))}
 return {centres:new Float32Array(c),opacity:new Float32Array(op),maxLogScale:new Float32Array(sc),surface};
}
function door(contacts,opts={}){
 const object=new THREE.Object3D();
 const joint=new JointedPart({object,kind:HINGE,axis:V(0,1,0),pivot:V(0,0,0),min:0,max:100*DEG,open:100*DEG,friction:1.5,frictionTorque:5,mass:25,
  box:{center:V(.45,1,0),size:V(.9,2,.04)},...opts});
 joint.contacts=contacts;
 return {object,joint,part:{id:'d',joint,contacts,interactive:true}};
}

test('floater filtering: faint, huge and isolated splats are dropped; the box is the door, not the junk',()=>{
 const s=scan();
 const {points,stats}=filterContactCentres(s.centres,{opacity:s.opacity,maxLogScale:s.maxLogScale});
 assert.equal(stats.input,s.surface+70);
 assert.equal(stats.faint,20);assert.equal(stats.huge,20);
 assert.ok(stats.isolated>=25,`isolated ${stats.isolated}`);
 assert.ok(stats.final>=s.surface-5&&stats.final<=s.surface+2,`kept ${stats.final} of ${stats.input} (surface ${s.surface})`);
 // Without the junk filter the same cloud's 5..95 % box is far larger in at least one direction.
 const set=buildContacts(s);
 const box=set.fitBox();
 const sizes=[box.size.x,box.size.y,box.size.z].sort((a,b)=>b-a);
 assert.ok(sizes[0]<2.1&&sizes[0]>1.6&&sizes[1]<.9&&sizes[1]>.6,`box ${sizes}`);
 assert.ok(set.thinReliable&&Math.abs(set.thinAxis.z)>.99,`thin axis ${set.thinAxis.toArray()} ratio ${set.thinRatio.toFixed(3)}`);
 // A cloud with no attributes (a PLY without opacity/scale) is still filtered for isolation.
 const bare=filterContactCentres(s.centres);assert.ok(bare.stats.isolated>=25);
 // Too sparse to filter: keep what there is rather than nothing.
 const tiny=filterContactCentres(new Float32Array([0,0,0, 1,0,0, 2,0,0, 3,0,0, 4,0,0]));
 assert.equal(tiny.stats.final,5);assert.ok(tiny.stats.fallback);
 // A big dense part is thinned to the point budget.
 const dense=[];for(let i=0;i<30000;i++)dense.push((i%100)*.01,Math.floor(i/100)%100*.01,Math.floor(i/10000)*.01);
 assert.ok(filterContactCentres(new Float32Array(dense)).stats.final<=8000);
});

test('plyPoints reads centres, opacity and the largest log scale',()=>{
 const props=['x','y','z','opacity','scale_0','scale_1','scale_2'],n=3;
 const header=`ply\nformat binary_little_endian 1.0\nelement vertex ${n}\n${props.map(p=>`property float ${p}`).join('\n')}\nend_header\n`;
 const buf=new ArrayBuffer(header.length+n*28),u8=new Uint8Array(buf),dv=new DataView(buf);
 for(let i=0;i<header.length;i++)u8[i]=header.charCodeAt(i);
 for(let i=0;i<n;i++){const b=header.length+i*28;[i,2*i,3*i,i-1,-3,-2-i,-4].forEach((v,k)=>dv.setFloat32(b+k*4,v,true))}
 const r=plyPoints(buf);
 assert.deepEqual([...r.centres.slice(3,6)],[1,2,3]);assert.deepEqual([...r.opacity],[-1,0,1]);assert.deepEqual([...r.maxLogScale],[-2,-3,-3]);
 // Missing properties give nulls, not zeros.
 const h2='ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n';
 const b2=new ArrayBuffer(h2.length+12);for(let i=0;i<h2.length;i++)new Uint8Array(b2)[i]=h2.charCodeAt(i);
 const r2=plyPoints(b2);assert.equal(r2.opacity,null);assert.equal(r2.maxLogScale,null);
});

test('grab by proximity: within 7 cm of a surface splat, nearest part wins, and it follows the joint',()=>{
 const s=scan(),set=buildContacts(s);
 const {joint,part}=door(set);
 // 5 cm in front of a splat of the door's face: in reach. 9 cm: not.
 const near=V(.5,1,.05),far=V(.5,1,.095);
 // (the splat grid has a point within a few mm of (.5,1,0), so distances are about the z offset)
 assert.ok(contactDistance(part,near,GRAB_RADIUS)<.07);
 assert.equal(contactDistance(part,far,GRAB_RADIUS),Infinity);
 // The old box would have been the 0.9 x 2 m leaf; a point 1 m beside the leaf's far edge is nowhere near the splats.
 assert.equal(nearestPart([part],V(1.4,1,0)),null);
 // Two parts: the closer one is picked.
 const other=door(buildContacts(scan()),{});other.part.id='other';
 other.joint.setPosition(90*DEG);// swung out to -Z: its splats are elsewhere
 const hit=nearestPart([other.part,part],V(.5,1,.04));
 assert.equal(hit.part,part);
 const hit2=nearestPart([part,other.part],V(0,1,-.5));// next to the swung door's face (x=0 plane turned onto z)
 assert.equal(hit2?.part,other.part,'the swung-out door is where the hand is');
 // Following the joint: after opening 40 degrees the old place is empty and the new place is grabbable.
 joint.setPosition(40*DEG);
 assert.equal(nearestPart([part],V(.5,1,.05)),null);
 const moved=joint.pointAt(V(.5,1,0)).add(V(0,0,-.05));
 assert.ok(nearestPart([part],moved),'the hand follows the door out');
 // A part that is not interactive is never grabbed.
 assert.equal(nearestPart([{...part,interactive:false}],moved),null);
});

test('push: a 7 cm shell around the splats, normal = thin axis signed toward the hand, turned with the part',()=>{
 const set=buildContacts(scan());
 const {joint}=door(set);
 const out={point:V(0,0,0),normal:V(0,0,0),depth:0};
 const front=contactSetContact(joint,set,V(.5,1,.06),HAND_RADIUS,out);
 assert.ok(front&&front.normal.distanceTo(V(0,0,1))<.05,`front normal ${front?.normal.toArray()}`);
 assert.ok(Math.abs(front.point.z)<.01&&front.depth>0&&front.depth<.07);
 const back=contactSetContact(joint,set,V(.5,1,-.06),HAND_RADIUS,out);
 assert.ok(back.normal.distanceTo(V(0,0,-1))<.05,'the other side, the other sign');
 assert.equal(contactSetContact(joint,set,V(.5,1,.1),HAND_RADIUS,out),null,'beyond 4.5 + 2.5 cm: no contact');
 // Door swung 90 degrees: its face normal now points along +X (rest +Z turned about +Y by 90).
 joint.setPosition(90*DEG);
 const p=joint.pointAt(V(.5,1,0));
 const turned=contactSetContact(joint,set,p.clone().add(V(.06,0,0)),HAND_RADIUS,out);
 assert.ok(turned.normal.distanceTo(V(1,0,0))<.06,`turned normal ${turned.normal.toArray()}`);
});

test('push with a contact set opens the door the right way, only near the splats, and a latched one holds',()=>{
 const set=buildContacts(scan());
 const {joint}=door(set,{latch:0});
 const hand=(pos,vel)=>({position:pos.clone(),velocity:vel.clone(),radius:HAND_RADIUS});
 const h=hand(V(.8,1,.065),V(0,0,-.5));
 const c=pushJoint(joint,h,STEP);
 assert.ok(c&&c.pushed);assert.ok(joint.qd>0&&Math.abs(joint.qd-.5/.8)<.15,`qd ${joint.qd}`);
 // Where the scan has no splats (the box would still have been there): nothing, e.g. 40 cm past the leaf's edge height.
 const {joint:j2}=door(set,{latch:0});
 assert.equal(pushJoint(j2,hand(V(.5,2.4,.04),V(0,0,-1)),STEP),null);
 // Latched: blocked.
 const {joint:j3}=door(set,{latch:3*DEG});j3.step(STEP);assert.ok(j3.latched);
 const r=pushJoint(j3,hand(V(.8,1,.065),V(0,0,-1.2)),STEP);
 j3.step(STEP);assert.ok(r.blocked&&j3.q<.05*DEG&&j3.latched);
});

test('a part with no clear thin axis pushes along the line from the nearest splat to the hand',()=>{
 // A blob: points on a 5 cm lattice in a 0.4 m cube.
 const pts=[];for(let x=0;x<9;x++)for(let y=0;y<9;y++)for(let z=0;z<9;z++)pts.push(x*.05,y*.05,z*.05);
 const set=new ContactSet(new Float32Array(pts));
 assert.ok(!set.thinReliable,`ratio ${set.thinRatio}`);
 const {joint}=door(set);
 const out={point:V(0,0,0),normal:V(0,0,0),depth:0};
 // The hand is diagonal to the nearest splat (the cube's corner at the origin).
 const c=contactSetContact(joint,set,V(-.04,-.04,0),.045,out);
 assert.ok(c);assert.ok(c.normal.distanceTo(V(-1,-1,0).normalize())<1e-6,`normal ${c.normal.toArray()}`);
});

test('ray pick on a point cloud: nearest surface along the ray, within 4 cm, through the joint pose',()=>{
 const set=buildContacts(scan());
 const {joint,part}=door(set);
 const o=V(.5,1,3),d=V(0,0,-1);
 const hit=pickParts([part],o,d,PICK_RADIUS);
 assert.ok(hit&&hit.part===part);assert.ok(Math.abs(hit.distance-(3-PICK_RADIUS))<.03,`distance ${hit.distance}`);
 assert.ok(Math.abs(hit.point.z-PICK_RADIUS)<.03,'the hit is on the sphere facing the ray');
 // 6 cm sideways of the lattice cell centre... still within a splat radius of a neighbour? the lattice is 4.5 x 5 cm: a ray
 // between rows still hits (nearest splat is at most ~3.4 cm away) but one beside the leaf misses.
 assert.ok(pickParts([part],V(.5,1.02,3),d,PICK_RADIUS));
 assert.equal(pickParts([part],V(1.1,1,3),d,PICK_RADIUS),null,'beside the leaf');
 assert.equal(pickParts([part],V(.5,1,3),V(0,0,1),PICK_RADIUS),null,'pointing away');
 // The door swung to 90 degrees lies along the x = 0 plane facing +X: a ray along -X from the +X side hits it.
 joint.setPosition(90*DEG);
 const side=pickParts([part],V(3,1,-.5),V(-1,0,0),PICK_RADIUS);
 assert.ok(side&&Math.abs(side.point.x-PICK_RADIUS)<.03,`swung door hit at ${side?.point.toArray()}`);
 assert.equal(pickParts([part],V(.5,1,3),d,PICK_RADIUS),null,'and nothing at the old place');
 // Two parts: the nearer along the ray wins.
 joint.setPosition(0);
 const behind=scan();for(let i=2;i<behind.centres.length;i+=3)behind.centres[i]-=1;// a second leaf 1 m further along the ray
 const second=door(buildContacts(behind));second.part.id='second';
 assert.equal(pickParts([second.part,part],o,d,PICK_RADIUS).part,part,'the nearer leaf along the ray wins');
 assert.equal(pickParts([part,second.part],V(.5,1,-.5),d,PICK_RADIUS).part,second.part,'from between them, the one ahead');
 // Non-interactive parts and parts without contacts are skipped.
 assert.equal(pickParts([{...part,interactive:false}],o,d),null);assert.equal(pickParts([{joint,interactive:true,contacts:null}],o,d),null);
});
