// WebXR controllers for the parts.
//   grip (squeeze) with the controller's grip point inside, or within 6 cm of, a part's box: grab. Alyx
//   `createDrive`: the grip point is stored on the part at the grab, and every physics step the part's
//   joint pulls that point toward where the controller's grip now is. Release on squeeze end; the part
//   coasts on its momentum.
//   trigger on / pointing at a part: toggle it (animated open / close)
//   pushing: each controller's grip point is a small sphere (HAND_RADIUS) that shoves any part it is not holding
//   (push.js: the Alyx holdContacts momentum exchange with the hand as an infinite mass). A latched or locked
//   part does not give way, and a push never unlatches it; a grab does (after a moment on the handle).
// Haptics: grab 0.35 for 0.06 s, 'limit' scaled by speed, 'latched', 'slam' (a door shut hard), a pulse on
// contact scaled by the impact speed, a long firm one when a locked part is grabbed or clicked, and a tick
// when a part comes in reach.
import * as THREE from 'three';
import {XRControllerModelFactory} from 'three/addons/webxr/XRControllerModelFactory.js';
import {Drive} from './drive.js';
import {distanceToProxy} from './obb.js';
import {nearestPart,pickParts,GRAB_RADIUS,AIM_RADIUS} from './contact.js';
import {pushJoint,HandMotion,HAND_RADIUS} from './push.js';
import {toggle} from './desktop-input.js';
import {smoothing} from './spring.js';
import {clamp} from './math.js';

// A hand grabs a part whose surface splats come within GRAB_RADIUS (7 cm) of the grip point; a part with no
// splat centres (no contact set) falls back to its box within GRAB_REACH.
export const GRAB_REACH=.06;
export {GRAB_RADIUS};
const IDLE_COLOR=0x66ccff,HIT_COLOR=0xffb020,IDLE_OPACITY=.16,HIT_OPACITY=.5;

export function createXrInput({renderer,scene,root,getParts,tune,onSelect=()=>{}}){
 const factory=new XRControllerModelFactory(),raycaster=new THREE.Raycaster();
 const colliderGeo=new THREE.SphereGeometry(HAND_RADIUS,18,12);
 const hands=[];
 const rayMat=new THREE.LineBasicMaterial({color:0xffffff,transparent:true,opacity:.35,depthTest:false});
 const rayGeo=new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(),new THREE.Vector3(0,0,-1)]);

 const pulse=(hand,intensity,seconds)=>{
  const gp=hand.source?.gamepad;if(!gp)return;
  try{
   const act=gp.hapticActuators?.[0];
   if(act?.pulse)act.pulse(clamp(intensity,0,1),seconds*1000);
   else gp.vibrationActuator?.playEffect?.('dual-rumble',{duration:seconds*1000,strongMagnitude:clamp(intensity,0,1),weakMagnitude:clamp(intensity,0,1)});
  }catch{}
 };
 // The hands that hold a part, or touched it within the last 0.3 s, feel what happens to it.
 let clock=0;
 const feel=(joint,intensity,seconds)=>{for(const h of hands)if(h.drive?.joint===joint||clock-(h.touched.get(joint)??-9)<.3)pulse(h,intensity,seconds)};
 // Joint events that should be felt in the hand holding the part.
 const unsub=[];
 for(const p of getParts())if(p.joint){
  unsub.push(p.joint.on('limit',speed=>feel(p.joint,.15+Math.min(.85,speed*.12),.05)));
  unsub.push(p.joint.on('latched',()=>feel(p.joint,.5,.04)));
  unsub.push(p.joint.on('slam',speed=>feel(p.joint,Math.min(1,.6+speed*.08),.09)));
  unsub.push(p.joint.on('unlatched',()=>feel(p.joint,.3,.03)));
  unsub.push(p.joint.on('activated',()=>feel(p.joint,.4,.05)));
  unsub.push(p.joint.on('deactivated',()=>feel(p.joint,.25,.05)));
 }

 function release(hand){if(hand.drive){hand.drive.release();hand.drive=null;hand.part=null}}

 function nearest(hand){
  // Nearest part over all parts, by distance from the grip point to the part's contact splats.
  const hit=nearestPart(getParts(),hand.local,GRAB_RADIUS);
  let best=hit?.part??null,bestD=hit?hit.distance:Infinity;
  for(const p of getParts()){
   if(!p.interactive||!p.proxy||p.contacts)continue;
   const d=distanceToProxy(p.proxy,hand.pos);
   if(d<=GRAB_REACH&&d<bestD){best=p;bestD=d}
  }
  return best;
 }
 const _ro=new THREE.Vector3(),_rd=new THREE.Vector3(),_inv=new THREE.Matrix4();
 function aimed(hand){
  hand.controller.updateWorldMatrix(true,false);
  raycaster.ray.origin.setFromMatrixPosition(hand.controller.matrixWorld);
  raycaster.ray.direction.set(0,0,-1).transformDirection(hand.controller.matrixWorld);
  raycaster.far=6;
  // The pointing ray against the contact splats (in the joint's frame), and the boxes of parts that have none.
  _inv.copy(root.matrixWorld).invert();
  _ro.copy(raycaster.ray.origin).applyMatrix4(_inv);_rd.copy(raycaster.ray.direction).transformDirection(_inv);
  let hit=pickParts(getParts(),_ro,_rd,AIM_RADIUS,6);// metres
  const proxies=getParts().filter(p=>p.interactive&&p.proxy&&!p.contacts).map(p=>p.proxy);
  for(const p of proxies)p.updateWorldMatrix(true,false);
  const box=proxies.length?raycaster.intersectObjects(proxies,false)[0]:null;
  if(box&&(!hit||box.distance<hit.distance))hit={part:box.object.userData.part,distance:box.distance};
  return hit?{part:hit.part,distance:hit.distance}:null;
 }

 for(let i=0;i<2;i++){
  const controller=renderer.xr.getController(i),grip=renderer.xr.getControllerGrip(i);
  grip.add(factory.createControllerModel(grip));
  const ray=new THREE.Line(rayGeo,rayMat);ray.scale.z=.5;controller.add(ray);
  scene.add(controller,grip);
  // The hand collider: a translucent sphere at the grip point, so you can see what pushes.
  const collider=new THREE.Mesh(colliderGeo,new THREE.MeshBasicMaterial({color:IDLE_COLOR,transparent:true,opacity:IDLE_OPACITY,depthWrite:false}));
  collider.name=`hand${i}:collider`;collider.renderOrder=5;grip.add(collider);
  const hand={i,controller,grip,ray,collider,source:null,hover:null,aim:null,drive:null,part:null,pos:new THREE.Vector3(),local:new THREE.Vector3(),squeezing:false,
   motion:new HandMotion(),touched:new Map(),touching:new Set(),pushNow:new Set(),handBody:{position:new THREE.Vector3(),velocity:new THREE.Vector3(),radius:HAND_RADIUS}};
  hands.push(hand);
  const on=(type,fn)=>{controller.addEventListener(type,fn);unsub.push(()=>controller.removeEventListener(type,fn))};
  on('connected',e=>{hand.source=e.data;controller.visible=grip.visible=true});
  on('disconnected',()=>{release(hand);hand.source=null;hand.motion.reset();controller.visible=grip.visible=false});
  on('squeezestart',()=>{
   hand.squeezing=true;
   if(hand.drive)return;
   refresh(hand);
   const part=hand.hover;if(!part?.joint)return;
   // The grip point (the controller's origin) is both where the part is held and where it is pulled to.
   hand.part=part;onSelect(part);
   hand.drive=new Drive(part.joint,{anchor:hand.local.clone(),tune,target:(a,out)=>{out.copy(hand.local);return true}});
   // A locked part only rattles its handle.
   if(part.joint.locked)pulse(hand,.5,.12);else pulse(hand,.35,.06);
  });
  on('squeezeend',()=>{hand.squeezing=false;release(hand)});
  on('selectstart',()=>{
   if(hand.drive)return;
   refresh(hand);
   const part=hand.hover??hand.aim?.part;if(!part?.joint)return;
   onSelect(part);toggle(part);
   if(part.joint.locked)pulse(hand,.5,.12);else pulse(hand,.2,.03);
  });
 }

 // The hand sphere against every part it is not holding. A hand holding nothing pushes too (squeezed or not).
 function pushStep(hand,dt){
  const body=hand.handBody;body.position.copy(hand.local);body.velocity.copy(hand.motion.velocity);
  hand.pushNow.clear();
  for(const p of getParts()){
   if(!p.interactive||!p.joint||p===hand.part)continue;
   const r=pushJoint(p.joint,body,dt);
   if(!r)continue;
   hand.pushNow.add(p);hand.touched.set(p.joint,clock);
   // A new contact is felt in proportion to how hard it landed (closing speed in m/s); a locked or latched part
   // that does not give way is felt as a firmer knock.
   if(!hand.touching.has(p)){
    const speed=r.closing,knock=r.blocked?.15:0;
    if(speed>.02)pulse(hand,clamp(.1+speed*.5+knock,0,1),.025+Math.min(.05,speed*.03));
   }
  }
  hand.touching.clear();for(const p of hand.pushNow)hand.touching.add(p);
 }

 // Where the controller's grip is now (world, and in the joint's frame), and what it is next to or aiming at.
 function refresh(hand){
  hand.grip.getWorldPosition(hand.pos);
  hand.local.copy(hand.pos);root.worldToLocal(hand.local);
  if(hand.drive){hand.hover=hand.part;hand.aim=null;return}
  const before=hand.hover;
  hand.hover=nearest(hand);hand.aim=hand.hover?null:aimed(hand);
  if(hand.hover&&hand.hover!==before)pulse(hand,.1,.02);
 }

 return {
  hands,
  get active(){return hands.some(h=>h.drive)},
  // Called once per frame, before the physics steps.
  update(dt){
   root.updateWorldMatrix(true,false);
   const k=smoothing(16,dt),shown=new Set();
   for(const h of hands){
    if(!h.source){h.motion.reset();continue}
    refresh(h);h.motion.maxSpeed=8/root.scale.x;h.motion.update(h.local,dt);
    // The sphere glows while it touches a part.
    const m=h.collider.material;m.opacity+=((h.touching.size?HIT_OPACITY:IDLE_OPACITY)-m.opacity)*k;
    m.color.setHex(h.touching.size?HIT_COLOR:IDLE_COLOR);
    const target=h.drive?h.part:h.hover??h.aim?.part;
    if(target)shown.add(target);
    // The pointing ray shortens to what it touches and hides while a part is held.
    h.ray.visible=!h.drive;h.ray.scale.z=h.aim?h.aim.distance:h.hover?.04:.4;
   }
   for(const p of getParts()){
    if(!p.outline)continue;
    const want=shown.has(p)?.9:0,o=p.outline.material.opacity+(want-p.outline.material.opacity)*k;
    // Desktop hover also drives this opacity; in XR only the controllers do.
    p.outline.material.opacity=o;p.outline.visible=o>.01;
   }
  },
  // Called every physics step.
  step(dt){
   clock+=dt;
   for(const h of hands){
    h.drive?.step(dt);
    if(h.source&&h.motion.has)pushStep(h,dt);
   }
  },
  // The view jumped (a snap turn, tracking came back): the hands' speeds across it mean nothing.
  noteJump(){for(const h of hands)h.motion.reset()},
  releaseAll(){for(const h of hands){release(h);h.touching.clear();h.touched.clear();h.motion.reset()}},
  // three caches the controller objects, so a second viewer instance gets the same ones: take everything we put on them off.
  dispose(){
   for(const u of unsub)u();
   for(const h of hands){release(h);h.touching.clear();h.touched.clear();h.controller.remove(h.ray);h.grip.clear();h.collider.material.dispose();scene.remove(h.controller,h.grip)}
   colliderGeo.dispose();rayGeo.dispose();rayMat.dispose();
  },
 };
}
