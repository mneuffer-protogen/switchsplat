// WebXR controllers for the parts.
//   grip (squeeze) with the controller's grip point inside, or within 6 cm of, a part's box: grab. Alyx
//   `createDrive`: the grip point is stored on the part at the grab, and every physics step the part's
//   joint pulls that point toward where the controller's grip now is. Release on squeeze end; the part
//   coasts on its momentum.
//   trigger on / pointing at a part: toggle it (animated open / close)
// Haptics: grab 0.35 for 0.06 s, 'limit' scaled by speed, 'latched', and a tick when a part comes in reach.
import * as THREE from 'three';
import {XRControllerModelFactory} from 'three/addons/webxr/XRControllerModelFactory.js';
import {Drive} from './drive.js';
import {distanceToProxy} from './obb.js';
import {toggle} from './desktop-input.js';
import {smoothing} from './spring.js';
import {clamp} from './math.js';

export const GRAB_REACH=.06;

export function createXrInput({renderer,scene,root,getParts,tune,onSelect=()=>{}}){
 const factory=new XRControllerModelFactory(),raycaster=new THREE.Raycaster();
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
 const pulseHolders=(joint,intensity,seconds)=>{for(const h of hands)if(h.drive?.joint===joint)pulse(h,intensity,seconds)};
 // Joint events that should be felt in the hand holding the part.
 const unsub=[];
 for(const p of getParts())if(p.joint){
  unsub.push(p.joint.on('limit',speed=>pulseHolders(p.joint,.15+Math.min(.85,speed*.12),.05)));
  unsub.push(p.joint.on('latched',()=>pulseHolders(p.joint,.5,.04)));
  unsub.push(p.joint.on('unlatched',()=>pulseHolders(p.joint,.3,.03)));
  unsub.push(p.joint.on('activated',()=>pulseHolders(p.joint,.4,.05)));
  unsub.push(p.joint.on('deactivated',()=>pulseHolders(p.joint,.25,.05)));
 }

 function release(hand){if(hand.drive){hand.drive.release();hand.drive=null;hand.part=null}}

 function nearest(hand){
  let best=null,bestD=GRAB_REACH;
  for(const p of getParts()){
   if(!p.interactive||!p.proxy)continue;
   const d=distanceToProxy(p.proxy,hand.pos);
   if(d<=bestD){best=p;bestD=d}
  }
  return best;
 }
 function aimed(hand){
  hand.controller.updateWorldMatrix(true,false);
  raycaster.ray.origin.setFromMatrixPosition(hand.controller.matrixWorld);
  raycaster.ray.direction.set(0,0,-1).transformDirection(hand.controller.matrixWorld);
  raycaster.far=6;
  const proxies=getParts().filter(p=>p.interactive&&p.proxy).map(p=>p.proxy);
  for(const p of proxies)p.updateWorldMatrix(true,false);
  const hit=raycaster.intersectObjects(proxies,false)[0];
  return hit?{part:hit.object.userData.part,distance:hit.distance}:null;
 }

 for(let i=0;i<2;i++){
  const controller=renderer.xr.getController(i),grip=renderer.xr.getControllerGrip(i);
  grip.add(factory.createControllerModel(grip));
  const ray=new THREE.Line(rayGeo,rayMat);ray.scale.z=.5;controller.add(ray);
  scene.add(controller,grip);
  const hand={i,controller,grip,ray,source:null,hover:null,aim:null,drive:null,part:null,pos:new THREE.Vector3(),local:new THREE.Vector3(),squeezing:false};
  hands.push(hand);
  controller.addEventListener('connected',e=>{hand.source=e.data;controller.visible=grip.visible=true});
  controller.addEventListener('disconnected',()=>{release(hand);hand.source=null;controller.visible=grip.visible=false});
  controller.addEventListener('squeezestart',()=>{
   hand.squeezing=true;
   if(hand.drive)return;
   refresh(hand);
   const part=hand.hover;if(!part?.joint)return;
   // The grip point (the controller's origin) is both where the part is held and where it is pulled to.
   hand.part=part;onSelect(part);
   hand.drive=new Drive(part.joint,{anchor:hand.local.clone(),tune,target:(a,out)=>{out.copy(hand.local);return true}});
   pulse(hand,.35,.06);
  });
  controller.addEventListener('squeezeend',()=>{hand.squeezing=false;release(hand)});
  controller.addEventListener('selectstart',()=>{
   if(hand.drive)return;
   refresh(hand);
   const part=hand.hover??hand.aim?.part;if(!part?.joint)return;
   onSelect(part);toggle(part);pulse(hand,.2,.03);
  });
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
    if(!h.source)continue;
    refresh(h);
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
  step(dt){for(const h of hands)h.drive?.step(dt)},
  releaseAll(){for(const h of hands)release(h)},
  dispose(){for(const u of unsub)u();for(const h of hands){release(h);scene.remove(h.controller,h.grip)}},
 };
}
