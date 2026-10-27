// Mouse / touch interaction with the parts, next to OrbitControls:
//   hover   outlines the part's box
//   click   (pointer moved < 5 px) animates it to its other end
//   drag    grabs it at the hit point: every physics step the mouse ray, cut with the plane through the
//           anchor perpendicular to the hinge (or the slide plane facing the camera), is the target the
//           joint drags the anchor toward. The camera stays put while it is held; release lets it coast.
//   Shift + move (no button down) over a part pushes it: the pointer's motion, taken on the plane through the
//   hit point facing the camera, is the velocity of a small virtual hand touching the surface (push.js).
//   A click opens or closes the part with a swing (a shove, then an ease into the end); a locked part only jiggles.
import * as THREE from 'three';
import {Drive,dragPlaneTarget} from './drive.js';
import {pushJoint} from './push.js';
import {pickParts,PICK_RADIUS} from './contact.js';
import {smoothing} from './spring.js';

const CLICK_PX=5;
const _dir=new THREE.Vector3();

// The click toggle: head for whichever end it is not at (or not already heading to).
export function toggleTarget(joint){
 const mid=(joint.closedQ+joint.openQ)/2,heading=joint.anim?joint.anim.target:joint.q;
 return (joint.openQ>=joint.closedQ?heading>mid:heading<mid)?joint.closedQ:joint.openQ;
}
export function toggle(part,opts){const j=part.joint;if(j)j.animateTo(toggleTarget(j),{style:'swing',...opts})}

export function createDesktopInput({canvas,camera,controls,root,getParts,isBlocked=()=>false,onSelect=()=>{},outlineBase=()=>0,tune}){
 const raycaster=new THREE.Raycaster(),ndc=new THREE.Vector2(),inv=new THREE.Matrix4();
 const pointer={x:0,y:0,inside:false,shift:false,buttons:0,px:0,py:0};
 // The virtual hand for Shift-pushing: where it touches (joint frame), how fast it moves, and what it pushes.
 const pusher={part:null,body:{position:new THREE.Vector3(),velocity:new THREE.Vector3(),radius:.04},plane:new THREE.Plane(),a:new THREE.Vector3(),b:new THREE.Vector3(),raw:new THREE.Vector3(),vel:new THREE.Vector3()};
 let pending=null,drive=null,hovered=null,heldPart=null;
 const origin=new THREE.Vector3(),dir=new THREE.Vector3(),eye=new THREE.Vector3();

 function setRay(x,y){
  const r=canvas.getBoundingClientRect();
  ndc.set(((x-r.left)/r.width)*2-1,-((y-r.top)/r.height)*2+1);
  raycaster.setFromCamera(ndc,camera);
 }
 // The ray against each part's surface splat centres (spheres of PICK_RADIUS), and the boxes of parts that have none.
 const lo=new THREE.Vector3(),ld=new THREE.Vector3();
 function pick(x,y){
  setRay(x,y);
  root.updateWorldMatrix(true,false);inv.copy(root.matrixWorld).invert();
  lo.copy(raycaster.ray.origin).applyMatrix4(inv);ld.copy(raycaster.ray.direction).transformDirection(inv);
  const hit=pickParts(getParts(),lo,ld,PICK_RADIUS);
  const proxies=getParts().filter(p=>p.interactive&&p.proxy&&!p.contacts).map(p=>p.proxy);
  for(const p of proxies)p.updateWorldMatrix(true,false);
  const box=proxies.length?raycaster.intersectObjects(proxies,false)[0]:null;
  const hitWorld=hit?{part:hit.part,point:root.localToWorld(hit.point.clone()),distance:hit.distance}:null;
  if(box&&(!hitWorld||box.distance<hitWorld.distance))return {part:box.object.userData.part,point:box.point,distance:box.distance};
  return hitWorld;
 }

 const onDown=e=>{
  if(e.target!==canvas||e.button!==0||e.ctrlKey||e.metaKey||e.shiftKey||e.altKey||isBlocked())return;
  const hit=pick(e.clientX,e.clientY);if(!hit)return;
  pending={...hit,x:e.clientX,y:e.clientY,id:e.pointerId};
  controls.enabled=false;
  try{canvas.setPointerCapture(e.pointerId)}catch{}
  onSelect(hit.part);
 };
 const onMove=e=>{
  pointer.x=e.clientX;pointer.y=e.clientY;pointer.inside=e.target===canvas||!!pending||!!drive;
  pointer.shift=e.shiftKey;pointer.buttons=e.buttons;
  if(pending&&!drive&&Math.hypot(e.clientX-pending.x,e.clientY-pending.y)>=CLICK_PX)startGrab();
 };
 const onUp=e=>{
  if(drive){drive.release();drive=null;heldPart=null}
  else if(pending)toggle(pending.part);
  pending=null;
  try{canvas.releasePointerCapture(e.pointerId)}catch{}
  controls.enabled=true;
 };
 const onCancel=e=>{if(drive){drive.release();drive=null;heldPart=null}pending=null;controls.enabled=true};

 function startGrab(){
  const {part,point}=pending;pending=null;
  const joint=part.joint;if(!joint)return;
  const anchor=root.worldToLocal(point.clone());
  heldPart=part;
  drive=new Drive(joint,{anchor,tune,target:(a,out)=>dragPlaneTarget(joint,a,origin,dir,eye,out)});
  updateRay();
 }
 // The mouse ray and eye in the joint's frame (the export's coordinates).
 function updateRay(){
  setRay(pointer.x,pointer.y);
  root.updateWorldMatrix(true,false);inv.copy(root.matrixWorld).invert();
  origin.copy(raycaster.ray.origin).applyMatrix4(inv);dir.copy(raycaster.ray.direction).transformDirection(inv);
  camera.getWorldPosition(eye).applyMatrix4(inv);
 }

 const onKey=e=>{if(e.key==='Shift')pointer.shift=e.type==='keydown'};
 window.addEventListener('keydown',onKey);window.addEventListener('keyup',onKey);
 // Shift-push: the pointer's world motion at the hit point's depth is the virtual hand's velocity.
 function updatePush(hit,dt){
  const on=pointer.shift&&!drive&&!pending&&pointer.buttons===0&&!!hit&&hit.part.interactive&&!isBlocked();
  if(!on){pusher.part=null;pusher.vel.set(0,0,0);pointer.px=pointer.x;pointer.py=pointer.y;return}
  const wasPart=pusher.part;pusher.part=hit.part;
  // The plane through the hit point facing the camera: the pointer's last and present positions on it are a motion.
  camera.getWorldDirection(_dir);pusher.plane.setFromNormalAndCoplanarPoint(_dir,hit.point);
  const raw=pusher.raw.set(0,0,0);
  if(wasPart===hit.part&&dt>0){
   setRay(pointer.px,pointer.py);const ok=raycaster.ray.intersectPlane(pusher.plane,pusher.a);
   setRay(pointer.x,pointer.y);const ok2=raycaster.ray.intersectPlane(pusher.plane,pusher.b);
   if(ok&&ok2)raw.copy(pusher.b).sub(pusher.a).divideScalar(dt);
  }
  pointer.px=pointer.x;pointer.py=pointer.y;
  // To the joint's frame (the root has no scale), capped, smoothed.
  root.updateWorldMatrix(true,false);inv.copy(root.matrixWorld).invert();
  const speed=Math.min(4,raw.length());
  if(speed>0)raw.transformDirection(inv).multiplyScalar(speed);
  pusher.vel.lerp(raw,smoothing(24,dt));
  // The virtual hand sits just off the surface on the camera's side, so the box is pressed with a clear normal.
  pusher.body.position.copy(hit.point).addScaledVector(raycaster.ray.direction,-.015).applyMatrix4(inv);
 }
 canvas.addEventListener('pointerdown',onDown,true);
 window.addEventListener('pointermove',onMove);window.addEventListener('pointerup',onUp);window.addEventListener('pointercancel',onCancel);
 // The pointer leaving the window mid-drag still ends the grab (pointer capture normally covers this).
 window.addEventListener('blur',onCancel);

 return {
  get held(){return heldPart},get hovered(){return hovered},
  // Called every physics step while the mouse holds a part.
  step(dt){
   drive?.step(dt);
   if(pusher.part?.joint&&pusher.vel.lengthSq()>1e-8){pusher.body.velocity.copy(pusher.vel);pushJoint(pusher.part.joint,pusher.body,dt)}
  },
  update(dt){
   let hit=null;
   if(drive)updateRay();
   else if(!pending&&pointer.inside&&!isBlocked()){hit=pick(pointer.x,pointer.y);hovered=hit?hit.part:null}
   else if(!pointer.inside)hovered=null;
   updatePush(hit,dt);
   canvas.style.cursor=drive?'grabbing':hovered||pending?'pointer':'';
   const k=smoothing(16,dt),base=outlineBase();
   for(const p of getParts()){
    if(!p.outline)continue;
    const want=p===heldPart?1:p===hovered||p.hudHover?.85:base;
    const o=p.outline.material.opacity+(want-p.outline.material.opacity)*k;
    p.outline.material.opacity=o;p.outline.visible=o>.01;
   }
  },
  dispose(){canvas.removeEventListener('pointerdown',onDown,true);window.removeEventListener('pointermove',onMove);window.removeEventListener('pointerup',onUp);window.removeEventListener('pointercancel',onCancel);window.removeEventListener('blur',onCancel);window.removeEventListener('keydown',onKey);window.removeEventListener('keyup',onKey)},
 };
}
