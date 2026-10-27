// Mouse / touch interaction with the parts, next to OrbitControls:
//   hover   outlines the part's box
//   click   (pointer moved < 5 px) animates it to its other end
//   drag    grabs it at the hit point: every physics step the mouse ray, cut with the plane through the
//           anchor perpendicular to the hinge (or the slide plane facing the camera), is the target the
//           joint drags the anchor toward. The camera stays put while it is held; release lets it coast.
import * as THREE from 'three';
import {Drive,dragPlaneTarget} from './drive.js';
import {smoothing} from './spring.js';

const CLICK_PX=5;

// The click toggle: head for whichever end it is not at (or not already heading to).
export function toggleTarget(joint){
 const mid=(joint.closedQ+joint.openQ)/2,heading=joint.anim?joint.anim.target:joint.q;
 return (joint.openQ>=joint.closedQ?heading>mid:heading<mid)?joint.closedQ:joint.openQ;
}
export function toggle(part,opts){const j=part.joint;if(j)j.animateTo(toggleTarget(j),opts)}

export function createDesktopInput({canvas,camera,controls,root,getParts,isBlocked=()=>false,onSelect=()=>{},outlineBase=()=>0,tune}){
 const raycaster=new THREE.Raycaster(),ndc=new THREE.Vector2(),inv=new THREE.Matrix4();
 const pointer={x:0,y:0,inside:false};
 let pending=null,drive=null,hovered=null,heldPart=null;
 const origin=new THREE.Vector3(),dir=new THREE.Vector3(),eye=new THREE.Vector3();

 function setRay(x,y){
  const r=canvas.getBoundingClientRect();
  ndc.set(((x-r.left)/r.width)*2-1,-((y-r.top)/r.height)*2+1);
  raycaster.setFromCamera(ndc,camera);
 }
 function pick(x,y){
  setRay(x,y);
  const proxies=getParts().filter(p=>p.interactive&&p.proxy).map(p=>p.proxy);
  for(const p of proxies)p.updateWorldMatrix(true,false);
  const hit=raycaster.intersectObjects(proxies,false)[0];
  return hit?{part:hit.object.userData.part,point:hit.point}:null;
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

 canvas.addEventListener('pointerdown',onDown,true);
 window.addEventListener('pointermove',onMove);window.addEventListener('pointerup',onUp);window.addEventListener('pointercancel',onCancel);
 // The pointer leaving the window mid-drag still ends the grab (pointer capture normally covers this).
 window.addEventListener('blur',onCancel);

 return {
  get held(){return heldPart},get hovered(){return hovered},
  // Called every physics step while the mouse holds a part.
  step(dt){drive?.step(dt)},
  update(dt){
   if(drive)updateRay();
   else if(!pending&&pointer.inside&&!isBlocked()){const hit=pick(pointer.x,pointer.y);hovered=hit?hit.part:null}
   else if(!pointer.inside)hovered=null;
   canvas.style.cursor=drive?'grabbing':hovered||pending?'pointer':'';
   const k=smoothing(16,dt),base=outlineBase();
   for(const p of getParts()){
    if(!p.outline)continue;
    const want=p===heldPart?1:p===hovered||p.hudHover?.85:base;
    const o=p.outline.material.opacity+(want-p.outline.material.opacity)*k;
    p.outline.material.opacity=o;p.outline.visible=o>.01;
   }
  },
  dispose(){canvas.removeEventListener('pointerdown',onDown,true);window.removeEventListener('pointermove',onMove);window.removeEventListener('pointerup',onUp);window.removeEventListener('pointercancel',onCancel);window.removeEventListener('blur',onCancel)},
 };
}
