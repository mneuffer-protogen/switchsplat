// Edit mode: tune each part's joint live and export the result as a new scene.json.
//   - a lil-gui panel with a folder per jointed part (kind, min/max/open, friction, frictionTorque, latch,
//     mass) and a scrub slider for q
//   - a pivot helper per part (a sphere on the hinge line and a line along the axis), moved and turned with
//     three's TransformControls; edits rebuild the joint on the spot
//   - parts whose joint fit was weak (fit.inlierRatio < 0.6) are flagged
// Everything is in scene.json units (degrees for hinges, metres for sliders) and in the joint's frame.
import * as THREE from 'three';
import GUI from 'lil-gui';
import {TransformControls} from 'three/addons/controls/TransformControls.js';
import {HINGE,SLIDER} from './joint.js';
import {jointToJson} from './scene-loader.js';
import {DEG} from './math.js';

const Y=new THREE.Vector3(0,1,0);

export function createEditMode({scene,loaded,camera,renderer,controls,onEnabled=()=>{}}){
 const {root,parts}=loaded;
 const jointed=parts.filter(p=>p.joint);
 const gui=new GUI({title:'Edit joints'});
 gui.domElement.classList.add('edit-gui');
 const transform=new TransformControls(camera,renderer.domElement);
 transform.setSize(.8);
 const gizmo=transform.getHelper();scene.add(gizmo);
 const state={enabled:false,selected:jointed[0]??null,mode:'translate',dragging:false};
 const rows=new Map();

 const helperMat=color=>new THREE.MeshBasicMaterial({color,depthTest:false,transparent:true,opacity:.95});
 function makeHelper(part){
  const g=new THREE.Group();g.name=`${part.id}:pivot`;
  const ball=new THREE.Mesh(new THREE.SphereGeometry(.018,16,12),helperMat(0xffb020));ball.renderOrder=1002;
  const line=new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0,-.4,0),new THREE.Vector3(0,.4,0)]),new THREE.LineBasicMaterial({color:0xffb020,depthTest:false,transparent:true}));
  line.renderOrder=1002;g.add(ball,line);g.visible=false;
  root.add(g);return g;
 }
 function placeHelper(part,helper){
  const j=part.joint;
  // A slider has no hinge line; show its axis through the box centre instead.
  helper.position.copy(j.kind===SLIDER&&part.obb?part.obb.center:j.pivot);
  helper.quaternion.setFromUnitVectors(Y,j.axis);
 }

 // GUI model in scene.json units.
 const toState=j=>{const u=j.kind===HINGE?1/DEG:1;return {kind:j.kind,min:j.min*u,max:j.max*u,open:j.open*u,friction:j.friction,frictionTorque:j.frictionTorque,latch:j.latch*u,mass:j.bodyMass,locked:j.locked,q:j.q*u}};
 function push(part,s){
  const j=part.joint,u=s.kind===HINGE?DEG:1;
  j.configure({kind:s.kind,min:s.min*u,max:s.max*u,open:Math.min(Math.max(s.open,s.min),s.max)*u,friction:s.friction,frictionTorque:s.frictionTorque,latch:s.latch*u,mass:s.mass});
  j.setLocked(s.locked);j.applyPose();
 }

 for(const part of jointed){
  const j=part.joint,s=toState(j),helper=makeHelper(part);placeHelper(part,helper);
  const folder=gui.addFolder(part.name);
  if(part.lowFit){folder.domElement.classList.add('low-fit');folder.title(`${part.name}   LOW FIT ${part.fit.inlierRatio.toFixed(2)}`)}
  const scrub=folder.add(s,'q',s.min,s.max,.1).name('q (scrub)').onChange(v=>{
   // Scrubbing is the author pulling the part by hand: let go of the latch and any animation.
   j.setPosition(v*(j.kind===HINGE?DEG:1));
  });
  const ctl={};
  const unitsFor=kind=>kind===HINGE?{step:1,range:[-360,360]}:{step:.005,range:[-5,5]};
  const apply=()=>push(part,s);
  const retune=()=>{
   const {step,range}=unitsFor(s.kind);
   for(const k of ['min','max','open']){ctl[k].step(step).min(range[0]).max(range[1]).updateDisplay()}
   ctl.latch.step(step/2).updateDisplay();
   scrub.min(s.min).max(s.max).step(step/10).updateDisplay();
  };
  ctl.kind=folder.add(s,'kind',[HINGE,SLIDER]).onChange(kind=>{
   // The numbers mean different things (degrees, metres), so a change of kind starts from sensible ones.
   Object.assign(s,kind===HINGE?{min:0,max:90,open:90,latch:3}:{min:0,max:.3,open:.3,latch:0});
   s.q=0;j.setPosition(0);retune();apply();placeHelper(part,helper);
  });
  const unit=unitsFor(s.kind);
  ctl.min=folder.add(s,'min',unit.range[0],unit.range[1],unit.step).onChange(apply);
  ctl.max=folder.add(s,'max',unit.range[0],unit.range[1],unit.step).onChange(apply);
  ctl.open=folder.add(s,'open',unit.range[0],unit.range[1],unit.step).name('open (reveal at)').onChange(apply);
  folder.add(s,'friction',0,10,.05).onChange(apply);
  folder.add(s,'frictionTorque',0,50,.1).name('friction torque').onChange(apply);
  ctl.latch=folder.add(s,'latch',0,20,unit.step/2).name('latch (deg / m)').onChange(apply);
  folder.add(s,'mass',.1,200,.1).name('mass (kg)').onChange(apply);
  ctl.locked=folder.add(s,'locked').name('locked').onChange(apply);
  folder.close();
  rows.set(part,{folder,s,scrub,helper,ctl});
 }

 const select=part=>{
  if(!part||!rows.has(part))return;
  state.selected=part;
  for(const [p,r] of rows)p===part?r.folder.open():r.folder.close();
  transform.attach(rows.get(part).helper);
  refreshVisibility();
 };
 const modeCtl=gui.add(state,'mode',['translate','rotate']).name('gizmo').onChange(m=>transform.setMode(m));
 // The scene's scale: metres per export unit (COLMAP exports are unscaled). Everything in the file stays in export units.
 gui.add({mpu:loaded.metresPerUnit},'mpu',.01,10,.001).name('metres per unit').onChange(v=>loaded.setScale(v));
 gui.add({download(){download(loaded,jointed)}},'download').name('Download scene.json');

 transform.addEventListener('dragging-changed',e=>{state.dragging=e.value;controls.enabled=!e.value});
 transform.addEventListener('objectChange',()=>{
  const part=state.selected,r=part&&rows.get(part);if(!r)return;
  part.joint.configure({pivot:part.joint.kind===SLIDER?part.joint.pivot:r.helper.position,axis:new THREE.Vector3(0,1,0).applyQuaternion(r.helper.quaternion)});
  part.joint.applyPose();
 });

 function refreshVisibility(){
  gui.domElement.style.display=state.enabled?'':'none';
  for(const [p,r] of rows)r.helper.visible=state.enabled;
  const on=state.enabled&&!!state.selected;
  gizmo.visible=on;transform.enabled=on;
  if(!on)transform.detach();else if(state.selected&&transform.object!==rows.get(state.selected).helper)transform.attach(rows.get(state.selected).helper);
 }
 refreshVisibility();
 if(state.selected)select(state.selected);
 refreshVisibility();

 return {
  get enabled(){return state.enabled},
  get selected(){return state.selected},
  setEnabled(v){state.enabled=!!v;refreshVisibility();onEnabled(state.enabled)},
  select,
  // True while the pointer is on the gizmo, so the viewer does not also grab the part behind it.
  gizmoActive(){return state.enabled&&(state.dragging||transform.axis!==null)},
  // Keep the scrub slider following the part while it moves.
  update(){
   if(!state.enabled)return;
   // The pivot markers are drawn at a fixed real size, whatever the scale of the root they sit in.
   for(const r of rows.values())r.helper.scale.setScalar(1/root.scale.x);
   for(const [p,r] of rows){const u=p.joint.kind===HINGE?1/DEG:1;r.s.q=p.joint.q*u;if(r.s.locked!==p.joint.locked){r.s.locked=p.joint.locked;r.ctl.locked.updateDisplay()}}
   const r=state.selected&&rows.get(state.selected);if(r)r.scrub.updateDisplay();
  },
  exportScene:()=>buildExport(loaded,jointed),
  dispose(){transform.dispose();gui.destroy();for(const r of rows.values())r.helper.removeFromParent()},
 };
}

export function buildExport(loaded,jointed=loaded.parts.filter(p=>p.joint)){
 const out=JSON.parse(JSON.stringify(loaded.data));
 if(loaded.metresPerUnit!==1||loaded.data.metresPerUnit!==undefined)out.metresPerUnit=loaded.metresPerUnit;
 for(const part of jointed){
  const i=loaded.parts.indexOf(part);
  out.parts[i].joint=jointToJson(part.joint,loaded.data.parts[i].joint);
 }
 return out;
}

function download(loaded,jointed){
 const blob=new Blob([JSON.stringify(buildExport(loaded,jointed),null,2)+'\n'],{type:'application/json'});
 const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='scene.json';
 document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(a.href),2000);
}
