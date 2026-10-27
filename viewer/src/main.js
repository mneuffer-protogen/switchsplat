// SwitchSplat viewer: renderer, Spark, the fixed-step joint loop, desktop and WebXR input.
//   ?scene=<url of the export directory>   default: the scene list's default (scenes/index.json), else scenes/synthetic/
//   ?scale=<metres per unit>                override scene.json metresPerUnit (the world is in metres)
//   ?edit                                   start in edit mode
//   ?sh=<0..3>                              cap view-dependent colour (spherical harmonics) degree (sh=1 for a Quest)
//   ?cam=px,py,pz,tx,ty,tz                  starting camera (world coordinates)
//   ?xr=none | x,y,z                        where the scene stands in VR (default: scene.json view/floor, else in front of you)
//   ?perf                                   expose window.switchsplatPerf {start(),stop()}: Spark update/sort rates while parts move
//   ?emulate                                dev server only: install the IWER Quest 3 emulator (no headset needed)
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {SparkRenderer} from '@sparkjsdev/spark';
import {loadScene} from './scene-loader.js';
import {FixedStepper} from './joint.js';
import {createDesktopInput,toggle} from './desktop-input.js';
import {createXrInput} from './xr-input.js';
import {createPersistence} from './persist.js';
import {createEditMode} from './edit-mode.js';
import {createHud} from './hud.js';
import {computeSpawn,viewInRootFrame,floorHeight} from './spawn.js';
import {createLocomotion} from './locomotion.js';
import {createPerf} from './perf.js';
import {parseSceneIndex,initialScene,pickerOptions,matchEntry,heavyHint,sceneDir,sceneUrl} from './scenes.js';

const $=id=>document.getElementById(id);
const params=new URLSearchParams(location.search);

const mb=n=>`${(n/1048576).toFixed(n<10485760?1:0)} MB`;
// One row per file being loaded: name, bytes received of total, a bar. Updated at most every 100 ms, so a fast
// download of a big PLY does not spend the frame budget on the DOM.
function createLoadList(el){
 const rows=new Map();let timer=0;
 const paint=()=>{
  timer=0;
  for(const r of rows.values()){
   const {stage,loaded,total}=r.state,pct=total>0?Math.min(1,loaded/total):stage==='done'?1:0;
   r.li.className=stage==='download'?'':stage;
   r.amount.textContent=stage==='failed'?'not loaded':stage==='done'?mb(loaded):stage==='decode'?`${mb(loaded)}, decoding`:total>0?`${mb(loaded)} / ${mb(total)}  ${Math.round(pct*100)}%`:mb(loaded);
   r.fill.style.width=`${(stage==='decode'||stage==='done'?1:pct)*100}%`;
  }
 };
 return {
  update(p){
   let r=rows.get(p.url);
   if(!r){
    const li=document.createElement('li'),file=document.createElement('span'),amount=document.createElement('span'),track=document.createElement('div'),fill=document.createElement('div');
    file.className='file';file.textContent=p.name;amount.className='amount';track.className='track';track.append(fill);li.append(file,amount,track);el.append(li);
    r={li,amount,fill,state:p};rows.set(p.url,r);
   }
   r.state=p;
   // Stage changes show at once; byte counts are throttled.
   if(p.stage!=='download'){clearTimeout(timer);paint()}
   else if(!timer){timer=setTimeout(paint,100)}
  },
  clear(){clearTimeout(timer);rows.clear();el.textContent=''},
 };
}

async function installEmulator(){
 // Meta's Immersive Web Emulation Runtime: a Quest 3 and its two controllers in a desktop browser.
 const {XRDevice,metaQuest3}=await import('iwer');
 const device=new XRDevice(metaQuest3,{stereoEnabled:false});
 device.installRuntime();device.position.set(0,1.6,0);
 device.controllers.left.position.set(-.25,1.1,-.3);device.controllers.right.position.set(.25,1.1,-.3);
 window.xrDevice=device;
}

async function main(){
 if(import.meta.env.DEV&&params.has('emulate'))await installEmulator();

 const canvas=$('view'),status=$('status'),loadingText=$('loading-text'),loadList=createLoadList($('load-files'));
 const renderer=new THREE.WebGLRenderer({canvas,antialias:false,powerPreference:'high-performance'});
 renderer.setPixelRatio(Math.min(devicePixelRatio,2));
 renderer.setClearColor(0x101114);
 renderer.xr.enabled=true;renderer.xr.setReferenceSpaceType('local-floor');
 const scene=new THREE.Scene(),world=new THREE.Group();world.name='world';scene.add(world);
 const camera=new THREE.PerspectiveCamera(60,1,.01,500);scene.add(camera);
 // Spark draws every SplatMesh in the scene as one sorted pass.
 const spark=new SparkRenderer({renderer});scene.add(spark);
 const resize=()=>{if(renderer.xr.isPresenting)return;renderer.setSize(innerWidth,innerHeight,false);camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix()};
 addEventListener('resize',resize);resize();

 const maxSh=params.has('sh')?Math.min(3,Math.max(0,parseInt(params.get('sh'),10)||0)):null;
 const base=location.href;

 // ---- scene list (scenes/index.json, optional) ----
 const index=await fetch(`${import.meta.env.BASE_URL}scenes/index.json`).then(r=>r.ok?r.json():null).then(parseSceneIndex).catch(()=>null);
 const init=initialScene(index,params.get('scene'),base);
 let customParam=init.custom?init.param:null;
 const select=$('scene-select');
 let pickerOpts=[];
 // Rebuild the dropdown (the list, plus a "Custom: <url>" entry for a scene that is not in it) and select the current scene.
 const syncPicker=currentParam=>{
  pickerOpts=pickerOptions(index,customParam??currentParam,base);
  select.textContent='';
  for(const o of pickerOpts){
   const el=document.createElement('option'),hint=heavyHint(o.entry);
   el.value=o.value;el.textContent=hint?`${o.label} (${hint})`:o.label;el.title=o.entry?.description??o.param;select.append(el);
  }
  $('scene-picker').hidden=!pickerOpts.length;
  const m=matchEntry(index,currentParam,base);
  select.value=m?m.id:'custom';
  $('scene-desc').textContent=m?.description??'';
 };

 // ---- per-scene state: everything that belongs to one loaded scene and is torn down on a switch ----
 let cur=null,loadCtl=null,pending=null,placed=false;
 const flush=()=>cur?.persist.flush();
 addEventListener('pagehide',flush);document.addEventListener('visibilitychange',()=>{if(document.hidden)flush()});
 const controls=new OrbitControls(camera,canvas);
 controls.enableDamping=true;controls.dampingFactor=.12;controls.screenSpacePanning=true;
 const editToggle=$('edit-toggle');
 editToggle.addEventListener('change',()=>cur?.edit.setEnabled(editToggle.checked));
 $('reset').addEventListener('click',()=>{for(const j of cur?.joints??[])j.animateTo(j.closedQ,{style:'swing'})});
 const tune=undefined;// Alyx's defaults: constrainedFollow .35, maxSpeed 4, maxForce 450
 const vrButton=$('enter-vr');

 // Where the scene stands in VR (see ?xr=): re-run on a scene switch during a session.
 const place=()=>{
  const spec=params.get('xr'),{loaded,parts}=cur;
  world.position.set(0,0,0);world.quaternion.identity();placed=false;
  if(spec==='none')return;
  const b=loaded.worldBounds(),size=b.getSize(new THREE.Vector3()),c=b.getCenter(new THREE.Vector3());
  const explicit=spec?.split(',').map(Number);
  const spawn=computeSpawn({rootQuaternion:loaded.root.quaternion,rootScale:loaded.metresPerUnit,floor:loaded.floor,view:loaded.view});
  if(explicit?.length===3&&explicit.every(Number.isFinite))world.position.set(...explicit);
  // A real scene says where the floor is and where to stand: floor at y = 0, you at the view's position, facing its target.
  else if(spawn){world.position.copy(spawn.position);world.quaternion.copy(spawn.quaternion)}
  // A small scene (a cabinet, a door) stands with its movable parts 0.9 m in front of you and its floor at y = 0;
  // a room puts you in the middle of it.
  else{
   if(Math.max(size.x,size.z)>5)world.position.set(-c.x,-b.min.y,-c.z);
   else{
    const pb=new THREE.Box3();for(const p of parts)if(p.proxy)pb.expandByObject(p.proxy);
    const pc=pb.isEmpty()?c:pb.getCenter(new THREE.Vector3());
    world.position.set(-pc.x,-b.min.y,-.9-pc.z);
   }
   // A floor without a view still fixes the height.
   const fy=floorHeight(loaded.floor,loaded.root.quaternion,loaded.metresPerUnit);if(fy!==null)world.position.y=-fy;
  }
  placed=true;
 };

 function activate(loaded,param,{useCam}){
  world.add(loaded.root);world.visible=true;
  $('scene-name').textContent=loaded.name;
  const parts=loaded.parts,jointList=parts.filter(p=>p.joint).map(p=>p.joint);
  // Where each part was left last time (per scene: the key is the scene's URL).
  const persist=createPersistence(loaded.key);
  for(const p of parts)if(p.joint)persist.watch(p.id,p.joint);
  for(const j of jointList)j.applyPose();
  loaded.updateFades();
  // Camera: a view of the whole scene, or ?cam= (first load only).
  const wb=loaded.worldBounds(),centre=wb.getCenter(new THREE.Vector3()),radius=Math.max(.3,wb.getSize(new THREE.Vector3()).length()/2);
  const cam=useCam?params.get('cam')?.split(',').map(Number):null;
  // scene.json `view` (export coordinates, turned by the root's up rotation) is the default camera.
  const view=viewInRootFrame(loaded.view,loaded.root.quaternion,loaded.metresPerUnit);
  if(cam?.length===6&&cam.every(Number.isFinite)){camera.position.set(cam[0],cam[1],cam[2]);controls.target.set(cam[3],cam[4],cam[5])}
  else if(view){camera.position.copy(view.position);controls.target.copy(view.target)}
  else{camera.position.copy(centre).add(new THREE.Vector3(.35,.3,1).normalize().multiplyScalar(radius*1.6));controls.target.copy(centre)}
  controls.minDistance=.05;controls.maxDistance=Math.max(20,radius*10);controls.enabled=!renderer.xr.isPresenting;controls.update();
  const state={loaded,param,parts,joints:jointList,persist,stepper:new FixedStepper()};
  state.edit=createEditMode({scene,loaded,camera,renderer,controls});
  state.desktop=createDesktopInput({canvas,camera,controls,root:loaded.root,getParts:()=>parts,tune,
   isBlocked:()=>state.edit.gizmoActive()||renderer.xr.isPresenting,onSelect:p=>{if(state.edit.enabled)state.edit.select(p)},outlineBase:()=>state.edit.enabled?.3:0});
  state.xr=createXrInput({renderer,scene,root:loaded.root,getParts:()=>parts,tune,onSelect:p=>{if(state.edit.enabled)state.edit.select(p)}});
  state.hud=createHud({parts});
  state.edit.setEnabled(editToggle.checked);
  cur=state;
  if(renderer.xr.isPresenting)place();
  status.textContent=`${parts.filter(p=>p.interactive).length} movable part(s)`;
  state.hud.update();
  window.switchsplat={THREE,renderer,scene,world,camera,controls,spark,loaded,parts,joints:jointList,persist,edit:state.edit,desktop:state.desktop,xr:state.xr,locomotion,toggle:id=>toggle(parts.find(p=>p.id===id))};
 }

 // Everything the current scene owns, in the order that keeps a held part from being driven after it is gone.
 function teardown(){
  if(!cur)return;
  const s=cur;cur=null;
  s.persist.flush();
  s.xr.releaseAll();s.xr.dispose();
  s.desktop.dispose();controls.enabled=!renderer.xr.isPresenting;canvas.style.cursor='';
  s.edit.dispose();
  s.loaded.dispose();
  // Spark's idle accumulators keep the previous mapping (and with it the disposed SplatMeshes and their splat
  // arrays) until they are next used: drop those so a switch does not leave the old scene in memory.
  for(const a of spark.accumulators)a.mapping=[];
 }

 // ---- loading overlay ----
 // After a cancelled or failed switch the address bar goes back to the scene that is still showing.
 const restoreUrl=()=>{if(cur&&sceneDir(new URLSearchParams(location.search).get('scene')??'',base)!==sceneDir(cur.param,base))history.replaceState(null,'',sceneUrl(location.href,cur.param))};
 const overlay=$('loading'),retryBtn=$('load-retry'),cancelBtn=$('load-cancel');
 const showOverlay=(text,{error=false,retry=false,cancel=false}={})=>{
  overlay.classList.remove('done');loadingText.textContent=text;loadingText.classList.toggle('error',error);
  retryBtn.hidden=!retry;cancelBtn.hidden=!cancel;$('load-actions').hidden=!retry&&!cancel;
 };
 const hideOverlay=()=>{loadList.clear();overlay.classList.add('done')};

 // Load a scene and swap it in. A new request cancels the one in flight (its downloads abort); a failed one
 // leaves the current scene in place.
 async function switchTo(param,{entry=null,initial=false,pushUrl=false}={}){
  loadCtl?.abort();
  const ctl=loadCtl=new AbortController();
  pending={param,entry,initial,pushUrl};
  loadList.clear();
  showOverlay(`Loading ${entry?.name??param}`,{cancel:!!cur});
  if(cur)world.visible=false;
  let loaded;
  try{
   loaded=await loadScene(param,{onStatus:t=>{if(loadCtl===ctl)loadingText.textContent=t},onProgress:p=>{if(loadCtl===ctl)loadList.update(p)},maxSh,signal:ctl.signal,scale:initial?params.get('scale'):null});
  }catch(e){
   if(loadCtl!==ctl)return;// superseded: the newer request owns the overlay
   if(ctl.signal.aborted){// cancelled by the user: back to the scene that is still loaded
    loadCtl=null;world.visible=true;hideOverlay();if(cur){syncPicker(cur.param);restoreUrl()}return;
   }
   console.error(e);loadCtl=null;
   showOverlay(e.message??String(e),{error:true,retry:true,cancel:!!cur});
   return;
  }
  if(loadCtl!==ctl){loaded.dispose();return}
  loadCtl=null;
  teardown();
  activate(loaded,param,{useCam:initial});
  if(pushUrl)history.pushState(null,'',sceneUrl(location.href,param,{keep:false}));
  if(!matchEntry(index,param,base))customParam=param;
  syncPicker(param);
  hideOverlay();
 }
 retryBtn.addEventListener('click',()=>{if(pending)switchTo(pending.param,pending)});
 cancelBtn.addEventListener('click',()=>{
  if(loadCtl)loadCtl.abort();// the pending switchTo restores the old scene
  else if(cur){world.visible=true;hideOverlay();syncPicker(cur.param);restoreUrl()}
 });
 select.addEventListener('change',()=>{
  const o=pickerOpts.find(x=>x.value===select.value);
  if(!o||(cur&&!loadCtl&&sceneDir(cur.param,base)===sceneDir(o.param,base)))return;
  switchTo(o.param,{entry:o.entry,pushUrl:true});
 });
 addEventListener('popstate',()=>{
  const t=initialScene(index,new URLSearchParams(location.search).get('scene'),base);
  const param=t.param??`${import.meta.env.BASE_URL}scenes/synthetic/`;
  if(cur&&!loadCtl&&sceneDir(cur.param,base)===sceneDir(param,base))return;
  if(t.custom)customParam=param;
  switchTo(param,{entry:t.entry});
 });

 // ---- VR ----
 let session=null;
 renderer.xr.addEventListener('sessionstart',()=>{cur?.desktop.update(0);controls.enabled=false;if(cur)place();vrButton.textContent='Exit VR'});
 renderer.xr.addEventListener('sessionend',()=>{cur?.xr.releaseAll();session=null;if(placed){world.position.set(0,0,0);world.quaternion.identity()}placed=false;controls.enabled=true;vrButton.textContent='Enter VR';resize()});
 const supported=navigator.xr?await navigator.xr.isSessionSupported('immersive-vr').catch(()=>false):false;
 if(supported){vrButton.disabled=false}
 else{vrButton.title=navigator.xr?'Immersive VR is not available on this device':'WebXR needs a secure context: open the HTTPS dev URL on the headset'}
 vrButton.addEventListener('click',async()=>{
  try{
   if(session){await session.end();return}
   session=await navigator.xr.requestSession('immersive-vr',{optionalFeatures:['local-floor','bounded-floor','hand-tracking']});
   session.addEventListener('end',()=>{session=null});
   await renderer.xr.setSession(session);
  }catch(e){console.error(e);status.textContent=`VR: ${e.message}`;session=null}
 });

 // Left stick walks, right stick snap-turns (not while a hand holds a part).
 const locomotion=createLocomotion({renderer,world,onTurn:()=>cur?.xr.noteJump()});

 // ---- loop ----
 let last=performance.now(),hudClock=0;
 renderer.setAnimationLoop(now=>{
  const dt=Math.min(.1,Math.max(0,(now-last)/1000))||1/60;last=now;
  const presenting=renderer.xr.isPresenting;
  if(cur){
   const {loaded,desktop,xr,hud,edit,stepper,joints:js}=cur;
   if(presenting)locomotion.update(dt,{blocked:xr.active});
   loaded.root.updateWorldMatrix(true,false);
   if(presenting)xr.update(dt);else desktop.update(dt);
   const left=stepper.run(dt,h=>{desktop.step(h);xr.step(h);for(const j of js)j.step(h)});
   for(const j of js)j.applyPose(left);
   loaded.updateFades();
   hudClock+=dt;if(hudClock>.08){hudClock=0;hud.update();edit.update()}
  }
  if(!presenting)controls.update();
  renderer.render(scene,camera);
 });

 if(params.has('perf'))window.switchsplatPerf=createPerf({spark,getParts:()=>cur?.parts??[]});
 await switchTo(init.param??`${import.meta.env.BASE_URL}scenes/synthetic/`,{entry:init.entry,initial:true});
}

main().catch(e=>{console.error(e);const t=$('loading-text');if(t){t.textContent=String(e.message??e);t.classList.add('error')}});
