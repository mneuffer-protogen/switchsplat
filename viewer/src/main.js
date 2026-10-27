// SwitchSplat viewer: renderer, Spark, the fixed-step joint loop, desktop and WebXR input.
//   ?scene=<url of the export directory>   default scenes/synthetic/
//   ?edit                                   start in edit mode
//   ?sh=<0..3>                              cap view-dependent colour (spherical harmonics) degree
//   ?cam=px,py,pz,tx,ty,tz                  starting camera (world coordinates)
//   ?xr=none | x,y,z                        where the scene stands in VR (default: in front of you, on the floor)
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

const $=id=>document.getElementById(id);
const params=new URLSearchParams(location.search);

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

 const canvas=$('view'),status=$('status'),loadingText=$('loading-text');
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

 const sceneParam=params.get('scene')??`${import.meta.env.BASE_URL}scenes/synthetic/`;
 const maxSh=params.has('sh')?Math.min(3,Math.max(0,parseInt(params.get('sh'),10)||0)):null;
 let loaded;
 try{loaded=await loadScene(sceneParam,{onStatus:t=>{loadingText.textContent=t},maxSh})}
 catch(e){console.error(e);loadingText.textContent=e.message;loadingText.classList.add('error');return}
 world.add(loaded.root);
 $('scene-name').textContent=loaded.name;
 const parts=loaded.parts,joints=parts.filter(p=>p.joint).map(p=>p.joint);

 // Where each part was left last time.
 const persist=createPersistence(loaded.key);
 for(const p of parts)if(p.joint)persist.watch(p.id,p.joint);
 const flush=()=>persist.flush();
 addEventListener('pagehide',flush);document.addEventListener('visibilitychange',()=>{if(document.hidden)flush()});
 for(const j of joints)j.applyPose();
 loaded.updateFades();

 // Camera: a view of the whole scene, or ?cam=.
 const wb=loaded.worldBounds(),centre=wb.getCenter(new THREE.Vector3()),radius=Math.max(.3,wb.getSize(new THREE.Vector3()).length()/2);
 const controls=new OrbitControls(camera,canvas);
 controls.enableDamping=true;controls.dampingFactor=.12;controls.screenSpacePanning=true;
 const cam=params.get('cam')?.split(',').map(Number);
 if(cam?.length===6&&cam.every(Number.isFinite)){camera.position.set(cam[0],cam[1],cam[2]);controls.target.set(cam[3],cam[4],cam[5])}
 else{camera.position.copy(centre).add(new THREE.Vector3(.35,.3,1).normalize().multiplyScalar(radius*1.6));controls.target.copy(centre)}
 controls.minDistance=.05;controls.maxDistance=Math.max(20,radius*10);controls.update();

 const tune=undefined;// Alyx's defaults: constrainedFollow .35, maxSpeed 4, maxForce 450
 const edit=createEditMode({scene,loaded,camera,renderer,controls});
 const desktop=createDesktopInput({canvas,camera,controls,root:loaded.root,getParts:()=>parts,tune,
  isBlocked:()=>edit.gizmoActive()||renderer.xr.isPresenting,onSelect:p=>{if(edit.enabled)edit.select(p)},outlineBase:()=>edit.enabled?.3:0});
 const xr=createXrInput({renderer,scene,root:loaded.root,getParts:()=>parts,tune,onSelect:p=>{if(edit.enabled)edit.select(p)}});
 const hud=createHud({parts});

 const editToggle=$('edit-toggle');
 editToggle.addEventListener('change',()=>edit.setEnabled(editToggle.checked));
 if(params.has('edit')){editToggle.checked=true;edit.setEnabled(true)}
 $('reset').addEventListener('click',()=>{for(const j of joints)j.animateTo(j.closedQ)});

 // ---- VR ----
 const vrButton=$('enter-vr');let session=null,placed=false;
 const place=()=>{
  const spec=params.get('xr');
  if(spec==='none')return;
  const b=loaded.worldBounds(),size=b.getSize(new THREE.Vector3()),c=b.getCenter(new THREE.Vector3());
  const explicit=spec?.split(',').map(Number);
  if(explicit?.length===3&&explicit.every(Number.isFinite))world.position.set(...explicit);
  // A small scene (a cabinet, a door) stands with its movable parts 0.9 m in front of you and its floor at y = 0;
  // a room puts you in the middle of it.
  else if(Math.max(size.x,size.z)>5)world.position.set(-c.x,-b.min.y,-c.z);
  else{
   const pb=new THREE.Box3();for(const p of parts)if(p.proxy)pb.expandByObject(p.proxy);
   const pc=pb.isEmpty()?c:pb.getCenter(new THREE.Vector3());
   world.position.set(-pc.x,-b.min.y,-.9-pc.z);
  }
  placed=true;
 };
 renderer.xr.addEventListener('sessionstart',()=>{desktop.update(0);controls.enabled=false;place();vrButton.textContent='Exit VR'});
 renderer.xr.addEventListener('sessionend',()=>{xr.releaseAll();session=null;if(placed)world.position.set(0,0,0);placed=false;controls.enabled=true;vrButton.textContent='Enter VR';resize()});
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

 // ---- loop ----
 const stepper=new FixedStepper();let last=performance.now(),hudClock=0;
 renderer.setAnimationLoop(now=>{
  const dt=Math.min(.1,Math.max(0,(now-last)/1000))||1/60;last=now;
  const presenting=renderer.xr.isPresenting;
  loaded.root.updateWorldMatrix(true,false);
  if(presenting)xr.update(dt);else desktop.update(dt);
  const left=stepper.run(dt,h=>{desktop.step(h);xr.step(h);for(const j of joints)j.step(h)});
  for(const j of joints)j.applyPose(left);
  loaded.updateFades();
  hudClock+=dt;if(hudClock>.08){hudClock=0;hud.update();edit.update()}
  if(!presenting)controls.update();
  renderer.render(scene,camera);
 });

 loadingText.textContent='';$('loading').classList.add('done');
 status.textContent=`${parts.filter(p=>p.interactive).length} movable part(s)`;
 hud.update();
 // For scripted checks and the browser console.
 window.switchsplat={THREE,renderer,scene,world,camera,controls,spark,loaded,parts,joints,persist,edit,desktop,xr,toggle:id=>toggle(parts.find(p=>p.id===id))};
}

main().catch(e=>{console.error(e);const t=$('loading-text');if(t){t.textContent=String(e.message??e);t.classList.add('error')}});
