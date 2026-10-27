// Loads a SwitchSplat export directory (scene.json + PLYs, see articulation-spec.md) into a three.js
// Group tree:
//
//   root (rotated so scene.up -> +Y)
//    |- static SplatMesh
//    |- per part: Group (posed by its JointedPart) > part SplatMesh + OBB proxy (+ hover outline)
//    '- per part: reveal / vanish SplatMeshes (not moved; opacity follows the joint)
//
// Spark composes every SplatMesh in the scene into one sorted draw, whatever Object3D transforms they
// sit under, so a door swinging through the static room still sorts correctly against it.
import * as THREE from 'three';
import {SplatMesh,SplatFileType} from '@sparkjsdev/spark';
import {plyCentres,plyPoints} from './ply-centres.js';
import {buildContacts} from './contact.js';
import {fitObb,trimmedBounds} from './obb.js';
import {JointedPart,HINGE,SLIDER} from './joint.js';
import {DEG} from './math.js';

const DEFAULTS={friction:1.5,frictionTorque:0,mass:2,latchDeg:3};

// "?scene=" is the URL of the export directory (trailing slash optional) or of its scene.json.
export function resolveSceneUrl(param,base=globalThis.location?.href??'http://localhost/'){
 const abs=new URL(param,base);
 if(abs.pathname.endsWith('.json'))return {dir:new URL('./',abs).href,json:abs.href,key:abs.href};
 if(!abs.pathname.endsWith('/'))abs.pathname+='/';
 return {dir:abs.href,json:new URL('scene.json',abs).href,key:abs.href};
}

// scene.json joint -> JointedPart settings. Hinge angles are degrees in the file, radians inside.
export function jointOptions(j){
 const hinge=j.kind!=='slider',u=hinge?DEG:1;
 const min=(j.min??0)*u;let max=(j.max??j.open??0)*u;
 if(!(max>min))max=min+(hinge?90*DEG:.3);
 const open=Math.min(Math.max((j.open??j.max??0)*u,min),max)||max;
 // A slider's latch is in metres; the hinge default of 3 degrees would pin a drawer shut.
 const latch=hinge?(j.latch??DEFAULTS.latchDeg)*DEG:Math.min(j.latch??0,.1*(max-min));
 return {kind:hinge?HINGE:SLIDER,axis:new THREE.Vector3(...(j.axis??[0,1,0])),pivot:new THREE.Vector3(...(j.pivot??[0,0,0])),min,max,open,
  friction:j.friction??DEFAULTS.friction,frictionTorque:j.frictionTorque??DEFAULTS.frictionTorque,latch,mass:j.mass??DEFAULTS.mass,locked:j.locked===true};
}
const round=(v,d=5)=>Math.round(v*10**d)/10**d;
// A JointedPart back into scene.json terms (for the edit panel's download).
export function jointToJson(joint,original={}){
 const hinge=joint.kind===HINGE,u=hinge?1/DEG:1;
 return {...original,kind:joint.kind,axis:joint.axis.toArray().map(v=>round(v)),pivot:joint.pivot.toArray().map(v=>round(v)),
  min:round(joint.min*u),max:round(joint.max*u),open:round(joint.open*u),friction:round(joint.friction),frictionTorque:round(joint.frictionTorque),
  latch:round(joint.latch*u),mass:round(joint.bodyMass),...(joint.locked||original.locked!==undefined?{locked:joint.locked}:{})};
}

// Download a file with progress: onProgress(received, total) per chunk, total 0 when the server does not say
// (or compresses, so content-length is not the size of what arrives). The bytes land in one buffer that
// is sized from content-length when known, so a 150 MB PLY is not held twice.
async function fetchBuffer(url,onProgress=()=>{}){
 const r=await fetch(url);
 if(!r.ok)throw new Error(`${r.status} ${r.statusText} for ${url}`);
 const length=Number(r.headers.get('content-length'))||0,total=r.headers.get('content-encoding')?0:length;
 if(!r.body?.getReader){const b=await r.arrayBuffer();onProgress(b.byteLength,b.byteLength);return b}
 const reader=r.body.getReader();
 let out=new Uint8Array(total||1<<20),received=0;
 onProgress(0,total);
 for(;;){
  const {done,value}=await reader.read();if(done)break;
  if(received+value.length>out.length){const grown=new Uint8Array(Math.max(out.length*2,received+value.length));grown.set(out.subarray(0,received));out=grown}
  out.set(value,received);received+=value.length;
  onProgress(received,total);
 }
 onProgress(received,received);
 return out.length===received?out.buffer:out.buffer.slice(0,received);
}

// At most `n` PLYs download and decode at once: all of them in parallel is the fastest way to run a phone out
// of memory when each is 100+ MB. Jobs start in the order they are asked for (the static scene first).
function createGate(n){
 let active=0;const waiting=[];
 const next=()=>{if(active<n&&waiting.length){active++;waiting.shift()()}};
 return async fn=>{
  await new Promise(res=>{waiting.push(res);next()});
  try{return await fn()}finally{active--;next()}
 };
}

// Read a PLY's centres, then hand its bytes to Spark. The centres come first: Spark may take ownership of
// the buffer. If the file is not a plain binary PLY the centres come from Spark's decoded splats instead.
// Nothing here walks the splats per frame; only the (strided) centres are read, once.
async function loadSplats(url,{wantCentres,wantAttrs=false,maxSh,max=60000,onProgress=()=>{}}){
 const base={url,name:url.split('/').pop()};
 onProgress({...base,stage:'download',loaded:0,total:0});
 const buffer=await fetchBuffer(url,(loaded,total)=>onProgress({...base,stage:'download',loaded,total}));
 onProgress({...base,stage:'decode',loaded:buffer.byteLength,total:buffer.byteLength});
 let centres=null,parsed=null;
 if(wantCentres){try{if(wantAttrs){parsed=plyPoints(buffer,{max});centres=parsed.centres}else centres=plyCentres(buffer,{max})}catch(e){console.warn(`${url}: ${e.message}; reading centres through Spark`)}}
 const mesh=new SplatMesh({fileBytes:buffer,fileType:SplatFileType.PLY,raycastable:false});
 if(maxSh!=null)mesh.maxSh=maxSh;
 await mesh.initialized;
 if(wantCentres&&!centres){
  const all=[];mesh.forEachSplat((i,c)=>{all.push(c.x,c.y,c.z)});
  const step=Math.max(1,Math.ceil(all.length/3/max)),out=[];
  for(let i=0;i<all.length/3;i+=step)out.push(all[i*3],all[i*3+1],all[i*3+2]);
  centres=new Float32Array(out);
 }
 onProgress({...base,stage:'done',loaded:buffer.byteLength,total:buffer.byteLength});
 return {mesh,centres,parsed};
}
const optional=async(url,opts,what)=>{try{return await loadSplats(url,opts)}catch(e){console.warn(`${what} not loaded: ${e.message}`);opts.onProgress?.({url,name:url.split('/').pop(),stage:'failed',loaded:0,total:0});return null}};

const BOX=new THREE.BoxGeometry(1,1,1),EDGES=new THREE.EdgesGeometry(BOX);

// onProgress({url,name,stage:'download'|'decode'|'done'|'failed',loaded,total}) per file, as bytes arrive.
export async function loadScene(sceneParam,{onStatus=()=>{},onProgress=()=>{},maxSh=null,concurrency=4}={}){
 const {dir,json,key}=resolveSceneUrl(sceneParam);
 onStatus('Loading scene.json');
 let res;try{res=await fetch(json)}catch(e){throw new Error(`Could not fetch ${json}: ${e.message}`)}
 if(!res.ok)throw new Error(`Could not load ${json} (${res.status}). Pass ?scene=<export directory>.`);
 const data=await res.json();
 const file=name=>new URL(name,dir).href;

 const root=new THREE.Group();root.name=data.name??'scene';
 const up=new THREE.Vector3(...(data.up??[0,1,0])).normalize();
 root.quaternion.setFromUnitVectors(up,new THREE.Vector3(0,1,0));
 const bounds=new THREE.Box3();let loaded=0,total=1+(data.parts?.length??0);
 const tick=what=>onStatus(`Loaded ${what} (${++loaded}/${total})`);
 const gate=createGate(concurrency);

 // Everything loads in parallel (a few files at a time); failures of optional files only warn.
 const staticJob=data.static?gate(()=>loadSplats(file(data.static),{wantCentres:true,maxSh,max:20000,onProgress})).then(r=>{tick(data.static);return r}):Promise.resolve(null);
 const partJobs=(data.parts??[]).map(async p=>{
  const hasJoint=!!p.joint,f=p.files??{};
  // Without a joint the slot is just its closed state, which is the whole base set.
  const main=hasJoint?(f.part??f.base):(f.base??f.part);
  const [body,reveal,vanish]=await Promise.all([
   main?gate(()=>optional(file(main),{wantCentres:true,wantAttrs:true,max:100000,maxSh,onProgress},`${p.id} ${main}`)):null,
   hasJoint&&f.reveal?gate(()=>optional(file(f.reveal),{maxSh,onProgress},`${p.id} reveal`)):null,
   hasJoint&&f.vanish?gate(()=>optional(file(f.vanish),{maxSh,onProgress},`${p.id} vanish`)):null,
  ]);
  tick(p.name??p.id);
  return {p,body,reveal,vanish};
 });
 const [staticRes,partRes]=await Promise.all([staticJob,Promise.all(partJobs)]);

 let staticMesh=null;
 if(staticRes){
  staticMesh=staticRes.mesh;staticMesh.name='static';root.add(staticMesh);
  if(staticRes.centres?.length)bounds.union(trimmedBounds(staticRes.centres,.005));
 }

 const parts=[];
 for(const {p,body,reveal,vanish} of partRes){
  const group=new THREE.Group();group.name=p.id;root.add(group);
  const part={id:p.id,name:p.name??p.id,data:p,group,mesh:body?.mesh??null,joint:null,proxy:null,outline:null,reveal:null,vanish:null,fit:p.joint?.fit??null,interactive:false,obb:null,contacts:null,merged:Array.isArray(p.merged)&&p.merged.length?p.merged:null};
  part.lowFit=part.fit?.inlierRatio!=null&&part.fit.inlierRatio<.6;
  if(body){body.mesh.name=`${p.id}:part`;group.add(body.mesh)}
  for(const [k,r] of [['reveal',reveal],['vanish',vanish]])if(r){r.mesh.name=`${p.id}:${k}`;root.add(r.mesh);part[k]=r.mesh}
  if(body?.centres?.length>=12){
   // The contact set (real surface splats) is what hands and the mouse touch; the box built from it is the outline.
   const contacts=buildContacts(body.parsed??{centres:body.centres});part.contacts=contacts;
   const obb=contacts.fitBox();part.obb=obb;
   if(obb){
    const proxy=new THREE.Mesh(BOX,new THREE.MeshBasicMaterial({visible:false}));
    proxy.scale.copy(obb.size);proxy.position.copy(obb.center);proxy.quaternion.copy(obb.quaternion);proxy.name=`${p.id}:proxy`;
    proxy.userData.part=part;
    const outline=new THREE.LineSegments(EDGES,new THREE.LineBasicMaterial({color:0x66ccff,transparent:true,opacity:0,depthTest:false,depthWrite:false}));
    outline.renderOrder=1000;outline.visible=false;outline.name=`${p.id}:outline`;proxy.add(outline);
    group.add(proxy);part.proxy=proxy;part.outline=outline;
    const half=obb.size.clone().multiplyScalar(.5);
    for(const sx of [-1,1])for(const sy of [-1,1])for(const sz of [-1,1])bounds.expandByPoint(new THREE.Vector3(half.x*sx,half.y*sy,half.z*sz).applyQuaternion(obb.quaternion).add(obb.center));
   }
  }
  if(p.joint&&body){
   part.joint=new JointedPart({object:group,...jointOptions(p.joint),box:part.obb??undefined});
   part.joint.contacts=part.contacts??null;
   part.interactive=!!part.proxy;
  }
  parts.push(part);
 }
 if(bounds.isEmpty())bounds.set(new THREE.Vector3(-1,0,-1),new THREE.Vector3(1,2,1));
 root.updateMatrixWorld(true);

 const scene={root,data,dir,key,name:data.name??'scene',staticMesh,parts,bounds,
  // Optional (spec addendum v1.1): the desktop camera and a point on the ground plane, in export coordinates.
  view:data.view?.position&&data.view?.target?data.view:null,floor:Array.isArray(data.floor)&&data.floor.length===3?data.floor:null,
  worldBounds:()=>{root.updateMatrixWorld(true);return bounds.clone().applyMatrix4(root.matrixWorld)},
  // The fade of the splats that appear (reveal) or disappear (vanish) as the part opens.
  updateFades(){
   for(const p of parts){
    if(!p.joint)continue;const f=p.joint.fraction;
    if(p.reveal)setFade(p.reveal,f);
    if(p.vanish)setFade(p.vanish,1-f);
   }
  },
  dispose(){root.traverse(o=>{if(o instanceof SplatMesh){try{o.dispose()}catch{}}});root.removeFromParent()}};
 scene.updateFades();
 return scene;
}

// Opacity multiplier on a whole SplatMesh; fully faded ones are hidden so Spark skips them.
function setFade(mesh,v){
 const o=Math.min(1,Math.max(0,v));
 if(mesh.opacity!==o)mesh.opacity=o;
 const show=o>.003;if(mesh.visible!==show)mesh.visible=show;
}
