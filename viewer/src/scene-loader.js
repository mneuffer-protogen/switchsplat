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
import {plyCentres} from './ply-centres.js';
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
  friction:j.friction??DEFAULTS.friction,frictionTorque:j.frictionTorque??DEFAULTS.frictionTorque,latch,mass:j.mass??DEFAULTS.mass};
}
const round=(v,d=5)=>Math.round(v*10**d)/10**d;
// A JointedPart back into scene.json terms (for the edit panel's download).
export function jointToJson(joint,original={}){
 const hinge=joint.kind===HINGE,u=hinge?1/DEG:1;
 return {...original,kind:joint.kind,axis:joint.axis.toArray().map(v=>round(v)),pivot:joint.pivot.toArray().map(v=>round(v)),
  min:round(joint.min*u),max:round(joint.max*u),open:round(joint.open*u),friction:round(joint.friction),frictionTorque:round(joint.frictionTorque),
  latch:round(joint.latch*u),mass:round(joint.bodyMass)};
}

async function fetchBuffer(url){
 const r=await fetch(url);
 if(!r.ok)throw new Error(`${r.status} ${r.statusText} for ${url}`);
 return r.arrayBuffer();
}

// Read a PLY's centres, then hand its bytes to Spark. The centres come first: Spark may take ownership of
// the buffer. If the file is not a plain binary PLY the centres come from Spark's decoded splats instead.
async function loadSplats(url,{wantCentres,maxSh,max=60000}){
 const buffer=await fetchBuffer(url);
 let centres=null;
 if(wantCentres){try{centres=plyCentres(buffer,{max})}catch(e){console.warn(`${url}: ${e.message}; reading centres through Spark`)}}
 const mesh=new SplatMesh({fileBytes:buffer,fileType:SplatFileType.PLY,raycastable:false});
 if(maxSh!=null)mesh.maxSh=maxSh;
 await mesh.initialized;
 if(wantCentres&&!centres){
  const all=[];mesh.forEachSplat((i,c)=>{all.push(c.x,c.y,c.z)});
  const step=Math.max(1,Math.ceil(all.length/3/max)),out=[];
  for(let i=0;i<all.length/3;i+=step)out.push(all[i*3],all[i*3+1],all[i*3+2]);
  centres=new Float32Array(out);
 }
 return {mesh,centres};
}
const optional=async(url,opts,what)=>{try{return await loadSplats(url,opts)}catch(e){console.warn(`${what} not loaded: ${e.message}`);return null}};

const BOX=new THREE.BoxGeometry(1,1,1),EDGES=new THREE.EdgesGeometry(BOX);

export async function loadScene(sceneParam,{onStatus=()=>{},maxSh=null}={}){
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

 // Everything loads in parallel; failures of optional files only warn.
 const staticJob=data.static?loadSplats(file(data.static),{wantCentres:true,maxSh,max:20000}).then(r=>{tick(data.static);return r}):Promise.resolve(null);
 const partJobs=(data.parts??[]).map(async p=>{
  const hasJoint=!!p.joint,f=p.files??{};
  // Without a joint the slot is just its closed state, which is the whole base set.
  const main=hasJoint?(f.part??f.base):(f.base??f.part);
  const [body,reveal,vanish]=await Promise.all([
   main?optional(file(main),{wantCentres:true,maxSh},`${p.id} ${main}`):null,
   hasJoint&&f.reveal?optional(file(f.reveal),{maxSh},`${p.id} reveal`):null,
   hasJoint&&f.vanish?optional(file(f.vanish),{maxSh},`${p.id} vanish`):null,
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
  const part={id:p.id,name:p.name??p.id,data:p,group,mesh:body?.mesh??null,joint:null,proxy:null,outline:null,reveal:null,vanish:null,fit:p.joint?.fit??null,interactive:false,obb:null};
  part.lowFit=part.fit?.inlierRatio!=null&&part.fit.inlierRatio<.6;
  if(body){body.mesh.name=`${p.id}:part`;group.add(body.mesh)}
  for(const [k,r] of [['reveal',reveal],['vanish',vanish]])if(r){r.mesh.name=`${p.id}:${k}`;root.add(r.mesh);part[k]=r.mesh}
  if(body?.centres?.length>=12){
   const obb=fitObb(body.centres);part.obb=obb;
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
   part.interactive=!!part.proxy;
  }
  parts.push(part);
 }
 if(bounds.isEmpty())bounds.set(new THREE.Vector3(-1,0,-1),new THREE.Vector3(1,2,1));
 root.updateMatrixWorld(true);

 const scene={root,data,dir,key,name:data.name??'scene',staticMesh,parts,bounds,
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
