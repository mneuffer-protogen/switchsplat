// The contact set of a part: its real surface splat centres (stray, faint and huge splats dropped), in the
// part's rest space, with a uniform grid for proximity queries. Grabbing, pushing and picking use these points,
// not the oriented box: on a sparse real scan the box is metres wide (floaters stretch it) and would grab or push
// from half a metre away. The box stays only as the hover outline and the joint's mass estimate.
//
// Everything here is O(points of one part) at load and O(1)-ish per query; nothing touches the static scene's splats.
import * as THREE from 'three';
import {eigenSym3,fitObb} from './obb.js';

export const GRAB_RADIUS=.07;       // a hand grabs a part when its grip is this close to a contact centre (m)
export const CONTACT_MARGIN=.025;   // a push touches a centre within hand radius + this
export const PICK_RADIUS=.04;       // a mouse ray hits a centre treated as a sphere of this radius
export const AIM_RADIUS=.05;        // a controller's pointing ray, a little more forgiving
export const FILTER={minOpacity:.15,maxScale:.15,neighbours:3,neighbourRadius:.08,cell:.1,maxPoints:8000,minPoints:12,illDefinedRatio:.35};

const sigmoid=x=>1/(1+Math.exp(-x));

// A uniform hash grid over a Float32Array of points: cell index lists by packed integer key.
export class PointGrid{
 constructor(points,cell=.1,indices=null){
  this.points=points;this.cell=cell;this.map=new Map();
  const n=indices?indices.length:points.length/3;
  for(let k=0;k<n;k++){
   const i=indices?indices[k]:k,key=this.key(this.c(points[i*3]),this.c(points[i*3+1]),this.c(points[i*3+2]));
   const l=this.map.get(key);if(l)l.push(i);else this.map.set(key,[i]);
  }
 }
 c(v){return Math.floor(v/this.cell)}
 key(ix,iy,iz){return ((ix+524288)*1048576+(iy+524288))*1048576+(iz+524288)}
 // Calls fn(i, dist2) for every point within r of (x,y,z).
 near(x,y,z,r,fn){
  const P=this.points,r2=r*r,x0=this.c(x-r),x1=this.c(x+r),y0=this.c(y-r),y1=this.c(y+r),z0=this.c(z-r),z1=this.c(z+r);
  for(let ix=x0;ix<=x1;ix++)for(let iy=y0;iy<=y1;iy++)for(let iz=z0;iz<=z1;iz++){
   const l=this.map.get(this.key(ix,iy,iz));if(!l)continue;
   for(const i of l){
    const dx=P[i*3]-x,dy=P[i*3+1]-y,dz=P[i*3+2]-z,d2=dx*dx+dy*dy+dz*dz;
    if(d2<=r2&&fn(i,d2)===false)return;
   }
  }
 }
}

// Drop the splats that are not surface: faint (sigmoid(opacity) < minOpacity), huge (largest scale > maxScale metres),
// and isolated (fewer than `neighbours` others within `neighbourRadius`). If that leaves too little the looser sets
// are used (never fewer points than a part can be grabbed by). Returns {points, stats}.
export function filterContactCentres(centres,{opacity=null,maxLogScale=null,...o}={}){
 const f={...FILTER,...o},n=Math.floor(centres.length/3);
 const stats={input:n,faint:0,huge:0,isolated:0,final:0,fallback:null};
 const keep=[];
 for(let i=0;i<n;i++){
  if(opacity&&sigmoid(opacity[i])<f.minOpacity){stats.faint++;continue}
  if(maxLogScale&&Math.exp(maxLogScale[i])>f.maxScale){stats.huge++;continue}
  keep.push(i);
 }
 let chosen=keep;
 if(keep.length>=f.minPoints){
  const grid=new PointGrid(centres,f.neighbourRadius,keep),dense=[];
  for(const i of keep){
   let count=0;
   grid.near(centres[i*3],centres[i*3+1],centres[i*3+2],f.neighbourRadius,j=>{if(j!==i&&++count>=f.neighbours)return false});
   if(count>=f.neighbours)dense.push(i);
  }
  stats.isolated=keep.length-dense.length;
  if(dense.length>=f.minPoints)chosen=dense;else stats.fallback='isolation';
 }else stats.fallback='opacity/scale';
 if(chosen.length<4){chosen=[...Array(n).keys()];stats.fallback='all'}
 // A dense part is thinned evenly: the point count bounds the ray pick.
 const stride=Math.max(1,Math.ceil(chosen.length/f.maxPoints)),pts=[];
 for(let k=0;k<chosen.length;k+=stride){const i=chosen[k];pts.push(centres[i*3],centres[i*3+1],centres[i*3+2])}
 stats.final=pts.length/3;
 return {points:new Float32Array(pts),stats};
}

export class ContactSet{
 constructor(points,{cell=FILTER.cell}={}){
  this.points=points;this.count=points.length/3;this.grid=new PointGrid(points,cell);
  // The thin axis: the eigenvector of the smallest variance (a door or hood is a sheet). `thinRatio` is smallest over
  // middle variance; above illDefinedRatio the cloud is not sheet-like and the direction hand-to-point is used instead.
  const n=this.count,m=[0,0,0],c=[0,0,0,0,0,0];
  for(let i=0;i<n;i++)for(let k=0;k<3;k++)m[k]+=points[i*3+k]/Math.max(1,n);
  for(let i=0;i<n;i++){const x=points[i*3]-m[0],y=points[i*3+1]-m[1],z=points[i*3+2]-m[2];c[0]+=x*x;c[1]+=x*y;c[2]+=x*z;c[3]+=y*y;c[4]+=y*z;c[5]+=z*z}
  for(let k=0;k<6;k++)c[k]/=Math.max(1,n);
  const {values,vectors}=n>=4?eigenSym3(c):{values:[1,1,1],vectors:[new THREE.Vector3(1,0,0),new THREE.Vector3(0,1,0),new THREE.Vector3(0,0,1)]};
  this.thinAxis=vectors[2].clone().normalize();
  this.thinRatio=values[2]/Math.max(values[1],1e-12);
  this.thinReliable=this.thinRatio<=FILTER.illDefinedRatio;
 }
 point(i,out=new THREE.Vector3()){return out.set(this.points[i*3],this.points[i*3+1],this.points[i*3+2])}
 // Nearest centre to p within `radius`: {index, distance} (a reused object) or null.
 nearest(p,radius){
  let best=-1,bd=radius*radius;
  this.grid.near(p.x,p.y,p.z,radius,(i,d2)=>{if(d2<=bd){bd=d2;best=i}});
  if(best<0)return null;
  _hit.index=best;_hit.distance=Math.sqrt(bd);return _hit;
 }
 distance(p,maxD){const h=this.nearest(p,maxD);return h?h.distance:Infinity}
 // Nearest hit of a ray (origin, unit dir) against the centres as spheres of `radius`: {index, t, point} or null.
 // Brute force over the part's points (at most FILTER.maxPoints).
 rayHit(origin,dir,radius,maxDist=Infinity,out={index:-1,t:0,point:new THREE.Vector3()}){
  const P=this.points,r2=radius*radius;let bt=maxDist,bi=-1;
  for(let i=0;i<this.count;i++){
   const dx=P[i*3]-origin.x,dy=P[i*3+1]-origin.y,dz=P[i*3+2]-origin.z,t=dx*dir.x+dy*dir.y+dz*dir.z;
   if(t<-radius)continue;
   const d2=dx*dx+dy*dy+dz*dz-t*t;
   if(d2>r2)continue;
   const th=Math.max(0,t-Math.sqrt(r2-d2));
   if(th<bt){bt=th;bi=i}
  }
  if(bi<0)return null;
  out.index=bi;out.t=bt;out.point.copy(origin).addScaledVector(dir,bt);return out;
 }
 // The part's box from the surface splats only, trimmed 5..95 %: the hover outline and the joint's mass.
 fitBox(){return fitObb(this.points,{trim:.05})}
}
const _hit={index:-1,distance:0};

// Build a part's contact set from its parsed PLY ({centres, opacity, maxLogScale}).
export function buildContacts(parsed,opts){
 const {points,stats}=filterContactCentres(parsed.centres,{opacity:parsed.opacity,maxLogScale:parsed.maxLogScale,...opts});
 const set=new ContactSet(points);set.stats=stats;return set;
}

const _r=new THREE.Vector3();
// How close the world-space (joint-frame) point is to the part's surface, or Infinity beyond maxD.
export function contactDistance(part,local,maxD=GRAB_RADIUS){
 const c=part.contacts,j=part.joint;if(!c||!j)return Infinity;
 return c.distance(j.toRest(local,j.q,_r),maxD);
}

// Nearest part (and its distance) to a point in the joint's frame, over every part with contacts.
export function nearestPart(parts,local,maxD=GRAB_RADIUS){
 let best=null,bd=maxD;
 for(const p of parts){if(!p.interactive||!p.contacts)continue;const d=contactDistance(p,local,bd);if(d<=bd){best=p;bd=d}}
 return best?{part:best,distance:bd}:null;
}

const _o=new THREE.Vector3(),_d=new THREE.Vector3(),_rq=new THREE.Quaternion(),_ray={index:-1,t:0,point:new THREE.Vector3()};
// A ray in the joint's frame against every part's centres: {part, point (joint frame), distance} for the nearest.
export function pickParts(parts,origin,dir,radius=PICK_RADIUS,maxDist=Infinity){
 let best=null,bt=maxDist;
 for(const p of parts){
  const c=p.contacts,j=p.joint;if(!p.interactive||!c||!j)continue;
  // The ray into the part's rest space: undo the joint's pose (hinge turn about its axis, or slide).
  _o.copy(origin);_d.copy(dir);
  if(j.kind==='hinge'){j.toRest(_o,j.q,_o);_d.applyQuaternion(_rq.setFromAxisAngle(j.axis,-j.q))}
  else j.toRest(_o,j.q,_o);
  const h=c.rayHit(_o,_d,radius,bt,_ray);
  if(h){bt=h.t;best={part:p,point:j.pointAt(h.point,j.q,new THREE.Vector3()),distance:h.t}}
 }
 return best;
}
