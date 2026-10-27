// An oriented box around a part's splat centres: PCA for the axes, then the 2nd..98th percentile along
// each so a few floaters do not blow the box up. The box is the grab/hover proxy (raycasting and "is the
// controller within 6 cm"), and it sizes the joint's mass.
import * as THREE from 'three';

// Eigen-decomposition of a symmetric 3x3 by cyclic Jacobi rotations. Returns {values, vectors} with
// vectors[i] the unit eigenvector of values[i], sorted by decreasing eigenvalue.
export function eigenSym3(m){
 const a=[[m[0],m[1],m[2]],[m[1],m[3],m[4]],[m[2],m[4],m[5]]];
 const v=[[1,0,0],[0,1,0],[0,0,1]];
 for(let sweep=0;sweep<24;sweep++){
  const off=Math.abs(a[0][1])+Math.abs(a[0][2])+Math.abs(a[1][2]);
  if(off<1e-14)break;
  for(const [p,q] of [[0,1],[0,2],[1,2]]){
   if(Math.abs(a[p][q])<1e-18)continue;
   const theta=(a[q][q]-a[p][p])/(2*a[p][q]),t=Math.sign(theta||1)/(Math.abs(theta)+Math.sqrt(theta*theta+1)),c=1/Math.sqrt(t*t+1),s=t*c;
   for(let k=0;k<3;k++){const akp=a[k][p],akq=a[k][q];a[k][p]=c*akp-s*akq;a[k][q]=s*akp+c*akq}
   for(let k=0;k<3;k++){const apk=a[p][k],aqk=a[q][k];a[p][k]=c*apk-s*aqk;a[q][k]=s*apk+c*aqk}
   for(let k=0;k<3;k++){const vkp=v[k][p],vkq=v[k][q];v[k][p]=c*vkp-s*vkq;v[k][q]=s*vkp+c*vkq}
  }
 }
 const order=[0,1,2].sort((i,j)=>a[j][j]-a[i][i]);
 return {values:order.map(i=>a[i][i]),vectors:order.map(i=>new THREE.Vector3(v[0][i],v[1][i],v[2][i]))};
}

const percentile=(sorted,p)=>sorted[Math.min(sorted.length-1,Math.max(0,Math.round(p*(sorted.length-1))))];

// centres: Float32Array [x,y,z,...]. Returns {center, size, quaternion} (all in the centres' frame), or
// null for fewer than 4 points. `minSize` keeps a paper-thin door grabbable.
export function fitObb(centres,{trim=.02,minSize=.03,maxPoints=60000}={}){
 const total=Math.floor(centres.length/3);if(total<4)return null;
 const stride=Math.max(1,Math.ceil(total/maxPoints)),pts=[];
 for(let i=0;i<total;i+=stride)pts.push(centres[i*3],centres[i*3+1],centres[i*3+2]);
 const n=pts.length/3;
 // Robust PCA: drop what lies outside the trimmed range on any world axis before taking the covariance.
 const lo=[0,0,0],hi=[0,0,0];
 for(let k=0;k<3;k++){const col=new Float64Array(n);for(let i=0;i<n;i++)col[i]=pts[i*3+k];col.sort();lo[k]=percentile(col,trim);hi[k]=percentile(col,1-trim)}
 const keep=[];
 for(let i=0;i<n;i++)if(pts[i*3]>=lo[0]&&pts[i*3]<=hi[0]&&pts[i*3+1]>=lo[1]&&pts[i*3+1]<=hi[1]&&pts[i*3+2]>=lo[2]&&pts[i*3+2]<=hi[2])keep.push(i);
 const src=keep.length>=4?keep:[...Array(n).keys()];
 const mean=[0,0,0];for(const i of src)for(let k=0;k<3;k++)mean[k]+=pts[i*3+k]/src.length;
 const c=[0,0,0,0,0,0];
 for(const i of src){const x=pts[i*3]-mean[0],y=pts[i*3+1]-mean[1],z=pts[i*3+2]-mean[2];c[0]+=x*x;c[1]+=x*y;c[2]+=x*z;c[3]+=y*y;c[4]+=y*z;c[5]+=z*z}
 for(let k=0;k<6;k++)c[k]/=src.length;
 const {vectors}=eigenSym3(c);
 // Right-handed basis: axes[2] = axes[0] x axes[1].
 const ax=vectors[0].clone().normalize(),ay=vectors[1].clone().normalize(),az=new THREE.Vector3().crossVectors(ax,ay).normalize();
 const q=new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(ax,ay,az));
 // Extent along each axis over every point, trimmed.
 const axes=[ax,ay,az],mn=[0,0,0],mx=[0,0,0];
 for(let k=0;k<3;k++){
  const proj=new Float64Array(n);
  for(let i=0;i<n;i++)proj[i]=pts[i*3]*axes[k].x+pts[i*3+1]*axes[k].y+pts[i*3+2]*axes[k].z;
  proj.sort();mn[k]=percentile(proj,trim);mx[k]=percentile(proj,1-trim);
 }
 const size=new THREE.Vector3(Math.max(minSize,mx[0]-mn[0]),Math.max(minSize,mx[1]-mn[1]),Math.max(minSize,mx[2]-mn[2]));
 const mid=[(mn[0]+mx[0])/2,(mn[1]+mx[1])/2,(mn[2]+mx[2])/2];
 const center=new THREE.Vector3().addScaledVector(ax,mid[0]).addScaledVector(ay,mid[1]).addScaledVector(az,mid[2]);
 return {center,size,quaternion:q};
}

// Distance in world units from a world point to a proxy box mesh (a unit cube scaled to the box size and posed
// by its ancestors); 0 inside. Ancestors may carry a uniform scale (the scene root's metres per unit): the box's
// world size is read from the proxy's world scale.
export function distanceToProxy(proxy,point,out=new THREE.Vector3()){
 proxy.updateWorldMatrix(true,false);
 out.copy(point);proxy.worldToLocal(out);
 proxy.getWorldScale(_ws);
 const sx=_ws.x,sy=_ws.y,sz=_ws.z;
 const dx=Math.max(Math.abs(out.x)*sx-sx/2,0),dy=Math.max(Math.abs(out.y)*sy-sy/2,0),dz=Math.max(Math.abs(out.z)*sz-sz/2,0);
 return Math.hypot(dx,dy,dz);
}

const _ws=new THREE.Vector3();

// Axis-aligned bounds of the centres without the outer `trim` fraction on each side (a floater far from
// the scene would otherwise send the camera to the horizon).
export function trimmedBounds(centres,trim=.01){
 const n=Math.floor(centres.length/3),box=new THREE.Box3();if(!n)return box;
 const lo=new THREE.Vector3(),hi=new THREE.Vector3();
 for(const [k,axis] of [[0,'x'],[1,'y'],[2,'z']]){
  const col=new Float64Array(n);for(let i=0;i<n;i++)col[i]=centres[i*3+k];col.sort();
  lo[axis]=percentile(col,trim);hi[axis]=percentile(col,1-trim);
 }
 return box.set(lo,hi);
}
