import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {fitObb,eigenSym3,distanceToProxy} from '../src/obb.js';
import {readPlyHeader,plyCentres} from '../src/ply-centres.js';

function lcg(seed){let s=seed;return ()=>((s=(s*1664525+1013904223)>>>0)/4294967296)}

test('eigenSym3 recovers the axes of a diagonal covariance',()=>{
 const {values,vectors}=eigenSym3([4,0,0,1,0,.25]);
 assert.deepEqual(values.map(v=>+v.toFixed(6)),[4,1,.25]);
 assert.ok(Math.abs(Math.abs(vectors[0].x)-1)<1e-9);
});

test('fitObb finds a rotated slab and ignores floaters',()=>{
 const rnd=lcg(7),q=new THREE.Quaternion().setFromEuler(new THREE.Euler(.3,.8,-.2)),pts=[];
 const centre=new THREE.Vector3(1,2,3),v=new THREE.Vector3();
 for(let i=0;i<20000;i++){v.set((rnd()-.5)*.9,(rnd()-.5)*2,(rnd()-.5)*.04).applyQuaternion(q).add(centre);pts.push(v.x,v.y,v.z)}
 for(let i=0;i<150;i++)pts.push(centre.x+(rnd()-.5)*20,centre.y+(rnd()-.5)*20,centre.z+(rnd()-.5)*20);
 const box=fitObb(new Float32Array(pts));
 const sizes=[box.size.x,box.size.y,box.size.z].sort((a,b)=>a-b);
 assert.ok(Math.abs(sizes[2]-2)<.1&&Math.abs(sizes[1]-.9)<.1,`sizes ${sizes}`);
 assert.ok(sizes[0]>=.03&&sizes[0]<.12,`thin axis ${sizes[0]}`);
 assert.ok(box.center.distanceTo(centre)<.1,`centre off by ${box.center.distanceTo(centre)}`);
 // The long axis of the box follows the slab's long axis.
 const long=[box.size.x,box.size.y,box.size.z].indexOf(sizes[2]);
 const axis=[new THREE.Vector3(1,0,0),new THREE.Vector3(0,1,0),new THREE.Vector3(0,0,1)][long].applyQuaternion(box.quaternion);
 assert.ok(Math.abs(axis.dot(new THREE.Vector3(0,1,0).applyQuaternion(q)))>.98);
});

test('distanceToProxy is zero inside and metric outside',()=>{
 const mesh=new THREE.Mesh(new THREE.BoxGeometry(1,1,1));mesh.scale.set(2,1,.5);mesh.position.set(1,0,0);
 const parent=new THREE.Group();parent.rotation.y=Math.PI/2;parent.add(mesh);
 const p=new THREE.Vector3();
 mesh.updateWorldMatrix(true,false);mesh.getWorldPosition(p);assert.equal(distanceToProxy(mesh,p),0);
 // Parent turned 90 deg about Y: the mesh centre is at (0,0,-1); its +X (length 2) lies along world -Z.
 assert.ok(Math.abs(distanceToProxy(mesh,new THREE.Vector3(0,0,-3))-1)<1e-9);
 assert.ok(Math.abs(distanceToProxy(mesh,new THREE.Vector3(.5,0,-1))-.25)<1e-9);
});

function makePly(n,{extra=true,format='binary_little_endian'}={}){
 const header=`ply\nformat ${format} 1.0\nelement vertex ${n}\nproperty float x\nproperty float y\nproperty float z\n${extra?'property float nx\nproperty float ny\nproperty float nz\nproperty float opacity\n':''}end_header\n`;
 const stride=extra?28:12,buf=new ArrayBuffer(header.length+n*stride),u8=new Uint8Array(buf),dv=new DataView(buf);
 for(let i=0;i<header.length;i++)u8[i]=header.charCodeAt(i);
 for(let i=0;i<n;i++){const b=header.length+i*stride;dv.setFloat32(b,i,true);dv.setFloat32(b+4,i*2,true);dv.setFloat32(b+8,i*3,true)}
 return buf;
}
test('plyCentres reads positions from a binary PLY, strided',()=>{
 const buf=makePly(100);
 const h=readPlyHeader(buf);assert.equal(h.count,100);assert.equal(h.stride,28);
 const all=plyCentres(buf);assert.equal(all.length,300);assert.deepEqual([...all.slice(30,33)],[10,20,30]);
 const some=plyCentres(buf,{max:10});assert.equal(some.length,30);assert.deepEqual([...some.slice(3,6)],[10,20,30]);
 assert.throws(()=>plyCentres(makePly(3,{format:'ascii'})),/binary_little_endian/);
 assert.throws(()=>readPlyHeader(new Uint8Array([1,2,3]).buffer),/not a PLY/);
});
