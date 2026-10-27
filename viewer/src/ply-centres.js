// Read only the splat centres out of a binary little-endian 3DGS PLY (see articulation-spec.md): enough
// to fit the grab proxy of a part and to frame the camera, without waiting for Spark to decode.
// Anything else (ASCII, big-endian, compressed PLY with no float x/y/z) throws, and the caller falls
// back to Spark's own SplatMesh.forEachSplat.
const SIZES={char:1,int8:1,uchar:1,uint8:1,short:2,int16:2,ushort:2,uint16:2,int:4,int32:4,uint:4,uint32:4,float:4,float32:4,double:8,float64:8};

export function readPlyHeader(buffer){
 const bytes=new Uint8Array(buffer,0,Math.min(buffer.byteLength,1<<16));
 let text='';for(let i=0;i<bytes.length;i++)text+=String.fromCharCode(bytes[i]);
 const end=text.indexOf('end_header');
 if(!text.startsWith('ply')||end<0)throw new Error('not a PLY file');
 let dataStart=end+'end_header'.length;
 if(text[dataStart]==='\r')dataStart++;
 if(text[dataStart]==='\n')dataStart++;
 let format=null,count=0,stride=0,inVertex=false,seenVertex=false;const props={};
 for(const raw of text.slice(0,end).split(/\r?\n/)){
  const line=raw.trim().split(/\s+/);
  if(line[0]==='format')format=line[1];
  else if(line[0]==='element'){inVertex=line[1]==='vertex'&&!seenVertex;if(inVertex){seenVertex=true;count=parseInt(line[2],10)}}
  else if(line[0]==='property'&&inVertex){
   if(line[1]==='list')throw new Error('PLY list property in vertex element');
   const size=SIZES[line[1]];if(!size)throw new Error(`PLY property type ${line[1]}`);
   props[line[2]]={type:line[1],offset:stride};stride+=size;
  }
 }
 if(format!=='binary_little_endian')throw new Error(`PLY format ${format} (need binary_little_endian)`);
 for(const k of ['x','y','z'])if(props[k]?.type!=='float'&&props[k]?.type!=='float32')throw new Error('PLY has no float x/y/z');
 if(dataStart+count*stride>buffer.byteLength)throw new Error('PLY is truncated');
 return {count,stride,dataStart,offsets:{x:props.x.offset,y:props.y.offset,z:props.z.offset},props};
}

// Centres as a Float32Array [x,y,z,...]; at most `max` of them, evenly strided, so a 5M-splat file costs a few ms.
export function plyCentres(buffer,{max=Infinity}={}){
 const h=readPlyHeader(buffer),dv=new DataView(buffer);
 const step=Math.max(1,Math.ceil(h.count/max)),n=Math.ceil(h.count/step),out=new Float32Array(n*3);
 let j=0;
 for(let i=0;i<h.count;i+=step){
  const b=h.dataStart+i*h.stride;
  out[j++]=dv.getFloat32(b+h.offsets.x,true);out[j++]=dv.getFloat32(b+h.offsets.y,true);out[j++]=dv.getFloat32(b+h.offsets.z,true);
 }
 return out.subarray(0,j);
}
