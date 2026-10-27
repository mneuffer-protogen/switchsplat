// ?perf: counts what Spark does while parts animate and exposes window.switchsplatPerf.
//   start()/stop() -> {fps, frameMs, generates/s, sorts/s, sortMs, updateMs, staleFrames%, ...}
// Hooks are plain wrappers around Spark's per-frame entry points; nothing runs without ?perf.
export function createPerf({spark,getParts}){
 const s={frames:0,t0:0,dts:[],gen:0,genMs:0,sorts:0,sortMs:0,upd:0,updMs:0,stale:0,staleMax:0,lagFrames:[],last:0,running:false};
 const wrap=(obj,name,fn)=>{const orig=obj[name].bind(obj);obj[name]=fn(orig)};
 wrap(spark,'updateInternal',orig=>function(a){
  const before=spark.current,t=performance.now();const r=orig(a);const dt=performance.now()-t;
  if(s.running){s.upd++;s.updMs+=dt;if(spark.current!==before){s.gen++;s.genMs+=dt}}
  return r;
 });
 wrap(spark,'readbackDepth',orig=>async function(a){
  const t=performance.now();const r=await orig(a);
  if(s.running){s.sorts++;s.sortMs+=performance.now()-t}
  return r;
 });
 // Per frame: how many versions of a moving part's pose the drawn accumulator is behind.
 const lag=()=>{
  let worst=0;
  for(const p of getParts()){
   if(!p.mesh||!p.joint?.anim&&!p.joint?.active)continue;
   const m=spark.display.mapping.find(e=>e.node===p.mesh);
   worst=Math.max(worst,m?p.mesh.version-m.version:0);
  }
  return worst;
 };
 const tick=now=>{
  if(s.running){
   if(s.last)s.dts.push(now-s.last);
   s.last=now;s.frames++;
   const l=lag();if(l>0)s.stale++;s.staleMax=Math.max(s.staleMax,l);
  }
  requestAnimationFrame(tick);
 };
 requestAnimationFrame(tick);
 return {
  start(){Object.assign(s,{frames:0,dts:[],gen:0,genMs:0,sorts:0,sortMs:0,upd:0,updMs:0,stale:0,staleMax:0,last:0,t0:performance.now(),running:true})},
  stop(){
   s.running=false;const sec=(performance.now()-s.t0)/1000,d=[...s.dts].sort((a,b)=>a-b);
   const r=x=>Math.round(x*10)/10;
   return {seconds:r(sec),fps:r(s.frames/sec),frameMsMedian:r(d[d.length>>1]??0),frameMsP95:r(d[Math.floor(d.length*.95)]??0),frameMsMax:r(d.at(-1)??0),
    generatesPerSec:r(s.gen/sec),syncMsPerGenerate:r(s.gen?s.genMs/s.gen:0),sortsPerSec:r(s.sorts/sec),msPerSort:r(s.sorts?s.sortMs/s.sorts:0),
    framesWithStalePose:s.stale,staleMaxVersions:s.staleMax,frames:s.frames,splats:spark.display.numSplats};
  },
 };
}
