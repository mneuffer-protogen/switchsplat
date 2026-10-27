// Remembers where each part was left (its joint coordinate q) in localStorage, keyed by scene URL and part
// id, so a door left half-open is still half-open after a reload. Every access is wrapped: storage can
// be blocked (private windows, sandboxed iframes) and the viewer must work without it.
const PREFIX='switchsplat:v1:';

export function createPersistence(sceneKey,{delay=400,storage=()=>globalThis.localStorage}={}){
 const keyFor=id=>`${PREFIX}${sceneKey}:${id}`;
 const timers=new Map(),pending=new Map();
 const write=(id,q)=>{try{storage().setItem(keyFor(id),JSON.stringify({q}))}catch{}};
 return {
  restore(id){
   try{const v=JSON.parse(storage().getItem(keyFor(id))??'null');return typeof v?.q==='number'&&Number.isFinite(v.q)?v.q:null}catch{return null}
  },
  // Save on every change, at most once per `delay` ms per part (the last value always lands).
  save(id,q){
   pending.set(id,q);
   if(timers.has(id))return;
   timers.set(id,setTimeout(()=>{timers.delete(id);write(id,pending.get(id))},delay));
  },
  flush(){for(const [id,t] of timers){clearTimeout(t);timers.delete(id);write(id,pending.get(id))}},
  // Bind a joint: restores its saved pose now and saves it whenever it moves.
  watch(id,joint){
   const q=this.restore(id);
   if(q!==null)joint.setPosition(q);
   joint.on('value',()=>this.save(id,joint.q));
   // 'value' fires per fixed step while moving, but the last step of a settle can land between events.
   joint.on('settled',()=>this.save(id,joint.q));
   joint.on('latched',()=>this.save(id,joint.q));
   return q;
  },
  clear(){this.flush();try{const s=storage();for(let i=s.length-1;i>=0;i--){const k=s.key(i);if(k?.startsWith(`${PREFIX}${sceneKey}:`))s.removeItem(k)}}catch{}},
 };
}
