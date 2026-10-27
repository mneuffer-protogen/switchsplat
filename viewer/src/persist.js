// Remembers where each part was left (its joint coordinate q) and whether it is locked, in localStorage, keyed by
// scene URL and part id, so a door left half-open is still half-open after a reload. Every access is wrapped:
// storage can be blocked (private windows, sandboxed iframes) and the viewer must work without it.
const PREFIX='switchsplat:v1:';

export function createPersistence(sceneKey,{delay=400,storage=()=>globalThis.localStorage}={}){
 const keyFor=id=>`${PREFIX}${sceneKey}:${id}`;
 const timers=new Map(),pending=new Map();
 const write=(id,state)=>{try{storage().setItem(keyFor(id),JSON.stringify(state))}catch{}};
 return {
  // {q, locked}: either may be null when it was never saved.
  restoreState(id){
   try{
    const v=JSON.parse(storage().getItem(keyFor(id))??'null');
    return {q:typeof v?.q==='number'&&Number.isFinite(v.q)?v.q:null,locked:typeof v?.locked==='boolean'?v.locked:null};
   }catch{return {q:null,locked:null}}
  },
  restore(id){return this.restoreState(id).q},
  // Save on every change, at most once per `delay` ms per part (the last value always lands).
  save(id,q,locked){
   pending.set(id,locked===undefined?{q}:{q,locked});
   if(timers.has(id))return;
   timers.set(id,setTimeout(()=>{timers.delete(id);write(id,pending.get(id))},delay));
  },
  flush(){for(const [id,t] of timers){clearTimeout(t);timers.delete(id);write(id,pending.get(id))}},
  // Bind a joint: restores its saved pose and lock now and saves them whenever they change.
  watch(id,joint){
   // The lock is only written when it differs from what scene.json says, so editing the scene's default still works.
   const initial=joint.locked,{q,locked}=this.restoreState(id);
   if(q!==null)joint.setPosition(q);
   if(locked!==null)joint.setLocked(locked);
   const save=()=>this.save(id,joint.q,joint.locked===initial?undefined:joint.locked);
   joint.on('value',save);
   // 'value' fires per fixed step while moving, but the last step of a settle can land between events.
   joint.on('settled',save);
   joint.on('latched',save);
   joint.on('locked',save);
   return q;
  },
  clear(){this.flush();try{const s=storage();for(let i=s.length-1;i>=0;i--){const k=s.key(i);if(k?.startsWith(`${PREFIX}${sceneKey}:`))s.removeItem(k)}}catch{}},
 };
}
