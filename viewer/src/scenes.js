// The scene list (public/scenes/index.json) and the ?scene= URL: parsing, which scene is the default, how an
// arbitrary ?scene= maps onto a list entry, and the URL a pick turns into. Pure functions, no DOM.
//   { version, default: <id>, scenes: [ {id, name, path, description?, metresPerUnit?, heavy?} ] }

// A scene's identity is its export directory: "scenes/car", "scenes/car/", "./scenes/car/scene.json" and the
// absolute URL of any of those are the same scene.
export function sceneDir(param,base=globalThis.location?.href??'http://localhost/'){
 const abs=new URL(param,base);
 if(abs.pathname.endsWith('.json'))return new URL('./',abs).href;
 if(!abs.pathname.endsWith('/'))abs.pathname+='/';
 return abs.href;
}

// Validated index, or null when there is nothing usable (missing/garbled file: the viewer then works as before).
export function parseSceneIndex(data){
 if(!data||typeof data!=='object'||!Array.isArray(data.scenes))return null;
 const seen=new Set(),scenes=[];
 for(const s of data.scenes){
  if(!s||typeof s.path!=='string'||!s.path.trim())continue;
  const id=typeof s.id==='string'&&s.id?s.id:s.path;
  if(seen.has(id))continue;seen.add(id);
  const mpu=Number(s.metresPerUnit);
  scenes.push({id,name:typeof s.name==='string'&&s.name?s.name:id,path:s.path.trim(),description:typeof s.description==='string'?s.description:'',
   metresPerUnit:Number.isFinite(mpu)&&mpu>0?mpu:null,heavy:s.heavy===true||(typeof s.heavy==='string'&&s.heavy!=='')?s.heavy:false});
 }
 if(!scenes.length)return null;
 const def=scenes.find(s=>s.id===data.default)??scenes[0];
 return {version:data.version??1,default:def.id,scenes};
}

// The list entry for a ?scene= value (or path), or null for a custom one.
export function matchEntry(index,param,base){
 if(!index||!param)return null;
 const dir=sceneDir(param,base);
 return index.scenes.find(s=>sceneDir(s.path,base)===dir)??null;
}

// What to load at startup: an explicit ?scene= always wins (listed or not); else the index default; else null
// (the caller falls back to scenes/synthetic/).
export function initialScene(index,explicit,base){
 if(explicit)return {param:explicit,entry:matchEntry(index,explicit,base),custom:!matchEntry(index,explicit,base)&&!!index};
 const e=index?.scenes.find(s=>s.id===index.default)??null;
 return e?{param:e.path,entry:e,custom:false}:{param:null,entry:null,custom:false};
}

// Dropdown options: the list, plus a "Custom: <url>" entry first when the current scene is not in it.
export function pickerOptions(index,currentParam,base){
 const out=(index?.scenes??[]).map(s=>({value:s.id,label:s.name,entry:s,param:s.path,heavy:s.heavy}));
 if(currentParam&&!matchEntry(index,currentParam,base))out.unshift({value:'custom',label:`Custom: ${currentParam}`,entry:null,param:currentParam,heavy:false});
 return out;
}

// "~110 MB download" style hint for a heavy entry (`heavy` may be true or a string like "~110 MB").
export function heavyHint(entry){
 if(!entry?.heavy)return '';
 return typeof entry.heavy==='string'?entry.heavy:'large download';
}

// The page URL for a pick: ?scene= set to the scene's path, `sh`, `perf`, `edit`, `emulate`... kept, and the
// per-scene overrides (`scale`, `cam`, `xr`) dropped because they describe the previous scene.
export function sceneUrl(href,param,{keep=true}={}){
 const u=new URL(href);
 u.searchParams.set('scene',param);
 if(!keep)for(const k of ['scale','cam','xr'])u.searchParams.delete(k);
 return u.toString();
}
