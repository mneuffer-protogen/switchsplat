// The overlay: one row per part (name, state %, open/close and lock buttons), plus the scene name and status line.
import {toggle} from './desktop-input.js';

// "merged from slot02, slot03" for a part the importer built out of several slots, else ''.
export const mergedLabel=part=>part.merged?.length?`merged from ${part.merged.join(', ')}`:'';

export function createHud({parts,onToggle=()=>{}}){
 const list=document.getElementById('parts'),rows=[];
 list.textContent='';
 for(const part of parts){
  const li=document.createElement('li');
  const name=document.createElement('span');name.className='name';name.textContent=part.name;
  li.append(name);
  const merged=mergedLabel(part);
  if(merged){li.title=`${part.name} (${merged})`;name.title=li.title}
  const pct=document.createElement('span'),btn=document.createElement('button'),lock=document.createElement('button'),bar=document.createElement('div'),fill=document.createElement('div');
  pct.className='pct';bar.className='bar';bar.append(fill);
  if(part.joint&&part.interactive){
   btn.type='button';btn.className='toggle';btn.addEventListener('click',()=>{toggle(part);onToggle(part)});
   lock.type='button';lock.className='lock';lock.addEventListener('click',()=>part.joint.setLocked(!part.joint.locked));
   li.append(pct,btn,lock);
  }else{pct.textContent='static';pct.title='No joint for this part';li.append(pct)}
  // Small print under the row: how the part was built and whether its joint fit is doubtful.
  const meta=document.createElement('div');meta.className='meta';
  if(merged){const m=document.createElement('span');m.className='merged';m.textContent=`(${merged})`;meta.append(m)}
  if(part.lowFit){
   const b=document.createElement('span');b.className='badge';b.textContent='low fit';
   b.title=`Joint fit inlier ratio ${part.fit.inlierRatio.toFixed(2)}: check this joint in edit mode`;meta.append(b);
  }
  if(meta.childNodes.length)li.append(meta);
  li.append(bar);
  list.append(li);rows.push({part,li,pct,btn,lock,fill,last:''});
  li.addEventListener('pointerenter',()=>part.hudHover=true);li.addEventListener('pointerleave',()=>part.hudHover=false);
 }
 return {
  update(){
   for(const r of rows){
    const j=r.part.joint;if(!j||!r.part.interactive)continue;
    const f=Math.round(j.fraction*100);
    const heading=j.anim?j.anim.target>(j.closedQ+j.openQ)/2:j.fraction>.5,label=heading?'Close':'Open',key=`${f}|${label}|${j.locked}`;
    if(key===r.last)continue;
    r.last=key;r.pct.textContent=j.locked?`${f}% locked`:`${f}%`;r.fill.style.width=`${f}%`;r.btn.textContent=label;
    r.lock.textContent=j.locked?'Unlock':'Lock';r.lock.title=j.locked?'Unlock this part':'Lock this part where it is: it only rattles when pulled or pushed';
    r.li.classList.toggle('open',j.fraction>.5);r.li.classList.toggle('locked',j.locked);
   }
  },
 };
}
