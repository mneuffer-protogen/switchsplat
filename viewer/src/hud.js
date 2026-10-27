// The overlay: one row per part (name, state %, open/close button), plus the scene name and status line.
import {toggle} from './desktop-input.js';

export function createHud({parts,onToggle=()=>{}}){
 const list=document.getElementById('parts'),rows=[];
 list.textContent='';
 for(const part of parts){
  const li=document.createElement('li');
  const name=document.createElement('span');name.className='name';name.textContent=part.name;
  li.append(name);
  if(part.lowFit){
   const b=document.createElement('span');b.className='badge';b.textContent='low fit';
   b.title=`Joint fit inlier ratio ${part.fit.inlierRatio.toFixed(2)}: check this joint in edit mode`;li.append(b);
  }
  const pct=document.createElement('span'),btn=document.createElement('button'),bar=document.createElement('div'),fill=document.createElement('div');
  pct.className='pct';bar.className='bar';bar.append(fill);
  if(part.joint&&part.interactive){
   btn.type='button';btn.className='toggle';btn.addEventListener('click',()=>{toggle(part);onToggle(part)});
   li.append(pct,btn);
  }else{pct.textContent='static';pct.title='No joint for this part';li.append(pct)}
  li.append(bar);
  list.append(li);rows.push({part,li,pct,btn,fill,last:''});
  li.addEventListener('pointerenter',()=>part.hudHover=true);li.addEventListener('pointerleave',()=>part.hudHover=false);
 }
 return {
  update(){
   for(const r of rows){
    const j=r.part.joint;if(!j||!r.part.interactive)continue;
    const f=Math.round(j.fraction*100);
    const heading=j.anim?j.anim.target>(j.closedQ+j.openQ)/2:j.fraction>.5,label=heading?'Close':'Open',key=`${f}|${label}`;
    if(key===r.last)continue;
    r.last=key;r.pct.textContent=`${f}%`;r.fill.style.width=`${f}%`;r.btn.textContent=label;
    r.li.classList.toggle('open',j.fraction>.5);
   }
  },
 };
}
