// Reveal/vanish fades must never change a SplatMesh's visibility: Spark freezes a moving part's redraw for a whole
// sort whenever the set of visible splat meshes changes (a 'mapping' change).
import test from 'node:test';
import assert from 'node:assert/strict';
import {setFade,smoothstep,fadeLevels,FADE_DEFAULTS} from '../src/fade.js';

test('setFade clamps opacity and leaves visibility alone',()=>{
 const m={opacity:1,visible:true};
 for(const [v,o] of [[0,0],[.4,.4],[-1,0],[7,1],[0.001,0.001]]){setFade(m,v);assert.equal(m.opacity,o);assert.equal(m.visible,true)}
});

test('reveal and vanish switch near the ends of the travel, not linearly',()=>{
 const at=f=>fadeLevels(f);
 assert.deepEqual(at(0),{reveal:0,vanish:1});
 assert.deepEqual(at(1),{reveal:1,vanish:0});
 // halfway through the swing nothing is ghosted: the uncovered content is fully in, the closed-only content fully out
 assert.equal(at(.5).reveal,1);assert.equal(at(.5).vanish,0);
 // and they are monotonic
 let r=-1,v=2;for(let i=0;i<=100;i++){const l=at(i/100);assert.ok(l.reveal>=r);assert.ok(l.vanish<=v);r=l.reveal;v=l.vanish}
});

test('a part with no fade setting (null) uses the defaults',()=>{
 assert.deepEqual(fadeLevels(.5,null),fadeLevels(.5));
 assert.deepEqual(fadeLevels(.5,undefined),fadeLevels(.5));
});

test('scene.json fade overrides the stretches, smoothstep is safe for empty ranges',()=>{
 const l=fadeLevels(.5,{reveal:[.4,.9],vanish:[.5,1]});
 assert.ok(l.reveal>0&&l.reveal<1);assert.equal(l.vanish,1);
 assert.equal(smoothstep(.3,.3,.2),0);assert.equal(smoothstep(.3,.3,.4),1);
 assert.ok(FADE_DEFAULTS.reveal[1]<=.5);
});
