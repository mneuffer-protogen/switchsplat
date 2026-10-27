// Reveal/vanish fades must never change a SplatMesh's visibility: Spark freezes a moving part's redraw for a whole
// sort whenever the set of visible splat meshes changes (a 'mapping' change).
import test from 'node:test';
import assert from 'node:assert/strict';
import {setFade} from '../src/fade.js';

test('setFade clamps opacity and leaves visibility alone',()=>{
 const m={opacity:1,visible:true};
 for(const [v,o] of [[0,0],[.4,.4],[-1,0],[7,1],[0.001,0.001]]){setFade(m,v);assert.equal(m.opacity,o);assert.equal(m.visible,true)}
});
