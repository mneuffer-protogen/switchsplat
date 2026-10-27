import test from 'node:test';
import assert from 'node:assert/strict';
import {sceneDir,parseSceneIndex,matchEntry,initialScene,pickerOptions,heavyHint,sceneUrl} from '../src/scenes.js';

const BASE='https://example.com/viewer/';
const raw={version:1,default:'car',scenes:[
 {id:'car',name:'Car',path:'scenes/car/',metresPerUnit:.586,heavy:true,description:'x'},
 {id:'synthetic',name:'Cabinet',path:'scenes/synthetic/'}]};

test('sceneDir treats the directory, the scene.json and absolute forms alike',()=>{
 const d='https://example.com/viewer/scenes/car/';
 for(const p of ['scenes/car','scenes/car/','./scenes/car/scene.json',d,`${d}scene.json`])assert.equal(sceneDir(p,BASE),d);
});

test('parseSceneIndex keeps valid entries and defaults',()=>{
 const i=parseSceneIndex(raw);
 assert.equal(i.scenes.length,2);assert.equal(i.default,'car');assert.equal(i.scenes[0].metresPerUnit,.586);assert.equal(i.scenes[1].heavy,false);
 assert.equal(parseSceneIndex({...raw,default:'nope'}).default,'car');// unknown default -> first
 assert.equal(parseSceneIndex({scenes:[{path:'a/'},{path:'b/',id:'a/'},null,{name:'no path'},{path:'  '}]}).scenes.length,1);
});

test('parseSceneIndex rejects garbage',()=>{
 for(const v of [null,undefined,'x',42,{},{scenes:'no'},{scenes:[]},{scenes:[{name:'x'}]}])assert.equal(parseSceneIndex(v),null);
});

test('initialScene: explicit ?scene= wins, else the index default, else null',()=>{
 const i=parseSceneIndex(raw);
 let s=initialScene(i,'scenes/synthetic',BASE);assert.equal(s.entry.id,'synthetic');assert.equal(s.custom,false);
 s=initialScene(i,'https://other.example/scene/',BASE);assert.equal(s.entry,null);assert.equal(s.custom,true);assert.equal(s.param,'https://other.example/scene/');
 s=initialScene(i,null,BASE);assert.equal(s.param,'scenes/car/');assert.equal(s.entry.id,'car');
 s=initialScene(null,null,BASE);assert.deepEqual(s,{param:null,entry:null,custom:false});
 s=initialScene(null,'foo/',BASE);assert.equal(s.param,'foo/');assert.equal(s.custom,false);// no list: nothing to be "custom" of
});

test('pickerOptions adds a Custom entry for an unlisted scene',()=>{
 const i=parseSceneIndex(raw);
 assert.deepEqual(pickerOptions(i,'scenes/car/',BASE).map(o=>o.value),['car','synthetic']);
 const o=pickerOptions(i,'my/scene/',BASE);assert.equal(o[0].value,'custom');assert.equal(o[0].label,'Custom: my/scene/');assert.equal(o.length,3);
 assert.deepEqual(pickerOptions(null,null,BASE),[]);
});

test('matchEntry and heavyHint',()=>{
 const i=parseSceneIndex(raw);
 assert.equal(matchEntry(i,'scenes/car/scene.json',BASE).id,'car');assert.equal(matchEntry(i,'nope/',BASE),null);
 assert.equal(heavyHint(i.scenes[1]),'');assert.equal(heavyHint(i.scenes[0]),'large download');
 assert.equal(heavyHint({heavy:'~110 MB download'}),'~110 MB download');
});

test('sceneUrl sets scene, keeps sh/perf, drops per-scene overrides',()=>{
 const href='https://example.com/viewer/?scene=scenes/car/&sh=1&scale=2&cam=1,2,3,4,5,6&xr=none&perf';
 const u=new URL(sceneUrl(href,'scenes/synthetic/',{keep:false}));
 assert.equal(u.searchParams.get('scene'),'scenes/synthetic/');assert.equal(u.searchParams.get('sh'),'1');assert.ok(u.searchParams.has('perf'));
 for(const k of ['scale','cam','xr'])assert.equal(u.searchParams.has(k),false);
 assert.equal(new URL(sceneUrl(href,'a/')).searchParams.get('scale'),'2');
});
