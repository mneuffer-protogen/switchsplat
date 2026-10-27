# SwitchSplat viewer

A WebXR Gaussian-splat viewer for SwitchSplat scenes in which each movable part (door, cabinet door,
drawer) is a persistent rigid object on a 1-DOF joint. Built with Vite, three.js and
[Spark](https://sparkjs.dev) (`@sparkjsdev/spark`). The joint physics is a cannon-free port of
Alyx-Physics' `JointedPart`: click swings a part open or shut, drag (or a VR grip) swings or slides it with
momentum, friction, end stops and a latch, a hand can push it, and it can be locked.

## Run

```sh
npm install
npm run dev:http            # http://localhost:5198, local browser testing
npm run dev -- --host       # https://<lan-ip>:5198 (self-signed) for a headset
npm run build               # static site in dist/
npm test                    # joint, push, lock, spawn, OBB and PLY tests (node:test, no browser)
```

Both dev servers use port 5198 with `strictPort`; stop whatever else is on it first.

The default scene is `public/scenes/synthetic/`. Put other exports under `public/scenes/` or serve them from
anywhere that sends CORS headers.

## URL parameters

| parameter | meaning |
| --- | --- |
| `?scene=<url>` | the export directory (or its `scene.json`); an explicit value always wins. Default: the `default` of `scenes/index.json`, else `scenes/synthetic/` |
| `?scale=<m>` | metres per export unit, overriding `scene.json` `metresPerUnit` (default 1) |
| `?edit` | start in edit mode |
| `?sh=0..3` | cap the view-dependent colour (spherical harmonics) degree, for speed. **Use `?sh=1` on a Quest** with a real (hundreds of thousands of splats) scene; `?sh=0` if it still drops frames |
| `?cam=px,py,pz,tx,ty,tz` | starting camera, world coordinates |
| `?xr=none` or `?xr=x,y,z` | where the scene stands in VR (default: from `scene.json` `view`/`floor` if both are there, else movable parts 0.9 m ahead, floor at y = 0) |
| `?perf` | expose `window.switchsplatPerf` (`start()` / `stop()`): frame times, Spark regenerations and sorts per second, and frames where a moving part was drawn at a stale pose |
| `?emulate` | dev server only: install the IWER Meta Quest 3 emulator, so "Enter VR" works in a desktop browser (`window.xrDevice` drives the emulated controllers) |

## Scene list

`public/scenes/index.json` lists the scenes the **Scene** dropdown at the top of the overlay offers:

```json
{ "version": 1, "default": "car",
  "scenes": [ { "id": "car", "name": "Car", "path": "scenes/car/", "description": "...", "metresPerUnit": 0.586, "heavy": true } ] }
```

`heavy` (true, or a string such as `"~110 MB download"`) is shown next to the name. The file is optional: without it
the viewer loads `?scene=` or `scenes/synthetic/` and shows no dropdown. A `?scene=` that is not in the list appears
as an extra "Custom: ..." entry. Picking a scene loads it without a page reload: the new scene downloads while the
old one stays (hidden, with a Cancel button), then the old one is torn down (persistence flushed, hands released,
edit panel, input listeners, Spark splat arrays) and the new one swaps in. Picking another scene meanwhile aborts the
download in flight. A failed load shows the error with **Retry** and **Cancel** and leaves the previous scene usable.
The address bar follows (`?scene=`, with `sh`, `edit`, `perf`... kept and `scale`/`cam`/`xr` dropped), and
back/forward switch scenes. Poses are remembered per scene.

## Controls

- Desktop: hover outlines a part. **Click** (moved under 5 px) swings it to its other end. **Drag** a
  part to take hold of it where you clicked: it swings about its hinge or slides along its rail, and keeps its
  momentum when you let go. **Shift + move the mouse** over a part pushes it. Drag empty space to orbit,
  right-drag to pan, wheel to zoom. The panel lists the parts with open/close and lock buttons and the state in
  percent; **Reset** closes everything.
- VR: **grip** (squeeze) with the controller inside, or within 6 cm of, a part to grab it; release to let go.
  **Trigger** on or pointing at a part toggles it. Each controller carries a small translucent **hand sphere**
  (4.5 cm) that pushes any part it is not holding, with or without the grip squeezed. **Left thumbstick**
  walks in the floor plane, **right thumbstick** snap-turns by 30 degrees.
  Haptics: grab, end stops, the latch, a hard shut (slam), a pulse on contact scaled by how fast the hand hit,
  and a long firm one when a locked part is grabbed or clicked.
- Edit mode (checkbox, or `?edit`): a panel per part (kind, min/max/open, friction, friction torque, latch,
  mass, locked, a `q` scrub slider), an orange pivot marker with a gizmo to move the pivot and turn the axis,
  and **Download scene.json** for the edited scene. Parts with a weak joint fit (`inlierRatio < 0.6`) carry a
  "low fit" badge.
- Where each part was left, and whether it is locked, is saved in `localStorage` (per scene URL and part id) and
  restored on reload. The lock is only stored when it differs from `joint.locked` in `scene.json`.

### Physics in a few lines

- **Latch.** A part that comes to rest within `latch` of its closed stop latches: its limits close to a hair
  past the stop. It is released only by a **grab** held for 0.25 s (Alyx `doorLatch`: a hand on the handle),
  or by the click/trigger animation. Pushing never unlatches. Closing a part latches it again, also when it is
  slammed shut (it stops dead on the stop, latches, and the hand feels an extra bump).
- **Lock.** `joint.locked` (scene.json, the panel's Lock button, edit mode) pins a part where it stands. Grabbing
  it, clicking it or animating it only **jiggles** it (about 1.5 degrees or 5 mm, decaying over 0.45 s, a haptic
  0.5 for 0.12 s); pushes do nothing. Unlock and it moves, and latches if it is shut.
- **Contact set** (`src/contact.js`). Grab, push and pick use the part's real splat centres, not its box (on a sparse
  scan the box is metres wide). At load, splats with opacity < 0.15, a largest scale > 0.15 m, or fewer than 3 neighbours
  within 8 cm are dropped; the rest go in a 10 cm hash grid (at most 8000 points). A hand grabs a part when its grip
  is within 7 cm of a centre (nearest part wins); the mouse ray and the controller's pointing ray hit centres as
  spheres of 4 / 5 cm. The box (5..95 % trims of the kept centres) is only the hover outline and the mass estimate.
- **Push** (`src/push.js`). The hand sphere is an infinite-mass body. Each fixed step: touching when within hand radius
  + 2.5 cm of a centre; contact point = that centre as posed now, normal = the part's thin PCA axis turned with the
  part and signed toward the hand (hand-to-centre direction when the part is not sheet-like, smallest/middle
  variance > 0.35). Then (a box is used only for parts with no contact set): `jn` = how fast the surface moves along the
  normal per unit of joint speed (`axis x (p - pivot) . n` for a hinge, `axis . n` for a slider). If the hand
  closes on the surface faster than the surface gives way (`sB < sP`), the joint speed becomes `sB / jn`
  (Alyx `holdContacts` with `invMass = 0`), capped (6 rad/s, 3 m/s) and cut to what the limits allow: a latched or
  locked part, or one at its stop, stays put and the hand just stops. A push interrupts a click animation.
- **Swing open.** Click/trigger/Reset use `animateTo(q,{style:'swing'})`: unlatch, a starting shove of 30 % of the
  speed that would just reach the target, then the critically damped drive eases into the end (it cannot
  overshoot, and a grab or a push takes over). Its settling time is sized by the part: about 0.9 s for a car
  door, 0.6 s for a cabinet door, 0.45 s for a drawer.

### Scale

`metresPerUnit` in `scene.json` (default 1; the COLMAP car is 0.586) is applied as the scene root's scale, so the world,
the desktop camera and VR are in metres and the car is life-size. Everything in the file (pivots, floor, view, slider
`min/max/open/latch`, splat positions) stays in export units. The joints also stay in export units and convert their
metric tuning with `unit = metresPerUnit`: a hinge angle is scale-free; a slider's inertia is `mass * unit^2`, its
force limit `N * unit`, and every linear speed or size in metres (slider speed caps, the 4 m/s drag cap, push caps, the
jiggle, the hand sphere 4.5 cm, the 7 cm grab radius, the 2.5 cm contact margin, the 4/5 cm pick radii, the 3 cm hinge-lever
minimum) is divided by it before it meets a coordinate. Hand positions and velocities are taken in the root's frame, so
surface and hand speeds are compared in the same units. Edit mode has a "metres per unit" field (live) and
**Download scene.json** writes it back.

### Real scenes

`scene.json` may carry `view` (`position`, `target`, the initial desktop camera), `floor` (a point on the ground
plane) and, per part, `merged` (`["slot02","slot03"]`, shown in the panel as "merged from ..."). With both `view`
and `floor`, entering VR puts the floor at y = 0 and you standing at `view.position` dropped to the floor, facing
`view.target` horizontally. Big PLYs (100+ MB) show per-file progress (bytes received, then "decoding") on the
loading screen; at most four files download and decode at once, static first. Nothing the viewer does per
frame depends on the splat count: parts move as whole objects, picking and pushing use the oriented boxes only.
On a Quest add `?sh=1` to the URL.

Spark composes every visible `SplatMesh` into one accumulator that it regenerates (every mesh, on the GPU) whenever
any mesh's transform or opacity changes, then sorts. Adding or removing a visible mesh changes the accumulator's
"mapping", and while a mapping change waits for a sort Spark neither regenerates nor shows the new accumulator, so
a moving part freezes for a whole sort and then jumps (about 0.5 s on a 450k-splat scene). The reveal/vanish
meshes therefore stay visible at opacity 0 instead of being hidden, which keeps the mapping fixed for the whole
session.

## Data format

See `articulation-spec.md` in the SwitchSplat export tooling: `scene.json` plus binary little-endian 3DGS PLYs
(`static.ply`, `slotNN_part.ply`, `slotNN_reveal.ply`, ...). Joint coordinate `q = 0` is the stored closed pose,
hinge angles are degrees in the file, `up` is rotated onto +Y. A part with `joint: null` is shown static (its
`base` splats) and cannot be grabbed; a part without `files.part` moves its `files.base` instead.

## Source map

| file | role |
| --- | --- |
| `src/joint.js` | `JointedPart` (reduced-coordinate hinge/slider, latch, lock and jiggle, `drag`, `animateTo`), `FixedStepper` |
| `src/contact.js` | contact set: filtering, grid, nearest/ray queries, grab and pick helpers |
| `src/push.js` | hand sphere against a part's box and the momentum exchange (`pushJoint`), hand velocity |
| `src/spawn.js`, `src/locomotion.js` | VR spawn from `view`/`floor`/`up`; thumbstick walk and snap turn |
| `src/drive.js` | grab drive (grip point stored on the part) and the mouse drag-plane target |
| `src/scene-loader.js` | scene.json + PLYs into Spark `SplatMesh`es, OBB proxies, reveal/vanish fades (`src/fade.js`) |
| `src/scenes.js` | the scene list, default selection and `?scene=` URL handling |
| `src/perf.js` | `?perf` instrumentation |
| `src/ply-centres.js`, `src/obb.js` | splat centres from a PLY and the PCA/percentile oriented box |
| `src/desktop-input.js`, `src/xr-input.js` | mouse and controller interaction |
| `src/edit-mode.js`, `src/hud.js`, `src/persist.js` | edit panel, overlay, saved poses |
| `src/math.js`, `src/spring.js` | helpers copied from Alyx-Physics |
