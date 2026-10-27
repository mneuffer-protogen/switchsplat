# SwitchSplat viewer

A WebXR Gaussian-splat viewer for SwitchSplat scenes in which each movable part (door, cabinet door,
drawer) is a persistent rigid object on a 1-DOF joint. Built with Vite, three.js and
[Spark](https://sparkjs.dev) (`@sparkjsdev/spark`). The joint physics is a cannon-free port of
Alyx-Physics' `JointedPart`: click animates a part open or shut, drag (or a VR grip) swings or slides it with
momentum, friction, end stops and a latch.

## Run

```sh
npm install
npm run dev:http            # http://localhost:5198, local browser testing
npm run dev -- --host       # https://<lan-ip>:5198 (self-signed) for a headset
npm run build               # static site in dist/
npm test                    # joint, OBB and PLY tests (node:test, no browser)
```

Both dev servers use port 5198 with `strictPort`; stop whatever else is on it first.

The default scene is `public/scenes/synthetic/`. Put other exports under `public/scenes/` or serve them from
anywhere that sends CORS headers.

## URL parameters

| parameter | meaning |
| --- | --- |
| `?scene=<url>` | the export directory (or its `scene.json`); default `scenes/synthetic/` |
| `?edit` | start in edit mode |
| `?sh=0..3` | cap the view-dependent colour (spherical harmonics) degree, for speed |
| `?cam=px,py,pz,tx,ty,tz` | starting camera, world coordinates |
| `?xr=none` or `?xr=x,y,z` | where the scene stands in VR (default: movable parts 0.9 m ahead, floor at y = 0) |
| `?emulate` | dev server only: install the IWER Meta Quest 3 emulator, so "Enter VR" works in a desktop browser (`window.xrDevice` drives the emulated controllers) |

## Controls

- Desktop: hover outlines a part. **Click** (moved under 5 px) animates it to its other end. **Drag** a
  part to take hold of it where you clicked: it swings about its hinge or slides along its rail, and keeps its
  momentum when you let go. Drag empty space to orbit, right-drag to pan, wheel to zoom. The panel lists the
  parts with open/close buttons and the state in percent; **Reset** closes everything.
- VR: **grip** (squeeze) with the controller inside, or within 6 cm of, a part to grab it; release to let go.
  **Trigger** on or pointing at a part toggles it. Haptics on grab, end stops and the latch.
- Edit mode (checkbox, or `?edit`): a panel per part (kind, min/max/open, friction, friction torque, latch,
  mass, a `q` scrub slider), an orange pivot marker with a gizmo to move the pivot and turn the axis, and
  **Download scene.json** for the edited scene. Parts with a weak joint fit (`inlierRatio < 0.6`) carry a
  "low fit" badge.
- Where each part was left is saved in `localStorage` (per scene URL and part id) and restored on reload.

## Data format

See `articulation-spec.md` in the SwitchSplat export tooling: `scene.json` plus binary little-endian 3DGS PLYs
(`static.ply`, `slotNN_part.ply`, `slotNN_reveal.ply`, ...). Joint coordinate `q = 0` is the stored closed pose,
hinge angles are degrees in the file, `up` is rotated onto +Y. A part with `joint: null` is shown static (its
`base` splats) and cannot be grabbed; a part without `files.part` moves its `files.base` instead.

## Source map

| file | role |
| --- | --- |
| `src/joint.js` | `JointedPart` (reduced-coordinate hinge/slider, latch, `drag`, `animateTo`), `FixedStepper` |
| `src/drive.js` | grab drive (grip point stored on the part) and the mouse drag-plane target |
| `src/scene-loader.js` | scene.json + PLYs into Spark `SplatMesh`es, OBB proxies, reveal/vanish fades |
| `src/ply-centres.js`, `src/obb.js` | splat centres from a PLY and the PCA/percentile oriented box |
| `src/desktop-input.js`, `src/xr-input.js` | mouse and controller interaction |
| `src/edit-mode.js`, `src/hud.js`, `src/persist.js` | edit panel, overlay, saved poses |
| `src/math.js`, `src/spring.js` | helpers copied from Alyx-Physics |
