# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import a trained scene from the public viewer's ``.gutb`` format into the articulation export layout.

The hosted SwitchSplat viewer streams every scene as a ``.gutb`` file (optionally gzipped):

    bytes 0-7    "GUTB0001"
    bytes 8-11   uint32 LE header length H
    bytes 12..   JSON header (count, shDegree, shEncoding, switchCount, kernelDegree, sections, ...)
    then         the sections, ``offset`` counted from the end of the header:

    geometry   N x 48 bytes = 12 x float32 (three vec4f):  x y z opacity | qw qx qy qz | sx sy sz 0 (pad)
               opacity is the ACTIVATED density in 0..1, scale is LINEAR (sigma along each axis),
               the quaternion is (w, x, y, z) (the WGSL ``gutQuatToMat3`` reads r = q.x).
    sh         N x 56 bytes for shEncoding "dc-f16/rest-i8": uint16 dc_r, dc_g, dc_b (float16),
               float16 restScale, then 45 int8 (x restScale), coefficient-major (c1_r c1_g c1_b c2_r ...)
               and 3 padding bytes.  "f16": 48 float16, "f32": 48 float32, both coefficient-major with the
               DC triplet first.  The DC triplet is the raw SH coefficient (``rgb = 0.5 + C0 * dc``).
    switches   N x switchCount float16 probabilities in SwitchSplat state order
               (0 static, 2k-1 slot k base, 2k slot k switched); they sum to ~1 per gaussian and the
               viewer shows a gaussian with alpha = opacity * (sum of the enabled states' probabilities).

The decoded gaussians are converted to the PLY convention of ``ply_io`` (logit opacity, log scale, wxyz,
channel-major f_rest), split by state with ``export_parts.split_by_state`` and written like
``export_parts.py`` does; the manifest entry of the hosted viewer (``index.js``) supplies part names, ``up``
and the initial view, and the floor is estimated from the static gaussians around the parts.

Run:  python -m tools.articulation.import_gutb scene.gutb.gz --out DIR [--manifest-js index.js --scene-id best-car]
then: python -m tools.articulation.estimate_joint DIR
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import export_parts
from .ply_io import Splats

MAGIC = b"GUTB0001"
OPACITY_EPS = 1.0e-5  # activated opacity is clipped to [eps, 1 - eps] before the logit
SCALE_FLOOR = 1.0e-9
SH_DC_WORDS = 3  # first three SH values are the DC triplet
SH_FLOATS = 48  # 16 coefficients x 3 channels (degree 3 layout, always stored in full)
SH_BYTES = {"f32": 192, "f16": 96, "dc-f16/rest-i8": 56}


# --------------------------------------------------------------------------------------------
# .gutb decoding
# --------------------------------------------------------------------------------------------

@dataclass
class Gutb:
    header: dict
    geometry: np.ndarray  # float32 [N, 12]
    sh: np.ndarray  # raw bytes [N, bytes_per_gaussian] (uint8)
    switches: np.ndarray | None  # float32 [N, switchCount]

    @property
    def count(self) -> int:
        return int(self.header["count"])


def _open(path: Path):
    path = Path(path)
    with open(path, "rb") as probe:
        gzipped = probe.read(2) == b"\x1f\x8b"
    return gzip.open(path, "rb") if gzipped else open(path, "rb")


def _read_exact(stream, n: int) -> bytes:
    """Read exactly n bytes (gzip streams may return short reads for large n)."""
    chunks, remaining = [], n
    while remaining > 0:
        chunk = stream.read(min(remaining, 1 << 28))
        if not chunk:
            raise ValueError("scene file ended early")
        chunks.append(chunk)
        remaining -= len(chunk)
    return chunks[0] if len(chunks) == 1 else b"".join(chunks)


def read_gutb(path: str | Path) -> Gutb:
    """Stream-decode a ``.gutb`` / ``.gutb.gz`` file: sections are read once and viewed with numpy."""
    with _open(path) as stream:
        head = _read_exact(stream, 12)
        if head[:8] != MAGIC:
            raise ValueError("not a .gutb scene file")
        header = json.loads(_read_exact(stream, struct.unpack("<I", head[8:12])[0]).decode("utf-8"))
        n = int(header["count"])
        encoding = header.get("shEncoding", "f32")
        if encoding not in SH_BYTES:
            raise ValueError(f"unknown SH encoding {encoding!r}")
        position = 0
        out: dict[str, np.ndarray] = {}
        for section in sorted(header["sections"], key=lambda s: s["offset"]):
            if section["offset"] < position:
                raise ValueError("overlapping sections")
            if section["offset"] > position:
                _read_exact(stream, section["offset"] - position)
            raw = _read_exact(stream, section["bytes"])
            position = section["offset"] + section["bytes"]
            name = section["name"]
            if name == "geometry":
                if len(raw) != n * 48:
                    raise ValueError(f"geometry section has {len(raw)} bytes, expected {n * 48}")
                out["geometry"] = np.frombuffer(raw, dtype="<f4").reshape(n, 12)
            elif name == "sh":
                width = SH_BYTES[encoding]
                if len(raw) != n * width:
                    raise ValueError(f"sh section has {len(raw)} bytes, expected {n * width}")
                out["sh"] = np.frombuffer(raw, dtype=np.uint8).reshape(n, width)
            elif name == "switches":
                k = int(header["switchCount"])
                if len(raw) != n * k * 2:
                    raise ValueError(f"switches section has {len(raw)} bytes, expected {n * k * 2}")
                out["switches"] = np.frombuffer(raw, dtype="<f2").reshape(n, k).astype(np.float32)
    if "geometry" not in out or "sh" not in out:
        raise ValueError("scene file has no geometry or sh section")
    return Gutb(header, out["geometry"], out["sh"], out.get("switches"))


def decode_geometry(geometry: np.ndarray):
    """[N, 12] float32 (x y z a | qw qx qy qz | sx sy sz pad) -> positions [N,3], activated opacity [N], quaternion wxyz [N,4], linear scale [N,3]."""
    g = np.asarray(geometry, dtype=np.float32)
    return g[:, 0:3], g[:, 3], g[:, 4:8], g[:, 8:11]


def decode_sh(raw: np.ndarray, encoding: str, degree: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Raw SH bytes -> (f_dc [N,3], f_rest [N, 3*((degree+1)^2-1)] channel-major, all float32).

    The stored layout is coefficient-major (c1_r c1_g c1_b c2_r ...); PLY wants all R, then all G, then all B.
    """
    n = len(raw)
    if encoding == "dc-f16/rest-i8":
        head = np.ascontiguousarray(raw[:, :8]).view("<f2").astype(np.float32)  # dc r, g, b, restScale
        f_dc, scale = head[:, :3], head[:, 3:4]
        values = np.ascontiguousarray(raw[:, 8:8 + 45]).view(np.int8).astype(np.float32) * scale  # [N, 45]
    elif encoding == "f16":
        flat = np.ascontiguousarray(raw).view("<f2").astype(np.float32)  # [N, 48]
        f_dc, values = flat[:, :3], flat[:, 3:]
    elif encoding == "f32":
        flat = np.ascontiguousarray(raw).view("<f4")
        f_dc, values = flat[:, :3].copy(), flat[:, 3:]
    else:
        raise ValueError(f"unknown SH encoding {encoding!r}")
    coeffs = (degree + 1) ** 2 - 1
    f_rest = values.reshape(n, 15, 3)[:, :coeffs].transpose(0, 2, 1).reshape(n, 3 * coeffs)
    return np.ascontiguousarray(f_dc, dtype=np.float32), np.ascontiguousarray(f_rest, dtype=np.float32)


def to_splats(gutb: Gutb, sh_degree: int | None = None) -> Splats:
    """All gaussians in PLY (pre-activation) units."""
    degree = int(gutb.header.get("shDegree", 3))
    degree = degree if sh_degree is None else min(int(sh_degree), degree)
    pos, opacity, quat, scale = decode_geometry(gutb.geometry)
    f_dc, f_rest = decode_sh(gutb.sh, gutb.header.get("shEncoding", "f32"), degree)
    alpha = np.clip(opacity.astype(np.float64), OPACITY_EPS, 1.0 - OPACITY_EPS)
    return Splats(
        positions=pos,
        f_dc=f_dc,
        f_rest=f_rest,
        opacity=np.log(alpha / (1.0 - alpha)),
        scale=np.log(np.maximum(scale, SCALE_FLOOR)),
        rot=quat,
    )


def probabilities_to_logits(probs: np.ndarray) -> np.ndarray:
    """``split_by_state`` takes logits; log of the stored (already normalised) probabilities is one."""
    return np.log(np.clip(np.asarray(probs, dtype=np.float64), 1.0e-9, None))


# --------------------------------------------------------------------------------------------
# Sanity statistics
# --------------------------------------------------------------------------------------------

def describe(splats: Splats, probs: np.ndarray | None = None) -> str:
    """Bounding box, mean colour, opacity histogram and scale statistics, as text."""
    pos = splats.positions
    alpha = 1.0 / (1.0 + np.exp(-splats.opacity[:, 0].astype(np.float64)))
    scale = np.exp(splats.scale)
    qnorm = np.linalg.norm(splats.rot, axis=1)
    rgb = np.clip(splats.rgb, 0.0, 1.0)
    lo, hi = np.percentile(pos, 1, axis=0), np.percentile(pos, 99, axis=0)
    lines = [
        f"gaussians: {len(splats)}   f_rest columns: {splats.f_rest.shape[1]}",
        f"bbox min {np.round(pos.min(axis=0), 2)}  max {np.round(pos.max(axis=0), 2)}",
        f"bbox 1%-99% {np.round(lo, 2)} .. {np.round(hi, 2)}  (extent {np.round(hi - lo, 2)})",
        f"mean colour (SH DC) {np.round(rgb.mean(axis=0), 3)}  mean f_dc {np.round(splats.f_dc.mean(axis=0), 3)}",
        f"opacity (activated) mean {alpha.mean():.3f}  median {np.median(alpha):.3f}  "
        f">0.9: {100 * (alpha > 0.9).mean():.1f}%  <0.1: {100 * (alpha < 0.1).mean():.1f}%",
        "opacity histogram (10 bins 0..1): " + " ".join(f"{int(c)}" for c in np.histogram(alpha, bins=10, range=(0, 1))[0]),
        f"scale (linear) median {np.median(scale):.4f}  p1 {np.percentile(scale, 1):.4f}  p99 {np.percentile(scale, 99):.4f}  "
        f"max {scale.max():.3f}",
        f"quaternion norm min {qnorm.min():.4f} median {np.median(qnorm):.4f} max {qnorm.max():.4f}",
    ]
    if probs is not None:
        sums = probs.sum(axis=1)
        assign = probs.argmax(axis=1)
        counts = np.bincount(assign, minlength=probs.shape[1])
        lines.append(f"switch prob sum min {sums.min():.3f} median {np.median(sums):.3f} max {sums.max():.3f}")
        lines.append("argmax state counts " + " ".join(f"s{i}:{int(c)}" for i, c in enumerate(counts)))
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# Manifest (index.js) parsing: a tolerant JS object-literal reader, nothing is executed
# --------------------------------------------------------------------------------------------

class _Reader:
    NUMBER = re.compile(r"-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
    IDENT = re.compile(r"[A-Za-z_$][\w$]*")

    def __init__(self, text: str, pos: int = 0):
        self.text, self.pos = text, pos

    def skip(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos] in " \t\r\n":
            self.pos += 1

    def peek(self) -> str:
        self.skip()
        return self.text[self.pos] if self.pos < len(self.text) else ""

    def expect(self, ch: str) -> None:
        if self.peek() != ch:
            raise ValueError(f"expected {ch!r} at {self.pos}: {self.text[self.pos:self.pos + 40]!r}")
        self.pos += 1

    def string(self) -> str:
        quote = self.text[self.pos]
        self.pos += 1
        out = []
        while self.text[self.pos] != quote:
            ch = self.text[self.pos]
            if ch == "\\":
                nxt = self.text[self.pos + 1]
                out.append({"n": "\n", "t": "\t", "r": "\r"}.get(nxt, nxt))
                self.pos += 2
            else:
                out.append(ch)
                self.pos += 1
        self.pos += 1
        return "".join(out)

    def skip_balanced(self) -> None:
        """Skip a (...) / [...] / {...} group, honouring string literals."""
        depth = 0
        while self.pos < len(self.text):
            ch = self.text[self.pos]
            if ch in "`'\"":
                self.string()
                continue
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
                if depth == 0:
                    self.pos += 1
                    return
            self.pos += 1
        raise ValueError("unbalanced group")

    def value(self):
        ch = self.peek()
        if ch == "{":
            return self.obj()
        if ch == "[":
            return self.array()
        if ch in "`'\"":
            return self.string()
        if self.text.startswith("!0", self.pos):
            self.pos += 2
            return True
        if self.text.startswith("!1", self.pos):
            self.pos += 2
            return False
        m = self.NUMBER.match(self.text, self.pos)
        if m:
            self.pos = m.end()
            return float(m.group())
        m = self.IDENT.match(self.text, self.pos)
        if m:
            self.pos = m.end()
            word = m.group()
            if word in ("true", "false", "null"):
                return {"true": True, "false": False, "null": None}[word]
            if word == "void":
                self.peek()
                self.pos += 1  # the operand digit
                return None
            while self.text[self.pos:self.pos + 1] in ("(", "[", "."):  # a call / member expression: not evaluated
                if self.text[self.pos] == ".":
                    member = self.IDENT.match(self.text, self.pos + 1)
                    if not member:
                        break
                    self.pos = member.end()
                else:
                    self.skip_balanced()
            return None
        raise ValueError(f"cannot parse value at {self.pos}: {self.text[self.pos:self.pos + 40]!r}")

    def array(self) -> list:
        self.expect("[")
        out = []
        while self.peek() != "]":
            out.append(self.value())
            if self.peek() == ",":
                self.pos += 1
        self.pos += 1
        return out

    def obj(self) -> dict:
        self.expect("{")
        out = {}
        while self.peek() != "}":
            ch = self.peek()
            if ch in "`'\"":
                key = self.string()
            else:
                m = self.IDENT.match(self.text, self.pos) or self.NUMBER.match(self.text, self.pos)
                if not m:
                    raise ValueError(f"bad object key at {self.pos}: {self.text[self.pos:self.pos + 40]!r}")
                key, self.pos = m.group(), m.end()
            self.expect(":")
            out[key] = self.value()
            if self.peek() == ",":
                self.pos += 1
        self.pos += 1
        return out


def parse_js_literal(text: str):
    """Parse one JS object / array literal (minified style: unquoted keys, backtick strings, ``!0``/``!1``)."""
    return _Reader(text).value()


def find_manifest_entry(js_text: str, scene_id: str) -> dict:
    """The scene entry ``{id:`scene_id`, ...}`` of the viewer bundle as a dict (not executed, only parsed)."""
    pattern = re.compile(r"\{\s*id\s*:\s*[`'\"]" + re.escape(scene_id) + r"[`'\"]\s*,")
    for match in pattern.finditer(js_text):
        try:
            entry = _Reader(js_text, match.start()).obj()
        except (ValueError, IndexError):
            continue
        if "camera" in entry or "switches" in entry or "name" in entry:
            return entry
    raise KeyError(f"no scene entry with id {scene_id!r} found")


def read_manifest(path: Path, scene_id: str) -> dict:
    return find_manifest_entry(Path(path).read_text(encoding="utf-8", errors="replace"), scene_id)


def names_from_manifest(entry: dict) -> list[str]:
    names = []
    for i, switch in enumerate(entry.get("switches") or []):
        label = ((switch.get("semantic") or {}).get("label") or "").strip()
        names.append(label or switch.get("label") or f"object {i + 1}")
    return names


def unit(vector, default=(0.0, -1.0, 0.0)) -> list[float]:
    v = np.asarray(vector if vector is not None else default, dtype=np.float64)
    norm = np.linalg.norm(v)
    return [float(x) for x in (v / norm if norm > 1e-12 else np.asarray(default, dtype=np.float64))]


# --------------------------------------------------------------------------------------------
# Floor estimation
# --------------------------------------------------------------------------------------------

def estimate_floor(points: np.ndarray, part_positions: np.ndarray, up, radius: float = 6.0,
                   bin_size: float | None = None) -> tuple[np.ndarray, float] | None:
    """(a point on the ground plane under the parts, ground thickness) or None without enough support.

    ``points`` are the scene's gaussian centres (static and part gaussians alike), ``part_positions`` the
    centres of the parts. Heights are projected on ``up`` (larger = higher). Gaussians within ``radius``
    (horizontally) of the parts' median centre vote in a height histogram, weighted by a gaussian fall-off
    with the horizontal distance (the ground is rarely perfectly level, so the nearby ground counts most);
    only heights at or below the parts' median height are considered, and the ground is the most populated
    bin there: the biggest horizontal slab beneath the parts. The thickness is the robust (MAD) spread of the
    heights around the peak. The returned point sits under the parts' median centre.
    """
    up = np.asarray(up, dtype=np.float64)
    up = up / np.linalg.norm(up)
    parts = np.asarray(part_positions, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    if len(parts) < 3 or len(points) < 50:
        return None
    centre = np.median(parts, axis=0)
    part_h = parts @ up
    span = max(np.percentile(part_h, 99) - np.percentile(part_h, 1), 1e-3)
    ceiling = float(np.median(part_h))

    rel = points - centre
    height = points @ up
    horizontal = np.linalg.norm(rel - np.outer(rel @ up, up), axis=1)
    keep = (horizontal <= radius) & (height <= ceiling)
    height, horizontal = height[keep], horizontal[keep]
    if len(height) < 50:
        return None
    weight = np.exp(-0.5 * (horizontal / (0.5 * radius)) ** 2)
    if bin_size is None:
        bin_size = max(span / 150.0, 1e-4)
    edges = np.arange(height.min(), height.max() + 2 * bin_size, bin_size)
    hist, edges = np.histogram(height, bins=edges, weights=weight)
    smooth = np.convolve(hist, np.ones(5), mode="same")  # a slightly tilted ground still gives one peak
    peak = int(np.argmax(smooth))
    centre_h = 0.5 * (edges[peak] + edges[peak + 1])
    window = np.abs(height - centre_h) <= 6.0 * bin_size
    h_win, w_win = height[window], weight[window]
    order = np.argsort(h_win)
    cum = np.cumsum(w_win[order])
    floor_h = float(h_win[order][np.searchsorted(cum, 0.5 * cum[-1])])  # weighted median
    mad = float(np.median(np.abs(h_win - floor_h)))
    thickness = max(1.4826 * mad, bin_size)
    return centre + up * (floor_h - float(centre @ up)), thickness


# --------------------------------------------------------------------------------------------
# Import
# --------------------------------------------------------------------------------------------

def demote_ground(logits: np.ndarray, positions: np.ndarray, slot_mask: np.ndarray, floor: np.ndarray, up,
                  margin: float) -> tuple[np.ndarray, np.ndarray]:
    """Make part gaussians that lie on the ground static; returns (new logits, demoted mask).

    The state labels of a real scene leak onto the ground around a part (the ground texture is visible in
    both the closed and the open state). Such gaussians are no part of a rigid body, and would register at
    the identity and hide the real motion. Slot gaussians less than ``margin`` above the floor become static.
    """
    up = np.asarray(up, dtype=np.float64)
    up = up / np.linalg.norm(up)
    above = (positions.astype(np.float64) - np.asarray(floor, dtype=np.float64)) @ up
    demoted = slot_mask & (above < margin)
    out = np.array(logits, dtype=np.float64, copy=True)
    out[demoted, 0] = out.max() + 50.0
    return out, demoted


def import_gutb(path: Path, out: Path, *, manifest_js: Path | None = None, scene_id: str | None = None,
                min_confidence: float = 0.6, sh_degree: int | None = None, name: str | None = None,
                up=None, ground_margin: float | None = None, verbose: bool = True) -> dict:
    """Decode, split by state, write the export directory and return the scene.json dictionary.

    ``ground_margin`` (same units as the scene; default: 1.25 x the ground layer's thickness, at least 3 % of the
    parts' height span) is the height above the estimated floor under which part gaussians count as ground.
    """
    gutb = read_gutb(path)
    if gutb.switches is None:
        raise SystemExit(f"{path} has no switches section; nothing to split")
    entry = read_manifest(manifest_js, scene_id) if manifest_js and scene_id else {}
    camera = entry.get("camera") or {}

    splats = to_splats(gutb, sh_degree)
    if verbose:
        print(describe(splats, gutb.switches), flush=True)
    logits = probabilities_to_logits(gutb.switches)
    names = names_from_manifest(entry) or None
    if up is None:
        up = unit(camera.get("up"))
    scene_name = name or entry.get("name") or Path(path).name.split(".")[0]

    # floor from every gaussian around the parts, then drop ground that leaked into the part sets
    split = export_parts.split_by_state(logits, min_confidence)
    slot_mask = np.zeros(len(splats), dtype=bool)
    for base_mask, switched_mask in split.slots.values():
        slot_mask |= base_mask | switched_mask
    found = estimate_floor(splats.positions, splats.positions[slot_mask], up)
    floor = None
    if found is not None:
        floor, thickness = found
        part_h = splats.positions[slot_mask].astype(np.float64) @ np.asarray(up)
        span = float(np.percentile(part_h, 99) - np.percentile(part_h, 1))
        margin = ground_margin if ground_margin is not None else max(1.25 * thickness, 0.03 * span)
        logits, demoted = demote_ground(logits, splats.positions, slot_mask, floor, up, margin)
        if verbose:
            print(f"floor point {np.round(floor, 4).tolist()} (up {np.round(up, 4).tolist()}), ground thickness "
                  f"{thickness:.3f}, margin {margin:.3f}: {int(demoted.sum())} part gaussians moved to static", flush=True)
    elif verbose:
        print("warning: could not estimate a floor", flush=True)

    scene = export_parts.export_parts(splats, logits, out, min_confidence=min_confidence, up=up, names=names,
                                      name=scene_name)
    # view and floor (spec addendum v1.1)
    if isinstance(camera.get("position"), list) and isinstance(camera.get("target"), list):
        scene["view"] = {"position": [round(float(x), 5) for x in camera["position"]],
                         "target": [round(float(x), 5) for x in camera["target"]]}
    if floor is not None:
        scene["floor"] = [round(float(x), 5) for x in floor]
    scene["source"] = {"format": "gutb", "file": Path(path).name, "id": scene_id,
                       "note": "decoded from the SwitchSplat hosted viewer dataset (CC BY 4.0)"}
    (out / "scene.json").write_text(json.dumps(scene, indent=2) + "\n", encoding="utf-8")
    return scene


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("gutb", type=Path, help="scene.gutb or scene.gutb.gz")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--manifest-js", type=Path, help="the hosted viewer's index.js (part names, up, view)")
    parser.add_argument("--scene-id", help="scene id inside the manifest, e.g. best-car")
    parser.add_argument("--min-confidence", type=float, default=0.6,
                        help="drop slot gaussians whose top state probability is lower (default 0.6)")
    parser.add_argument("--sh-degree", type=int, default=None,
                        help="keep SH up to this degree (default: all stored, usually 3)")
    parser.add_argument("--name", help="scene name (default: the manifest name)")
    parser.add_argument("--up", type=float, nargs=3, metavar=("X", "Y", "Z"), help="override the up vector")
    parser.add_argument("--ground-margin", type=float, default=None,
                        help="part gaussians below floor + this height become static (default: automatic)")
    args = parser.parse_args(argv)
    if bool(args.manifest_js) != bool(args.scene_id):
        parser.error("--manifest-js and --scene-id go together")
    scene = import_gutb(args.gutb, args.out, manifest_js=args.manifest_js, scene_id=args.scene_id,
                        min_confidence=args.min_confidence, sh_degree=args.sh_degree, name=args.name,
                        up=args.up, ground_margin=args.ground_margin)
    print(f"wrote {args.out} with {len(scene['parts'])} parts", flush=True)


if __name__ == "__main__":
    main()
