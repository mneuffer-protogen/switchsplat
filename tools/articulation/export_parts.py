# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export a trained SwitchSplat checkpoint as static + per-slot base/switched PLY sets.

Every gaussian is assigned to its most probable switch state: 0 = static background, 2k-1 = slot k
base state, 2k = slot k switched state (the argmax of the softmax of ``switch_logits``). The
gaussians are written exactly like ``threedgrut/export/ply_exporter.py`` writes them (raw,
pre-activation values: logit opacity, log scale, unnormalised quaternion, every stored SH
coefficient). Slot sets drop gaussians whose top probability is below --min-confidence and
statistical outliers; the static set keeps every state-0 gaussian.

Soft membership (``soft_threshold`` > 0, the importer's default): the hosted viewer draws a gaussian with
alpha = opacity x (sum of the enabled states' probabilities), so a gaussian that is 40 % static and 60 % slot-1
base is half there in every state. A hard argmax throws that mass away. ``soft_split`` instead emits every
gaussian in each file whose state probability is at least the threshold, with the opacity scaled by the mass
the file stands for:

* a gaussian is emitted once per *state* with p >= threshold (static, slot-k base, slot-k switched), so a
  single gaussian appears twice only when two states each carry real mass (static copy x p_static, part copy
  x p_base); the argmax state is always emitted;
* the multiplier of an emitted copy is its state's probability, and the most probable emitted copy of the
  default render (static + all base states) also carries the probability that was *not* emitted but is drawn
  there (p_static below the threshold, small base probabilities): the closed render then shows exactly
  p_static + sum(p_base) of the gaussian. A copy of the switched state of slot k stands for the open render of
  slot k (static + the other bases + its own switched state): it takes the dropped mass too when no emitted copy
  of that render carries it. A gaussian with one clear owner keeps (nearly) its full opacity, the copies of a
  split gaussian carry their shares;
* multipliers above 0.97 are snapped to 1 (float16 probabilities sum to 0.9996, there is nothing to correct).

The estimator still fits the joint on the confident sets only (argmax state with p >= --min-confidence,
statistical outliers removed), which is stored per slot in ``slotNN_membership.npz`` (``base_fit`` /
``switched_fit`` row flags, ``base_mult`` / ``switched_mult`` the opacity multipliers that were applied).

Writes OUT/static.ply, OUT/slotNN_{base,switched}.ply and OUT/scene.json (joints null). Then run
``python -m tools.articulation.estimate_joint OUT``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .ply_io import Splats, write_ply


@dataclass
class StateSplit:
    """Boolean masks over the gaussians; ``slots[k]`` is (base, switched) of 1-based slot k."""

    static: np.ndarray
    slots: dict[int, tuple[np.ndarray, np.ndarray]]
    confidence: np.ndarray
    assignment: np.ndarray


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def split_by_state(logits: np.ndarray, min_confidence: float = 0.6) -> StateSplit:
    """Assign gaussians to states from ``switch_logits`` [N, 1 + 2K] (argmax of the softmax).

    The static set is every gaussian whose argmax is 0, whatever its confidence. A slot's base /
    switched set holds the gaussians whose argmax is its state and whose top probability is at
    least ``min_confidence``; ambiguous gaussians that are not static are dropped.
    """
    logits = np.asarray(logits)
    if logits.ndim != 2 or logits.shape[1] < 3 or logits.shape[1] % 2 != 1:
        raise ValueError(f"switch_logits must have shape [N, 1 + 2K] with K >= 1, got {logits.shape}")
    probs = softmax(logits)
    assignment = probs.argmax(axis=-1)
    confidence = probs.max(axis=-1)
    confident = confidence >= min_confidence
    slots = {
        slot: ((assignment == 2 * slot - 1) & confident, (assignment == 2 * slot) & confident)
        for slot in range(1, (logits.shape[1] - 1) // 2 + 1)
    }
    return StateSplit(assignment == 0, slots, confidence, assignment)


@dataclass
class SoftSplit:
    """Per-gaussian opacity multipliers of every emitted copy (0 = the file does not hold the gaussian)."""

    static: np.ndarray  # [N]
    base: dict[int, np.ndarray]  # slot -> [N]
    switched: dict[int, np.ndarray]  # slot -> [N]


SNAP_MULTIPLIER = 0.97


def soft_split(probs: np.ndarray, threshold: float = 0.1) -> SoftSplit:
    """Soft state membership of gaussians from their state probabilities [N, 1 + 2K] (see the module docstring)."""
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] < 3 or probs.shape[1] % 2 != 1:
        raise ValueError(f"probabilities must have shape [N, 1 + 2K] with K >= 1, got {probs.shape}")
    n, states = probs.shape
    emitted = probs >= threshold
    emitted[np.arange(n), probs.argmax(axis=1)] = True
    num_slots = (states - 1) // 2
    closed = [0] + [2 * k - 1 for k in range(1, num_slots + 1)]  # states drawn by the default (closed) render
    dropped = probs * ~emitted
    dropped_closed = dropped[:, closed].sum(axis=1)  # mass of closed-render states that get no copy
    candidates = np.where(emitted[:, closed], probs[:, closed], -1.0)
    carrier = np.asarray(closed)[candidates.argmax(axis=1)]  # the closed copy that carries that mass (most probable)
    has_carrier = candidates.max(axis=1) >= 0.0

    def finish(mult: np.ndarray, mask: np.ndarray) -> np.ndarray:
        mult = np.where(mask, np.minimum(mult, 1.0), 0.0)
        return np.where(mult >= SNAP_MULTIPLIER, 1.0, mult)

    static = finish(probs[:, 0] + np.where(has_carrier & (carrier == 0), dropped_closed, 0.0), emitted[:, 0])
    base, switched = {}, {}
    for k in range(1, num_slots + 1):
        b, s = 2 * k - 1, 2 * k
        base[k] = finish(probs[:, b] + np.where(has_carrier & (carrier == b), dropped_closed, 0.0), emitted[:, b])
        # the open state of slot k replaces its base state: a dropped p_base_k is not drawn next to p_switched_k, and
        # the rest of the dropped closed mass is drawn there unless a copy that is part of that render carries it
        in_open_render = has_carrier & (carrier != b)
        switched[k] = finish(probs[:, s] + np.where(in_open_render, 0.0, dropped_closed - dropped[:, b]), emitted[:, s])
    return SoftSplit(static, base, switched)


def scale_opacity(splats: Splats, multiplier: np.ndarray) -> Splats:
    """``splats`` with alpha = multiplier x sigmoid(opacity) (back in logit units); rows with 1 are untouched."""
    multiplier = np.asarray(multiplier, dtype=np.float64)
    opacity = splats.opacity[:, 0].astype(np.float64)
    change = multiplier < 1.0
    if change.any():
        alpha = 1.0 / (1.0 + np.exp(-opacity[change])) * multiplier[change]
        alpha = np.clip(alpha, 1.0e-5, 1.0 - 1.0e-5)
        opacity[change] = np.log(alpha / (1.0 - alpha))
    return Splats(splats.positions, splats.f_dc, splats.f_rest, opacity, splats.scale, splats.rot)


def membership_path(out: Path, slot_id: str) -> Path:
    return Path(out) / f"{slot_id}_membership.npz"


def outlier_mask(points: np.ndarray, k: int = 8, std_ratio: float = 2.0) -> np.ndarray:
    """Statistical outlier filter: True for points whose mean k-NN distance is <= mean + std_ratio * std."""
    if len(points) <= k + 1:
        return np.ones(len(points), dtype=bool)
    dist, _ = cKDTree(points).query(points, k=k + 1, workers=-1)
    mean_dist = dist[:, 1:].mean(axis=1)
    return mean_dist <= mean_dist.mean() + std_ratio * mean_dist.std()


def splats_from_checkpoint(checkpoint: dict) -> Splats:
    """The checkpoint's raw tensors as PLY columns, following ``PLYExporter.export`` exactly.

    Positions, density, scale and rotation are the parameters themselves (the exporter reads them
    with ``preactivation=True``). features_specular [N, 3 * num_speculars] is viewed as
    (N, num_speculars, 3), transposed to channel-major and flattened. All stored coefficients are
    written; ``n_active_features`` (progressive training state) is not used, as in the exporter.
    """
    if "features_albedo" not in checkpoint:
        raise SystemExit("checkpoint has no SH features (features_albedo); NHT checkpoints cannot be exported")

    def array(name: str) -> np.ndarray:
        return checkpoint[name].detach().cpu().float().numpy()

    positions = array("positions")
    n = len(positions)
    specular = array("features_specular")
    num_speculars = specular.shape[1] // 3
    expected = (int(checkpoint["max_n_features"]) + 1) ** 2 - 1
    if specular.shape[1] != 3 * num_speculars or num_speculars != expected:
        raise SystemExit(
            f"features_specular has {specular.shape[1]} columns, expected 3 * {expected} for SH degree "
            f"{checkpoint['max_n_features']}"
        )
    f_rest = specular.reshape(n, num_speculars, 3).transpose(0, 2, 1).reshape(n, num_speculars * 3)
    return Splats(
        positions=positions,
        f_dc=array("features_albedo"),
        f_rest=f_rest,
        opacity=array("density").reshape(n, 1),
        scale=array("scale"),
        rot=array("rotation"),
    )


def load_checkpoint(path: Path) -> tuple[Splats, np.ndarray]:
    """Gaussians and switch logits [N, 1 + 2K] of a SwitchSplat checkpoint (the model is not built)."""
    import torch  # imported here so the numpy logic above works without torch

    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    if "switch_logits" not in checkpoint:
        raise SystemExit(f"{path} has no switch_logits; it is not a SwitchSplat checkpoint")
    splats = splats_from_checkpoint(checkpoint)
    logits = checkpoint["switch_logits"].detach().cpu().float().numpy()
    if len(logits) != len(splats):
        raise SystemExit(f"switch_logits has {len(logits)} rows but the checkpoint has {len(splats)} gaussians")
    return splats, logits


def names_from_observation(path: Path, num_slots: int) -> list[str]:
    """Part names from a VLM raw_observation.json, in slot order.

    Slots follow the interaction windows, which are the observation's events after the paper's
    post-processing (short events dropped, overlapping ones suppressed) sorted by (on_frame,
    off_frame). The same post-processing is applied here; if the event count still differs from the
    number of slots the names are unreliable and the unmatched slots are called "object N".
    """
    events = json.loads(Path(path).read_text(encoding="utf-8"))["events"]
    try:
        from tools.vlm_windows.postprocess import process_events

        events, _ = process_events(events)
    except Exception:  # fall back to plain ordering when the VLM tools are not importable
        events = sorted(events, key=lambda e: (e["on_frame"], e["off_frame"]))
    if len(events) != num_slots:
        print(f"warning: {path} gives {len(events)} events for {num_slots} slots; names may not line up", flush=True)
    names = [str(event["object_description"]).strip() for event in events]
    return [names[i] if i < len(names) and names[i] else f"object {i + 1}" for i in range(num_slots)]


def export_parts(splats: Splats, logits: np.ndarray, out: Path, *, min_confidence: float = 0.6,
                 up=(0.0, -1.0, 0.0), names: list[str] | None = None, name: str = "scene",
                 soft_threshold: float = 0.0) -> dict:
    """Write the export directory; returns the scene.json dictionary.

    ``soft_threshold`` > 0 turns on the soft membership of the module docstring (0 = hard argmax sets).
    """
    split = split_by_state(logits, min_confidence)
    out.mkdir(parents=True, exist_ok=True)
    soft = soft_split(softmax(logits), min(soft_threshold, min_confidence)) if soft_threshold > 0 else None
    if soft is None:
        write_ply(out / "static.ply", splats.subset(split.static))
        print(f"static: {int(split.static.sum())} gaussians", flush=True)
    else:
        rows = soft.static > 0
        write_ply(out / "static.ply", scale_opacity(splats.subset(rows), soft.static[rows]))
        print(f"static: {int(rows.sum())} gaussians ({int(split.static.sum())} by argmax)", flush=True)

    parts = []
    for slot, (base_mask, switched_mask) in split.slots.items():
        sets = {}
        for state, mask in (("base", base_mask), ("switched", switched_mask)):
            idx = np.flatnonzero(mask)
            keep = outlier_mask(splats.positions[idx])
            sets[state] = idx[keep]
            print(f"slot{slot:02d} {state}: {len(idx)} gaussians, {int((~keep).sum())} outliers removed", flush=True)
        if len(sets["base"]) == 0 or len(sets["switched"]) == 0:
            print(f"warning: slot{slot:02d} has an empty base or switched set; skipped", flush=True)
            continue
        slot_id = f"slot{slot:02d}"
        if soft is None:
            for state, idx in sets.items():
                write_ply(out / f"{slot_id}_{state}.ply", splats.subset(idx))
            membership_path(out, slot_id).unlink(missing_ok=True)
        else:
            fit, mult = {}, {}
            for state, table in (("base", soft.base[slot]), ("switched", soft.switched[slot])):
                rows = np.flatnonzero(table > 0)
                write_ply(out / f"{slot_id}_{state}.ply", scale_opacity(splats.subset(rows), table[rows]))
                fit[state] = np.isin(rows, sets[state])
                mult[state] = table[rows].astype(np.float32)
                print(f"{slot_id} {state}: {len(rows)} soft candidates (confident {int(fit[state].sum())})", flush=True)
            np.savez(membership_path(out, slot_id), base_fit=fit["base"], switched_fit=fit["switched"],
                     base_mult=mult["base"], switched_mult=mult["switched"])
        parts.append({
            "id": slot_id,
            "name": names[slot - 1] if names and slot <= len(names) else f"object {slot}",
            "files": {"base": f"{slot_id}_base.ply", "switched": f"{slot_id}_switched.ply", "part": None,
                      "reveal": None, "vanish": None},
            "joint": None,
        })
    scene = {"version": 1, "name": name, "up": [float(x) for x in up], "static": "static.ply", "parts": parts}
    (out / "scene.json").write_text(json.dumps(scene, indent=2) + "\n", encoding="utf-8")
    return scene


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="runs/<run>/ckpt_last.pt")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--min-confidence", type=float, default=0.6,
                        help="drop slot gaussians whose top state probability is lower (default 0.6)")
    parser.add_argument("--soft-threshold", type=float, default=0.1,
                        help="soft membership: emit a gaussian in every state file where its probability is at least "
                             "this, opacity scaled by the mass (default 0.1; 0 = hard argmax sets)")
    parser.add_argument("--up", type=float, nargs=3, default=[0.0, -1.0, 0.0], metavar=("X", "Y", "Z"),
                        help="world up vector of the exported coordinates (COLMAP default 0 -1 0)")
    parser.add_argument("--names-from", type=Path, help="VLM raw_observation.json providing the part names")
    parser.add_argument("--name", help="scene name (default: the checkpoint's run folder)")
    args = parser.parse_args(argv)

    splats, logits = load_checkpoint(args.checkpoint)
    num_slots = (logits.shape[1] - 1) // 2
    names = names_from_observation(args.names_from, num_slots) if args.names_from else None
    name = args.name or args.checkpoint.resolve().parent.name
    scene = export_parts(splats, logits, args.out, min_confidence=args.min_confidence, up=args.up,
                         names=names, name=name, soft_threshold=args.soft_threshold)
    print(f"wrote {args.out} with {len(scene['parts'])} parts", flush=True)


if __name__ == "__main__":
    main()
