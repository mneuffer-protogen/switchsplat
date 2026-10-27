# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export a trained SwitchSplat checkpoint as static + per-slot base/switched PLY sets.

Every gaussian is assigned to its most probable switch state: 0 = static background, 2k-1 = slot k
base state, 2k = slot k switched state (the argmax of the softmax of ``switch_logits``). The
gaussians are written exactly like ``threedgrut/export/ply_exporter.py`` writes them (raw,
pre-activation values: logit opacity, log scale, unnormalised quaternion, every stored SH
coefficient). Slot sets drop gaussians whose top probability is below --min-confidence and
statistical outliers; the static set keeps every state-0 gaussian.

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
                 up=(0.0, -1.0, 0.0), names: list[str] | None = None, name: str = "scene") -> dict:
    """Write the export directory; returns the scene.json dictionary."""
    split = split_by_state(logits, min_confidence)
    out.mkdir(parents=True, exist_ok=True)
    write_ply(out / "static.ply", splats.subset(split.static))
    print(f"static: {int(split.static.sum())} gaussians", flush=True)

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
        for state, idx in sets.items():
            write_ply(out / f"{slot_id}_{state}.ply", splats.subset(idx))
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
                         names=names, name=name)
    print(f"wrote {args.out} with {len(scene['parts'])} parts", flush=True)


if __name__ == "__main__":
    main()
