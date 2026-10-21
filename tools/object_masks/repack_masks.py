# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repack settled masks into the compact set the fine-tuning loss reads, and verify the copy.

The mask loss (threedgrut/utils/slot_mask_supervision.py, as configured in
configs/apps/colmap_switchsplat_fisheye_finetune.yaml) reads, per slot,
``pred_masks_sam`` (> 0.5) and ``time_indices`` from slotNN/mask/masks.npz, and ``valid`` (> 0.5)
and ``time_indices`` from slotNN/labels/labels.npz. The copy keeps exactly these arrays, with the
binary masks stored as uint8 0/1 instead of float32. Source masks must hold only 0 and 1, so
nothing is lost. Files are written with fixed zip timestamps at deflate level 9, so the same masks
always give the same bytes.

After writing, the loss's own loading code runs on the source and on the copy, and everything it
builds (mask stack, label-valid stack, frame and row lookups) must be identical.

Usage:
    python -m tools.object_masks.repack_masks --out OUT --scene NAME=SETTLED_DIR [--scene ...] [--note TEXT]

Writes OUT/NAME/slotNN/{mask/masks.npz,labels/labels.npz} and OUT/MANIFEST.json.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .common import write_text_atomic

MASK_KEY = "pred_masks_sam"
MASK_FILE = "mask/masks.npz"
LABELS_FILE = "labels/labels.npz"
# Mask-loading settings of loss.slot_mask in configs/apps/colmap_switchsplat_fisheye_finetune.yaml.
READER_SETTINGS = {
    "mask_key": MASK_KEY,
    "camera_id": "camera1",
    "other_cameras_empty": False,
    "soft_gating": False,
    "use_label_valid": True,
    "mask_dilate_pixels": 6,
}
FORMAT = (
    "slotNN/mask/masks.npz: pred_masks_sam [N,512,512] uint8 (1 = object region), time_indices [N] int32; "
    "slotNN/labels/labels.npz: valid [N,512,512] uint8 (1 = pixel may be supervised), time_indices [N] int32. "
    "Masks are at 512x512 for the front lens (camera1); time_indices are the frame numbers of frame_<time>.png."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """A deflate-9 .npz with fixed timestamps (np.load reads it like any .npz)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in arrays.items():
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.ascontiguousarray(value), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o644 << 16
            archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    tmp_path.replace(path)


def binary_uint8(values: np.ndarray, what: str) -> np.ndarray:
    if not ((values == 0) | (values == 1)).all():
        raise SystemExit(f"{what} holds values other than 0 and 1; a uint8 copy would not be lossless")
    return values.astype(np.uint8)


def slot_names(run_dir: Path) -> list[str]:
    names = sorted(path.name for path in run_dir.glob("slot[0-9][0-9]") if path.is_dir())
    if not names or names != [f"slot{slot:02d}" for slot in range(1, len(names) + 1)]:
        raise SystemExit(f"{run_dir}: expected folders slot01..slotNN, found {names}")
    return names


def repack_slot(src: Path, dst: Path) -> dict:
    """Write one slot's compact copy; return its manifest entry."""
    with np.load(src / MASK_FILE) as masks:
        mask_times = masks["time_indices"]
        sam = binary_uint8(masks[MASK_KEY], f"{src / MASK_FILE}:{MASK_KEY}")
    with np.load(src / LABELS_FILE) as labels:
        label_times = labels["time_indices"]
        valid = binary_uint8(labels["valid"], f"{src / LABELS_FILE}:valid")
    if not np.array_equal(mask_times, label_times):
        raise SystemExit(f"{src}: masks and labels cover different frames")
    save_npz(dst / MASK_FILE, {MASK_KEY: sam, "time_indices": mask_times})
    save_npz(dst / LABELS_FILE, {"valid": valid, "time_indices": label_times})
    return {
        "n_frames": int(len(mask_times)),
        "frames_kept": [int(t) for t in mask_times],
        "files": {
            name: {"sha256": sha256(dst / name), "bytes": (dst / name).stat().st_size}
            for name in (MASK_FILE, LABELS_FILE)
        },
    }


def load_with_reader(run_dir: Path, num_slots: int):
    """Run the fine-tuning loss's mask loading on a mask folder and return the loss object."""
    from threedgrut.utils.slot_mask_supervision import SlotMaskSupervision

    settings = SimpleNamespace(mask_run=str(run_dir), **READER_SETTINGS)
    conf = SimpleNamespace(loss=SimpleNamespace(slot_mask=settings), render=SimpleNamespace(method="gsplat_3dgut"))
    num_states = 1 + 2 * num_slots
    model = SimpleNamespace(
        switch_splat_controller=SimpleNamespace(num_slots=num_slots, num_states=num_states),
        switch_logits=torch.zeros((1, num_states)),
    )
    dataset = SimpleNamespace(get_camera_names=lambda: [READER_SETTINGS["camera_id"]])
    return SlotMaskSupervision(conf, model, dataset)


def same_for_reader(a, b) -> bool:
    """Whether two loaded mask sets give the loss exactly the same data."""
    return (
        a.masks.dtype == b.masks.dtype
        and a.masks.shape == b.masks.shape
        and torch.equal(a.masks, b.masks)
        and a.label_valids.dtype == b.label_valids.dtype
        and a.label_valids.shape == b.label_valids.shape
        and torch.equal(a.label_valids, b.label_valids)
        and a.slots_of_time == b.slots_of_time
        and a.row_of_time == b.row_of_time
    )


def verify(src: Path, dst: Path, num_slots: int) -> dict:
    source, copy = load_with_reader(src, num_slots), load_with_reader(dst, num_slots)
    if not same_for_reader(source, copy):
        raise SystemExit(f"verification failed: {dst} does not load like {src}")
    from threedgrut.utils import slot_mask_supervision

    module = Path(inspect.getsourcefile(slot_mask_supervision))
    return {
        "reader": "threedgrut/utils/slot_mask_supervision.py",
        "reader_sha256": sha256(module),
        "settings": READER_SETTINGS,
        "identical": True,
        "mask_rows": int(source.masks.shape[0]),
        "masked_times": len(source.slots_of_time),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="output folder (one subfolder per scene)")
    parser.add_argument(
        "--scene", action="append", required=True, metavar="NAME=DIR", help="output name and settled mask folder"
    )
    parser.add_argument("--note", help="provenance note stored in MANIFEST.json")
    parser.add_argument("--no-verify", action="store_true", help="skip the check with the fine-tuning loss's loader")
    args = parser.parse_args(argv)

    scenes = []
    for spec in args.scene:
        name, sep, src = spec.partition("=")
        if not sep or not name or not src:
            raise SystemExit(f"--scene expects NAME=DIR, got {spec!r}")
        if (args.out / name).exists():
            raise SystemExit(f"{args.out / name} already exists; use a new --out")
        scenes.append((name, Path(src)))

    manifest = {"format": FORMAT, "note": args.note, "scenes": {}}
    total_bytes = 0
    for name, src in scenes:
        names = slot_names(src)
        record_path = src / "DROP_TRANSITIONS.json"
        record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.is_file() else None
        slots = {}
        for slot_name in names:
            slots[slot_name] = repack_slot(src / slot_name, args.out / name / slot_name)
            if record is not None:
                slots[slot_name]["drop_transitions"] = record.get("slots", {}).get(slot_name)
            total_bytes += sum(item["bytes"] for item in slots[slot_name]["files"].values())
        entry = {
            "source_scene": src.name,
            "source_run": None if record is None else record.get("source"),
            # The source's DROP_TRANSITIONS.json without its per-slot part, which each slot entry holds.
            "drop_transitions": None if record is None else {k: v for k, v in record.items() if k != "slots"},
            "num_slots": len(names),
            "verification": None if args.no_verify else verify(src, args.out / name, len(names)),
            "slots": slots,
        }
        manifest["scenes"][name] = entry
        frames = sum(slot["n_frames"] for slot in slots.values())
        print(f"{name}: {len(names)} slots, {frames} frames, verified={not args.no_verify}", flush=True)

    manifest["total_npz_bytes"] = total_bytes
    args.out.mkdir(parents=True, exist_ok=True)
    write_text_atomic(args.out / "MANIFEST.json", json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {args.out / 'MANIFEST.json'}: {total_bytes / 1e6:.1f} MB of masks", flush=True)


if __name__ == "__main__":
    main()
