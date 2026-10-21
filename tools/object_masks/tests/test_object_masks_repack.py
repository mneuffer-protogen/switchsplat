# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The compact mask copy must load in the fine-tuning loss exactly like the full settled set."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.object_masks import repack_masks


def _settled_set(root: Path, mask_run) -> Path:
    # Frame 17 belongs to both slots, as where two objects are switched at once.
    mask_run(root, {1: [5, 9, 17], 2: [17, 21]})
    record = {
        "source": "runs/demo/scene",
        "transitions": "data/demo/transition_frames.txt",
        "slots": {
            "slot01": {"frames": 4, "kept": 3, "dropped": [12]},
            "slot02": {"frames": 2, "kept": 2, "dropped": []},
        },
    }
    (root / "DROP_TRANSITIONS.json").write_text(json.dumps(record))
    return root


def test_repack_is_lossless_for_the_fine_tune_reader(tmp_path: Path, mask_run, reader):
    src = _settled_set(tmp_path / "settled", mask_run)
    repack_masks.main(["--out", str(tmp_path / "out"), "--scene", f"demo={src}", "--note", "test set"])
    out = tmp_path / "out" / "demo"

    full, compact = reader(src, 2), reader(out, 2)
    assert full.masks.dtype == compact.masks.dtype == torch.uint8
    assert torch.equal(full.masks, compact.masks)
    assert torch.equal(full.label_valids, compact.label_valids)
    assert full.slots_of_time == compact.slots_of_time == {5: [1], 9: [1], 17: [1, 2], 21: [2]}
    assert full.row_of_time == compact.row_of_time

    # The copy holds only what the loss reads, with the binary masks as uint8.
    for slot in ("slot01", "slot02"):
        with np.load(src / slot / "mask" / "masks.npz") as a, np.load(out / slot / "mask" / "masks.npz") as b:
            assert sorted(b.files) == ["pred_masks_sam", "time_indices"]
            assert b["pred_masks_sam"].dtype == np.uint8
            assert np.array_equal(b["pred_masks_sam"], a["pred_masks_sam"])
            assert np.array_equal(b["time_indices"], a["time_indices"])
        with np.load(src / slot / "labels" / "labels.npz") as a, np.load(out / slot / "labels" / "labels.npz") as b:
            assert sorted(b.files) == ["time_indices", "valid"]
            assert np.array_equal(b["valid"], a["valid"]) and b["valid"].dtype == np.uint8

    manifest = json.loads((tmp_path / "out" / "MANIFEST.json").read_text())
    assert manifest["note"] == "test set"
    entry = manifest["scenes"]["demo"]
    assert entry["verification"]["identical"] and entry["verification"]["mask_rows"] == 5
    assert entry["source_run"] == "runs/demo/scene" and entry["drop_transitions"]["transitions"].endswith(".txt")
    assert entry["slots"]["slot01"]["frames_kept"] == [5, 9, 17]
    assert entry["slots"]["slot01"]["drop_transitions"]["dropped"] == [12]
    for slot_name, slot in entry["slots"].items():
        for name, info in slot["files"].items():
            data = (out / slot_name / name).read_bytes()
            assert info["sha256"] == hashlib.sha256(data).hexdigest() and info["bytes"] == len(data)
    assert manifest["total_npz_bytes"] == sum(path.stat().st_size for path in out.rglob("*.npz"))


def test_repack_gives_the_same_bytes_every_time(tmp_path: Path, mask_run):
    src = _settled_set(tmp_path / "settled", mask_run)
    for run in ("a", "b"):
        repack_masks.main(["--out", str(tmp_path / run), "--scene", f"demo={src}", "--no-verify"])
    files = sorted(path.relative_to(tmp_path / "a") for path in (tmp_path / "a").rglob("*.npz"))
    assert len(files) == 4
    for name in files:
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_repack_refuses_masks_that_are_not_binary(tmp_path: Path, mask_run):
    src = _settled_set(tmp_path / "settled", mask_run)
    with np.load(src / "slot02" / "mask" / "masks.npz") as bundle:
        arrays = dict(bundle)
    arrays["pred_masks_sam"][0, 0, 0] = 0.75
    np.savez_compressed(src / "slot02" / "mask" / "masks.npz", **arrays)
    with pytest.raises(SystemExit):
        repack_masks.main(["--out", str(tmp_path / "out"), "--scene", f"demo={src}"])
