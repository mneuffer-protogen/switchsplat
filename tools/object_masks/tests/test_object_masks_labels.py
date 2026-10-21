# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU smoke tests of the pseudo-labels and of the mask classifier on small synthetic data."""

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from tools.object_masks import make_slot_labels
from tools.object_masks.common import to_u8
from tools.object_masks.train_slot_mask import load_features, train_classifier

LABELLED = [10, 11, 12, 13, 14]
CONTROLS = [30, 31, 32, 33, 34]
OTHER_ONLY = 18  # inside the other slot's window only


def _scene_with_pairs(root: Path) -> Path:
    """Slot 1 of a two-object scene: an object square appears in the photos of its window only."""
    rng = np.random.default_rng(0)
    size = 64
    yy, xx = np.mgrid[0:size, 0:size]
    texture = np.repeat((0.4 + 0.15 * np.sin(xx / 2.0) * np.cos(yy / 3.0))[..., None], 3, axis=2)
    root.mkdir(parents=True)
    (root / "windows.json").write_text(json.dumps({"source": "file", "windows": [[10, 14], [12, 20]]}))
    pairs = root / "slot01" / "pairs"
    rows = []
    for sub in ("renders", "gt", "valid_masks"):
        (pairs / sub).mkdir(parents=True)
    for time_index in LABELLED + [OTHER_ONLY] + CONTROLS:
        render = texture + rng.normal(0.0, 0.002, texture.shape)
        gt = render.copy()
        if time_index in LABELLED:
            gt[16:32, 16:32] = 0.95  # the object in its switched state: only in the photo
        elif time_index == OTHER_ONLY:
            gt = 1.0 - gt  # a large change by another object must not raise the thresholds
        else:
            gt = gt + rng.normal(0.0, 0.002, texture.shape)
        valid = np.full((size, size), 255, dtype=np.uint8)
        valid[:, :2] = 0  # e.g. the camera holder
        stem = f"time_{time_index:04d}.png"
        Image.fromarray(to_u8(render)).save(pairs / "renders" / stem)
        Image.fromarray(to_u8(gt)).save(pairs / "gt" / stem)
        Image.fromarray(valid).save(pairs / "valid_masks" / stem)
        rows.append(
            {
                "time_index": time_index,
                "image_name": f"frame_{time_index:04d}",
                "render_path": f"renders/{stem}",
                "gt_path": f"gt/{stem}",
                "valid_mask_path": f"valid_masks/{stem}",
                "mean_l1": "0",
            }
        )
    with (pairs / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return root


def test_labels_mark_the_changed_region(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)  # CPU even on GPU machines
    scene = _scene_with_pairs(tmp_path / "scene")
    make_slot_labels.main(["--scene-dir", str(scene), "--slot", "1", "--no-previews"])

    labels_dir = scene / "slot01" / "labels"
    meta = json.loads((labels_dir / "labels_meta.json").read_text())
    assert meta["n_control_frames"] == len(CONTROLS) and meta["window"] == [10.0, 14.0]
    with np.load(labels_dir / "labels.npz") as labels:
        assert labels["time_indices"].tolist() == LABELLED
        assert labels["image_names"].tolist() == [f"frame_{t:04d}" for t in LABELLED]
        high, low, valid = labels["masks_high"] > 0, labels["masks_low"] > 0, labels["valid"] > 0
        assert labels["trusted"].tolist() == list(range(len(LABELLED)))

    square = np.zeros((512, 512), dtype=bool)
    square[128:256, 128:256] = True  # the 64 px square at the 512 px working size
    near_square = np.zeros_like(square)
    near_square[112:272, 112:272] = True
    assert high[:, square].mean() > 0.9  # the object region is labelled object ...
    assert high[:, ~near_square].sum() == 0  # ... and nothing away from it
    assert low[:, ~near_square].mean() > 0.4 and low[:, square].sum() == 0
    # The invalid columns (0-15 at 512 px) plus the 8 px erosion carry no label.
    assert not valid[:, :, :24].any() and valid[:, 8:504, 24:504].all()
    assert not (high | low)[:, :, :24].any()
    assert sorted(path.name for path in (labels_dir / "gt_512").iterdir()) == [f"time_{t:04d}.png" for t in LABELLED]
    cached = sorted(path.name for path in (scene / "slot01" / "residual_cache").iterdir())
    assert cached == [f"time_{t:04d}.npz" for t in LABELLED + CONTROLS]


def test_classifier_learns_the_labelled_region(tmp_path: Path):
    random.seed(0)
    torch.manual_seed(0)
    names = [f"frame_{t:04d}" for t in range(3)]
    for name in names:
        feature = 0.1 * torch.randn(16, 4, 4)
        feature[0, :2, :2] += 3.0  # one feature channel responds to the object
        torch.save(feature.half(), tmp_path / f"{name}.pt")
    features = load_features(tmp_path, names)
    high = torch.zeros(3, 16, 16)
    high[:, :8, :8] = 1.0
    low = torch.zeros(3, 16, 16)
    low[:, 9:, :] = 1.0
    low[:, :, 9:] = 1.0
    preds, classifier = train_classifier(features, high, low, trusted=[0, 1, 2])
    assert preds.shape == (3, 16, 16) and classifier.mlp[0].in_features == 18
    assert (preds[:, :7, :7] > 0.5).float().mean() > 0.9
    assert (preds[:, 10:, 10:] < 0.5).float().mean() > 0.9
