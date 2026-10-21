# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fit the per-slot mask classifier on DINOv3+XY features, then refine with SAM 2.1 video propagation.

Classifier and loss follow RoMo/SpotLess: a 1026->8->1 sigmoid MLP trained with the bounded loss
that pushes red pixels up, blue pixels down, and ignores the rest. Two changes keep DINOv3 training
alive (without them ReLU units die and the mask collapses to all zeros): per-channel z-scored
features and Adam lr 5e-3.

SAM 2.1 (Hiera-L) then propagates the classifier masks of the odd frames to the even frames and
those of the even frames to the odd frames, forward in chunks of 80 frames. SAM 2's small-hole
filling is switched off, so the result does not depend on whether its optional CUDA extension is
built.

Writes OUT_DIR/masks.npz (pred_masks: classifier probabilities, pred_masks_sam: final masks) and
classifier.pt. Needs a CUDA GPU.
"""

from __future__ import annotations

import argparse
import random
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from .common import frame_time_index, savez_compressed_atomic

EPOCHS = 25
LEARNING_RATE = 5e-3
SAM_CHUNK = 80
SAM_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


class Classifier(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(num_features, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid())

    def get_regularizer(self) -> torch.Tensor:
        # Uses .data as in the released RoMo code, so it adds no gradient.
        return torch.max(abs(self.mlp[0].weight.data)) * torch.max(abs(self.mlp[2].weight.data))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def sls_loss(pred: torch.Tensor, minimum: torch.Tensor, maximum: torch.Tensor, delta: float = 0.01) -> torch.Tensor:
    over = F.relu(pred - minimum).square()
    under = F.relu(maximum - pred).square()
    return torch.mean(over / (delta + over) + under / (delta + under))


def load_features(cache_dir: Path, image_names: list[str]) -> list[torch.Tensor]:
    features = [
        torch.load(cache_dir / f"{name}.pt", map_location="cpu", weights_only=True).float() for name in image_names
    ]
    stack = torch.stack(features)
    mean = stack.mean(dim=(0, 2, 3))[:, None, None]
    std = stack.std(dim=(0, 2, 3))[:, None, None]
    return [((f - mean) / (std + 1e-4)).clamp(-6.0, 6.0) for f in features]


def pixel_features(feature: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    size = coords.shape[-2:]
    upsampled = F.interpolate(feature[None].to(coords.device), size=size, mode="bilinear", align_corners=False)[0]
    channels = torch.cat((upsampled, coords), dim=0)
    return channels.reshape(channels.shape[0], -1).transpose(0, 1).contiguous()


def train_classifier(features, masks_high, masks_low, trusted) -> tuple[torch.Tensor, Classifier]:
    """Train on the trusted frames; runs on the device and at the resolution of the label masks."""
    device, resolution = masks_high.device, masks_high.shape[-1]
    classifier = Classifier(features[0].shape[0] + 2).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LEARNING_RATE)
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, resolution, device=device),
        torch.linspace(-1.0, 1.0, resolution, device=device),
        indexing="ij",
    )
    coords = torch.stack((x, y), dim=0)
    order = list(range(len(features)))
    random.shuffle(order)
    trusted = set(trusted)
    preds = torch.zeros((len(features), resolution, resolution), device=device)
    classifier.train()
    for epoch in range(EPOCHS):
        total = 0.0
        for j in order:
            pred = classifier(pixel_features(features[j], coords)).reshape(resolution, resolution)
            preds[j] = pred.detach()
            if j in trusted:
                loss = sls_loss(pred.flatten(), 1.0 - masks_low[j].flatten(), masks_high[j].flatten())
                (loss + 100 * classifier.get_regularizer()).backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                total += loss.item() / len(features)
        if epoch % 6 == 0 or epoch == EPOCHS - 1:
            print(f"epoch {epoch} loss={total:.6f} frac>=0.5={float((preds >= 0.5).float().mean()):.4f}", flush=True)
    return preds, classifier


def refine_with_sam(frames: np.ndarray, prompts: torch.Tensor, output: torch.Tensor, checkpoint: Path) -> None:
    """Two-pass even/odd SAM 2 propagation.

    Odd frames' classifier masks prompt SAM and each even frame reads its own propagated mask; then
    the roles swap. No frame keeps its own classifier mask.
    """
    from sam2.build_sam import build_sam2_video_predictor

    def propagate(predictor, state, masks, parity):
        predictor.reset_state(state)
        prompted = False
        for index, mask in enumerate(masks):
            if index % 2 == parity:
                predictor.add_new_mask(state, frame_idx=index, obj_id=1, mask=mask)
                prompted = True
        if not prompted:  # singleton final chunk: keep the classifier mask
            return {}
        return {
            index: (logits[0] > 0.0).float()
            for index, _, logits in predictor.propagate_in_video(state, start_frame_idx=0)
        }

    for start in range(0, len(frames), SAM_CHUNK):
        end = min(start + SAM_CHUNK, len(frames))
        predictor = build_sam2_video_predictor(SAM_CONFIG, str(checkpoint), device="cuda")
        # Small-hole filling off: it needs SAM 2's optional CUDA extension, so the masks would depend on the build.
        if not hasattr(predictor, "fill_hole_area"):
            raise RuntimeError("unsupported sam2 version: the video predictor has no fill_hole_area")
        predictor.fill_hole_area = 0
        temp_dir = tempfile.mkdtemp()
        try:
            for index in range(start, end):
                cv2.imwrite(f"{temp_dir}/{index - start:05d}.jpg", cv2.cvtColor(frames[index], cv2.COLOR_RGB2BGR))
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                state = predictor.init_state(temp_dir)
                chunk = prompts[start:end]
                from_odd = propagate(predictor, state, chunk, parity=1)
                from_even = propagate(predictor, state, chunk, parity=0)
                for index in range(end - start):
                    segments = from_odd if index % 2 == 0 else from_even
                    if index in segments:
                        output[start + index] = segments[index][0]
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            del predictor
            torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels-dir", type=Path, required=True, help="SCENE_DIR/slotNN/labels")
    parser.add_argument("--feature-cache", type=Path, required=True, help="DINOv3 cache of the photos")
    parser.add_argument("--sam-checkpoint", type=Path, required=True, help="sam2.1_hiera_large.pt")
    parser.add_argument("--out-dir", type=Path, required=True, help="SCENE_DIR/slotNN/mask")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    labels = np.load(args.labels_dir / "labels.npz")
    time_indices = [int(t) for t in labels["time_indices"]]
    image_names = [str(name) for name in labels["image_names"]]
    if any(frame_time_index(name) != t for name, t in zip(image_names, time_indices)):
        raise SystemExit("label time indices do not match image names")
    masks_high = torch.from_numpy(labels["masks_high"]).cuda()
    masks_low = torch.from_numpy(labels["masks_low"]).cuda()
    # Pixels outside the camera's valid region (fisheye border, camera-holder masks) carry no
    # label, so the classifier is free to light them up; a mask is only meaningful where the image is.
    valid = torch.from_numpy(labels["valid"]).float().cuda()
    trusted = [int(i) for i in labels["trusted"]]
    if not trusted:
        raise SystemExit("no frame has any red or blue label")
    print(f"labels frames={len(time_indices)} pos={int(masks_high.sum())} neg={int(masks_low.sum())}", flush=True)

    preds, classifier = train_classifier(load_features(args.feature_cache, image_names), masks_high, masks_low, trusted)
    preds = preds * valid

    final = (preds > 0.5).float()
    if final.any():
        frames = np.stack(
            [
                np.asarray(Image.open(args.labels_dir / "gt_512" / f"time_{t:04d}.png").convert("RGB"))
                for t in time_indices
            ]
        )
        print("refining with SAM 2", flush=True)
        refine_with_sam(frames, preds > 0.5, final, args.sam_checkpoint)
        final = final * valid
    else:
        print("classifier predicted no pixels above 0.5; skipping SAM", flush=True)

    torch.save(classifier.state_dict(), args.out_dir / "classifier.pt")
    # masks.npz is written last: it marks the stage as done.
    savez_compressed_atomic(
        args.out_dir / "masks.npz",
        pred_masks=preds.cpu().numpy(),
        pred_masks_sam=final.cpu().numpy(),
        masks_high=labels["masks_high"],
        masks_low=labels["masks_low"],
        time_indices=np.asarray(time_indices, dtype=np.int32),
        image_names=np.asarray(image_names),
    )
    mlp_fraction, sam_fraction = float((preds >= 0.5).float().mean()), float(final.mean())
    print(f"done mlp_frac={mlp_fraction:.4f} sam_frac={sam_fraction:.4f} out={args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
