# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cache DINOv3 ViT-L/16 patch features ([1024, 32, 32], fp16) of the photos at 512x512.

By default every image of --image-dir is cached; with --labels only the frames named in those
labels.npz files (the frames the slot classifiers train on). Images already in the cache are
skipped. The DINOv3 weights are gated on Hugging Face: accept the licence on the model page and log
in (``hf auth login``) or pass a local copy with --model. Needs a CUDA GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .common import IMAGE_SUFFIXES, write_text_atomic

DEFAULT_MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def load_image(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
    tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0
    return (tensor - MEAN) / STD


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--image-dir", type=Path, required=True, help="photos of the masked camera, e.g. images_2/camera1"
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id or local model folder")
    parser.add_argument("--labels", type=Path, nargs="+", help="cache only the frames named in these labels.npz")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)

    paths = sorted(p for p in args.image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if args.labels:
        wanted = set()
        for labels_path in args.labels:
            with np.load(labels_path) as labels:
                wanted.update(str(name) for name in labels["image_names"])
        absent = sorted(wanted - {p.stem for p in paths})
        if absent:
            raise SystemExit(f"{len(absent)} labelled frames have no image in {args.image_dir}: {absent[:10]}")
        paths = [p for p in paths if p.stem in wanted]
    if not paths:
        raise SystemExit(f"no images under {args.image_dir}")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    missing = [p for p in paths if not (args.cache_dir / f"{p.stem}.pt").is_file()]

    if missing:
        from transformers import DINOv3ViTModel

        model = DINOv3ViTModel.from_pretrained(args.model).eval().cuda()
        num_prefix_tokens = 1 + int(model.config.num_register_tokens)  # CLS + register tokens
        with torch.inference_mode():
            for start in range(0, len(missing), args.batch_size):
                chunk = missing[start : start + args.batch_size]
                batch = torch.stack([load_image(p) for p in chunk]).cuda()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    hidden = model(pixel_values=batch).last_hidden_state
                patch = hidden[:, num_prefix_tokens:, :]
                side = int(patch.shape[1] ** 0.5)
                patch = patch.transpose(1, 2).reshape(len(chunk), patch.shape[-1], side, side)
                for path, feature in zip(chunk, patch):
                    tmp_path = args.cache_dir / f".{path.stem}.pt.tmp"
                    torch.save(feature.float().cpu().half(), tmp_path)
                    tmp_path.replace(args.cache_dir / f"{path.stem}.pt")
                done = min(start + args.batch_size, len(missing))
                if done % (20 * args.batch_size) == 0 or done == len(missing):
                    print(f"cached {done}/{len(missing)}", flush=True)

    metadata = {
        "image_dir": str(args.image_dir),
        "model": str(args.model),
        "input_resolution": [512, 512],
        "feature_shape": [1024, 32, 32],
        "dtype": "float16",
        "n_images": len(paths),
    }
    write_text_atomic(args.cache_dir / "metadata.json", json.dumps(metadata, indent=2) + "\n")
    print(f"DINOv3 cache ready: {args.cache_dir} ({len(paths)} images, {len(missing)} new)", flush=True)


if __name__ == "__main__":
    main()
