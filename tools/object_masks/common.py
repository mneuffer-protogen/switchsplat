# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the object-mask pipeline."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

# Working size of labels, classifier and SAM 2 (square, in pixels).
RESOLUTION = 512

REPO_ROOT = Path(__file__).resolve().parents[2]

# Frame names follow the dataset loader (threedgrut/datasets): ``frame_<time>[_...]`` carries the
# time index, and novel-view / novel-combination test images are never training frames.
FRAME_TIME_PATTERN = re.compile(r"^frame_(\d+)(?:_.*)?$")
NOVEL_EVAL_PATTERN = re.compile(r"novel_(view|combo|combination)_([01]+)_\d+")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def frame_time_index(name: str) -> int | None:
    """Time index of an image file or stem named ``frame_<time>[_...]``, or None."""
    match = FRAME_TIME_PATTERN.fullmatch(Path(name).stem)
    return int(match.group(1)) if match else None


def slot_dir(scene_dir: Path, slot: int) -> Path:
    """Folder of one object slot (1-based), e.g. ``scene_dir/slot03``."""
    return Path(scene_dir) / f"slot{slot:02d}"


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def load_valid_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) > 127


def to_u8(rgb: np.ndarray) -> np.ndarray:
    return np.round(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def _gaussian_kernel_2d(sigma: float, device: torch.device) -> torch.Tensor:
    radius = int(4 * sigma + 0.5)
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    k1 = torch.exp(-0.5 * (x / sigma) ** 2)
    k1 = k1 / k1.sum()
    return (k1[:, None] * k1[None, :])[None, None]


def raw_ssim_error(render: np.ndarray, gt: np.ndarray, device: torch.device) -> np.ndarray:
    """Per-pixel grayscale 1-SSIM with a Gaussian window of 0.5% of the short side."""
    sigma = max(0.005 * min(render.shape[:2]), 1.0e-6)
    kernel = _gaussian_kernel_2d(sigma, device)
    weights = torch.tensor([0.2989, 0.5870, 0.1140], device=device)
    pred_g = (torch.from_numpy(render).to(device) * weights).sum(dim=-1)[None, None]
    gt_g = (torch.from_numpy(gt).to(device) * weights).sum(dim=-1)[None, None]
    pad = kernel.shape[-1] // 2
    c1, c2 = 0.01**2, 0.03**2

    def conv_reflect(x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), kernel)

    with torch.no_grad():
        mu1 = conv_reflect(pred_g)
        mu2 = conv_reflect(gt_g)
        mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
        s1 = conv_reflect(pred_g * pred_g) - mu1_sq
        s2 = conv_reflect(gt_g * gt_g) - mu2_sq
        s12 = conv_reflect(pred_g * gt_g) - mu1_mu2
        ssim = ((2 * mu1_mu2 + c1) * (2 * s12 + c2)) / ((mu1_sq + mu2_sq + c1) * (s1 + s2 + c2) + 1e-12)
        return (1.0 - ssim.clamp(-1.0, 1.0))[0, 0].cpu().numpy()


def annotate(panel: np.ndarray, lines: list[str]) -> np.ndarray:
    """Draw white-on-black text lines onto a uint8 RGB panel."""
    image = Image.fromarray(panel)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    y = 6
    for line in lines:
        draw.rectangle((4, y - 1, 4 + 7 * len(line) + 8, y + 12), fill=(0, 0, 0))
        draw.text((8, y), line, fill=(255, 255, 255), font=font)
        y += 14
    return np.asarray(image)


def encode_frames(frames_dir: Path, video_path: Path, fps: int) -> bool:
    """Encode ``frames_dir/*.jpg`` (sorted by name) into an H.264 MP4.

    Returns False, keeping only the frames, when ffmpeg is not installed.
    """
    if shutil.which("ffmpeg") is None:
        print(f"ffmpeg not found; the preview frames stay in {frames_dir} without a video", flush=True)
        return False
    for codec in ("libx264", "mpeg4"):
        command = ["ffmpeg", "-y", "-v", "error", "-framerate", str(fps), "-pattern_type", "glob"]
        command += ["-i", str(frames_dir / "*.jpg"), "-c:v", codec, "-pix_fmt", "yuv420p"]
        if codec == "libx264":
            command += ["-crf", "18"]
        if subprocess.run(command + [str(video_path)]).returncode == 0:
            return True
    raise RuntimeError(f"ffmpeg failed to encode {video_path}")


def savez_compressed_atomic(path: Path, **arrays) -> None:
    """``np.savez_compressed`` through a temporary file, so an interrupted run leaves no partial file."""
    path = Path(path)
    tmp_path = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp_path, **arrays)
    tmp_path.replace(path)


def write_text_atomic(path: Path, text: str) -> None:
    path = Path(path)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)
