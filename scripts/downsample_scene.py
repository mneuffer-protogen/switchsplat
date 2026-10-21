# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Half-size frames and masks of a preprocessed capture, and its test split, in place.

  images/<camera>/frame_NNNN.png              ->  images_2/<camera>/frame_NNNN.png
  masks/masks-20/<camera>/frame_NNNN_mask.png ->  masks_2/masks-20_2/<camera>/frame_NNNN_mask.png
  test_frames.txt: every 8th frame (1, 9, 17, ...) of each camera, unless the scene has one already.

Frames are downsampled with Lanczos, masks by area averaging (training treats mask values above
127 as masked).
"""

import argparse
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REGULAR_FRAME = re.compile(r"^frame_(\d+)\.\w+$")


def downsample(src: Path, dst: Path, factor: int, is_mask: bool) -> None:
    image = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE if is_mask else cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"cannot read {src}")
    size = (image.shape[1] // factor, image.shape[0] // factor)
    if is_mask:
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    else:
        image = np.asarray(Image.fromarray(image[:, :, ::-1]).resize(size, Image.LANCZOS))[:, :, ::-1]
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), image)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene", type=Path, help="scene folder, e.g. data/<scene_name>")
    parser.add_argument("--factor", type=int, default=2)
    parser.add_argument("--mask-dilation", type=int, default=20, help="which masks/masks-<N> to downsample")
    parser.add_argument("--test-interval", type=int, default=8)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    scene, factor, dilation = args.scene, args.factor, args.mask_dilation
    jobs, frames = [], []
    for camera in sorted(p.name for p in (scene / "images").iterdir() if p.is_dir()):
        for image in sorted((scene / "images" / camera).glob("*.png")):
            frames.append(f"{camera}/{image.name}")
            jobs.append((image, scene / f"images_{factor}" / camera / image.name, False))
            mask = scene / "masks" / f"masks-{dilation}" / camera / f"{image.stem}_mask.png"
            if mask.is_file():
                jobs.append((mask, scene / f"masks_{factor}" / f"masks-{dilation}_{factor}" / camera / mask.name, True))
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(lambda job: downsample(job[0], job[1], factor, job[2]), jobs))
    print(f"{len(frames)} frames downsampled by {factor}")

    split = scene / "test_frames.txt"
    if not split.exists():
        held_out = []
        for frame in frames:
            match = REGULAR_FRAME.match(frame.split("/")[1])
            if match and (int(match.group(1)) - 1) % args.test_interval == 0:
                held_out.append(frame)
        split.write_text("".join(f"{frame}\n" for frame in held_out), encoding="utf-8")
        print(f"test_frames.txt: {len(held_out)} held-out frames")


if __name__ == "__main__":
    main()
