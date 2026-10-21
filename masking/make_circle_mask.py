# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trim the fisheye valid mask to a centred circle for gsplat 3DGUT training.

The result is the existing mask (which also removes the capture rig) AND a disk
of ``--radius`` full-resolution pixels around the principal point.  The source
mask is never modified.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radius", type=float, required=True, help="disk radius in full-resolution pixels")
    parser.add_argument("--source", default="masking/fisheye_valid_mask.png")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    source = np.asarray(Image.open(args.source).convert("L")) > 127
    height, width = source.shape
    # All SwitchSplat fisheye calibrations put the principal point at the image centre.
    yy, xx = np.mgrid[0:height, 0:width]
    disk = np.hypot(xx + 0.5 - width / 2, yy + 0.5 - height / 2) <= args.radius
    mask = source & disk
    Image.fromarray(mask.astype(np.uint8) * 255).save(out)
    print(f"{out}: radius={args.radius} valid_fraction={mask.mean():.4f} (source {source.mean():.4f})")


if __name__ == "__main__":
    main()
