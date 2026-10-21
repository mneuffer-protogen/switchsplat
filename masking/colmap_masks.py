# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""COLMAP feature masks of a capture: white inside the lens circle and away from the person holding
the camera. Writes masks-colmap/<camera>/<frame>.png.png, the name COLMAP expects next to each image."""

import argparse
import os

import cv2
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scene", required=True, help="scene folder, e.g. data/<scene_name>")
parser.add_argument("--masks", default="masks/masks-20", help="person masks inside the scene folder")
parser.add_argument(
    "--valid",
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "fisheye_valid_mask.png"),
    help="lens circle (white = image)",
)
args = parser.parse_args()

border = cv2.imread(args.valid, cv2.IMREAD_GRAYSCALE) > 127
for camera in sorted(os.listdir(os.path.join(args.scene, "images"))):
    out_dir = os.path.join(args.scene, "masks-colmap", camera)
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(os.path.join(args.scene, "images", camera))):
        stem = os.path.splitext(name)[0]
        person_path = os.path.join(args.scene, args.masks, camera, f"{stem}_mask.png")
        valid = border.copy()
        if os.path.exists(person_path):
            person = cv2.imread(person_path, cv2.IMREAD_GRAYSCALE) > 127
            if person.shape != valid.shape:
                person = cv2.resize(person.astype(np.uint8), valid.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
            valid &= ~person
        cv2.imwrite(os.path.join(out_dir, f"{name}.png"), valid.astype(np.uint8) * 255)
