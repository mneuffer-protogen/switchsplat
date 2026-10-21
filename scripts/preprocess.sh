#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Preprocess a 360° capture in place (masking and COLMAP as in FullCircle):
#   bash scripts/preprocess.sh data/<scene_name>
# Input:  images/{camera1,camera2}/frame_NNNN.png (fisheye frames), omni/images/frame_NNNN.png (panoramas)
# Output: masks/ (person holding the camera), sparse/0 (COLMAP), images_2/, masks_2/, test_frames.txt
# Rerunning skips finished steps. COLMAP_DATABASE moves the COLMAP database elsewhere, e.g. off a
# network drive (default: the scene folder).
set -euo pipefail

SCENE=$(realpath "${1:?usage: bash scripts/preprocess.sh data/<scene_name>}")
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PRE="$SCENE/pre_masking"
DB="${COLMAP_DATABASE:-$SCENE/database.db}"

mkdir -p "$PRE"

step() {  # step NAME COMMAND...: run COMMAND, then mark NAME done so that a rerun skips it
    local name=$1; shift
    if [ -f "$PRE/.done_$name" ]; then echo "  $name: done before, skipped"; return; fi
    "$@"
    touch "$PRE/.done_$name"
}

colmap_poses() {
    python masking/colmap_masks.py --scene "$SCENE"
    colmap feature_extractor --image_path "$SCENE/images" --database_path "$DB" \
        --ImageReader.mask_path "$SCENE/masks-colmap" --ImageReader.single_camera_per_folder 1 \
        --ImageReader.camera_model OPENCV_FISHEYE
    colmap exhaustive_matcher --database_path "$DB"
    rm -rf "$SCENE/sparse" && mkdir -p "$SCENE/sparse"
    colmap mapper --image_path "$SCENE/images" --database_path "$DB" --output_path "$SCENE/sparse"
}

echo "[1/3] masks of the person holding the camera"
step perspectives python masking/omni2perspective.py --input_path "$SCENE/omni/images" --output_path "$PRE/perspectives"
step person_masks python masking/mask_perspectives.py --input_path "$PRE/perspectives" --output_path "$PRE/perspective_masks"
step omni_masks python masking/perspective2omni.py --input_path "$PRE/perspective_masks" --output_path "$PRE/omni_masks_primary"
step synthetics python masking/omni2synthetic.py --scene "$SCENE"
step tracking env PYTHONPATH="thirdparty/sam-ui${PYTHONPATH:+:$PYTHONPATH}" python thirdparty/sam-ui/scripts/tracking_gui.py \
    --frames-path "$PRE/synthetics" --output-path "$PRE/synthetic_masks" --headless
step omni_final python masking/synthetic2omni.py --scene "$SCENE"
step fisheye_masks python masking/omni2fisheye.py --scene "$SCENE"
step dilate python masking/dilate.py --scene "$SCENE"

echo "[2/3] camera poses (COLMAP)"
step colmap colmap_poses

echo "[3/3] half-size frames and masks, test split"
step downsample python scripts/downsample_scene.py "$SCENE"
echo "done: $SCENE"
