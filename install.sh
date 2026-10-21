#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Install SwitchSplat and everything its pipeline needs into one conda environment:
#   bash install.sh [ENV_NAME]     (default: switchsplat; needs conda, uv and a CUDA 12.x driver)
# The CUDA code is built for the GPUs of this machine, with one compile job per CPU core; set
# TORCH_CUDA_ARCH_LIST (e.g. "8.0;8.9;12.0+PTX") and MAX_JOBS to change that.
# 1. conda environment with the CUDA 12.8 toolkit and a matching compiler (scripts/create_conda.sh)
# 2. PyTorch, the renderers' CUDA extensions and gsplat (install_env_uv.sh)
# 3. preprocessing of new captures: COLMAP (CUDA), YOLO and SAM for the camera-holder masks
# 4. interaction windows: the Codex CLI (version used for the paper)
# 5. object masks for the refinement: transformers, SAM 2 and the SAM 2.1 weights (checkpoints/)
set -euo pipefail

ENV_NAME=${1:-switchsplat}
cd "$(dirname "${BASH_SOURCE[0]}")"
eval "$(conda shell.bash hook)"

if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ] && nvidia-smi -L > /dev/null 2>&1; then
    TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -t. -k1,1n -k2,2n -u | paste -sd ';')+PTX"
    export TORCH_CUDA_ARCH_LIST
fi
export MAX_JOBS=${MAX_JOBS:-$(nproc)}

CUDA_VERSION=${CUDA_VERSION:-12.8} bash scripts/create_conda.sh "$ENV_NAME"
set +u; conda activate "$ENV_NAME"; set -u
bash install_env_uv.sh "$ENV_NAME"

set +u; conda install -y -c conda-forge "colmap=3.12.6=cuda*" "nodejs>=20"; set -u
uv pip install ultralytics segment-anything "pycolmap==3.12.6"
npm install -g @openai/codex@0.148.0

uv pip install "transformers==4.57.1"
SAM2_BUILD_CUDA=0 uv pip install --no-build-isolation \
    "git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4"
mkdir -p checkpoints
for url in https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt \
           https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
           https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt; do
    [ -f "checkpoints/$(basename "$url")" ] || wget -q -P checkpoints "$url"
done

echo "SwitchSplat is installed. Activate it with: conda activate $ENV_NAME"
