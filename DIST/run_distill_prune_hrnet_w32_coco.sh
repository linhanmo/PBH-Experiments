#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/rivermind-data/PoseBH"
CFG="${ROOT_DIR}/experiments/DIST/hrnet_w32_distill_prune_coco_256x192.py"
WORK_DIR="${ROOT_DIR}/experiments/DIST/work_dirs/hrnet_w32_distill_prune_coco_256x192"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python "${ROOT_DIR}/tools/train.py" "${CFG}" --work-dir "${WORK_DIR}" "${@}"
