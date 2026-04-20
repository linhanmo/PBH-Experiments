#!/usr/bin/env bash
# Stage 2: Pruned20 微调脚本
# 目标：生成 best_AP_epoch_*.pth

set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate PoseBH

CFG="${CFG:-experiments/CUT/pruned20_coco_finetune.py}"
WORK_DIR="${WORK_DIR:-experiments/CUT/work_dirs/pruned20_coco_finetune}"
LOG_FILE="${LOG_FILE:-experiments/CUT/logs/finetune_pruned20_coco.log}"
SUMMARY_LOG="${SUMMARY_LOG:-experiments/CUT/logs/finetune_pruned20_coco.summary.log}"
LATEST_CKPT="${WORK_DIR}/latest.pth"
RESUME="${RESUME:-0}"

mkdir -p experiments/CUT/logs
mkdir -p "${WORK_DIR}"

AWK_BIN="awk"
if command -v gawk >/dev/null 2>&1; then
  AWK_BIN="gawk"
fi

stream_filter='
/(Epoch\(val\)|Best AP is|Now best checkpoint|ERROR|Traceback|FPS)/ {
  print $0;
  fflush();
}
/\[.*\]/ {
  next;
}'

echo "=========================================="
echo "Stage 2 Pruned20 微调"
echo "CFG       : ${CFG}"
echo "WORK_DIR  : ${WORK_DIR}"
echo "LOG_FILE  : ${LOG_FILE}"
echo "=========================================="

run_train() {
  local mode="$1"
  shift
  if [ "$mode" = "append" ]; then
    "$@" |& tee -a "${LOG_FILE}" >(stdbuf -oL "${AWK_BIN}" "${stream_filter}" >> "${SUMMARY_LOG}")
  else
    "$@" |& tee "${LOG_FILE}" >(stdbuf -oL "${AWK_BIN}" "${stream_filter}" > "${SUMMARY_LOG}")
  fi
}

if [ "${RESUME}" = "1" ] && [ -f "${LATEST_CKPT}" ]; then
  echo "检测到断点：${LATEST_CKPT}"
  echo "模式：RESUME=1，执行断点续训"
  run_train append bash tools/dist_train.sh "${CFG}" 1 --seed 0 --resume-from "${LATEST_CKPT}"
else
  if [ "${RESUME}" = "1" ] && [ ! -f "${LATEST_CKPT}" ]; then
    echo "RESUME=1 但未找到 ${LATEST_CKPT}，自动改为从头训练。"
  fi
  echo "模式：从头训练（推荐）"
  run_train overwrite bash tools/dist_train.sh "${CFG}" 1 --seed 0
fi

echo "微调完成"