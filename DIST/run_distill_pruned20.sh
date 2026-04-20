#!/usr/bin/env bash
# Stage 3: COCO 上的剪枝模型蒸馏训练单一入口脚本（Pruned20）
# 目标：稳定冲击“核心精度损失 <= 2%”

set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate PoseBH

CFG="${CFG:-experiments/DIST/pruned20_distill_coco.py}"
WORK_DIR="${WORK_DIR:-experiments/DIST/work_dirs/pruned20_distill_coco}"
LOG_FILE="${LOG_FILE:-experiments/DIST/logs/distill_pruned20_coco.log}"
SUMMARY_LOG="${SUMMARY_LOG:-experiments/DIST/logs/distill_pruned20_coco.summary.log}"
LATEST_CKPT="${WORK_DIR}/latest.pth"
BASELINE_AP="${BASELINE_AP:-0.7728}"      # Stage1 COCO baseline AP（可填 0.7728 或 77.28）
TARGET_LOSS_PCT="${TARGET_LOSS_PCT:-2.0}" # 目标：核心精度损失 <= 2%

mkdir -p experiments/DIST/logs
mkdir -p "${WORK_DIR}"

# 用法：
# 1) 默认（推荐）：从头跑，避免继续沿用历史不佳轨迹
#    bash experiments/DIST/run_distill_pruned20.sh
# 2) 断点续训：
#    RESUME=1 bash experiments/DIST/run_distill_pruned20.sh
RESUME="${RESUME:-0}"

TARGET_AP=$(python - <<PY
baseline = float("${BASELINE_AP}")
if baseline > 1.0:
    baseline = baseline / 100.0
target_loss = float("${TARGET_LOSS_PCT}") / 100.0
print(f"{baseline * (1.0 - target_loss):.4f}")
PY
)

echo "=========================================="
echo "Stage3 Distill (Pruned20) 启动参数"
echo "CFG         : ${CFG}"
echo "WORK_DIR    : ${WORK_DIR}"
echo "FULL LOG    : ${LOG_FILE}"
echo "SUMMARY LOG : ${SUMMARY_LOG}"
echo "BASELINE AP : ${BASELINE_AP} (支持 0~1 或 0~100 输入)"
echo "TARGET AP   : >= ${TARGET_AP} (loss <= ${TARGET_LOSS_PCT}%)"
echo "=========================================="

AWK_BIN="awk"
if command -v gawk >/dev/null 2>&1; then
  AWK_BIN="gawk"
fi

stream_filter='
/(Epoch\(val\)|Best AP is|Now best checkpoint|ERROR|Traceback|The model and loaded state dict do not match exactly|size mismatch|missing keys in source state_dict|unexpected key in source state_dict)/ {
  print $0;
  fflush();
}
/\[.*\]/ {
  next;
}'
# 过滤掉验证时的进度条输出

run_train() {
  local mode="$1"  # append | overwrite
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

BEST_AP=$(grep -E "Best AP is" "${LOG_FILE}" | tail -n 1 | sed -E 's/.*Best AP is ([0-9.]+).*/\1/' || true)
if [ -n "${BEST_AP}" ]; then
  python - <<PY
baseline = float("${BASELINE_AP}")
if baseline > 1.0:
    baseline = baseline / 100.0
best_ap = float("${BEST_AP}")
target_ap = float("${TARGET_AP}")
loss_pct = (baseline - best_ap) / baseline * 100.0
status = "达标" if best_ap >= target_ap else "未达标"
print("==========================================")
print(f"训练结束: Best AP = {best_ap:.4f}")
print(f"相对精度损失 = {loss_pct:.2f}% (Baseline={baseline:.4f})")
print(f"目标判定: {status} (目标 AP >= {target_ap:.4f})")
print("==========================================")
PY
else
  echo "未在日志中解析到 Best AP，请检查训练是否正常结束。"
fi