#!/usr/bin/env bash
# Stage 3: COCO 上的剪枝模型蒸馏训练单一入口脚本
# 目标：稳定冲击“核心精度损失 <= 2%”
# 顺序执行：先 Pruned20，再 Pruned30

set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate PoseBH

BASELINE_AP="${BASELINE_AP:-0.7728}"      # Stage1 COCO baseline AP（可填 0.7728 或 77.28）
TARGET_LOSS_PCT="${TARGET_LOSS_PCT:-2.0}" # 目标：核心精度损失 <= 2%

mkdir -p experiments/DIST/logs

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

run_distill() {
  local pruned_name="$1"
  local cfg="$2"
  local work_dir="$3"
  local log_file="$4"
  local summary_log="$5"
  
  local latest_ckpt="${work_dir}/latest.pth"
  local resume="${RESUME:-0}"
  
  TARGET_AP=$(python - <<PY
baseline = float("${BASELINE_AP}")
if baseline > 1.0:
    baseline = baseline / 100.0
target_loss = float("${TARGET_LOSS_PCT}") / 100.0
print(f"{baseline * (1.0 - target_loss):.4f}")
PY
  )
  
  echo "=========================================="
  echo "Stage3 Distill (${pruned_name}) 启动参数"
  echo "CFG         : ${cfg}"
  echo "WORK_DIR    : ${work_dir}"
  echo "FULL LOG    : ${log_file}"
  echo "SUMMARY LOG : ${summary_log}"
  echo "BASELINE AP : ${BASELINE_AP} (支持 0~1 或 0~100 输入)"
  echo "TARGET AP   : >= ${TARGET_AP} (loss <= ${TARGET_LOSS_PCT}%)"
  echo "=========================================="
  
  mkdir -p "${work_dir}"
  
  run_train() {
    local mode="$1"  # append | overwrite
    shift
    if [ "$mode" = "append" ]; then
      "$@" |& tee -a "${log_file}" >(stdbuf -oL "${AWK_BIN}" "${stream_filter}" >> "${summary_log}")
    else
      "$@" |& tee "${log_file}" >(stdbuf -oL "${AWK_BIN}" "${stream_filter}" > "${summary_log}")
    fi
  }
  
  if [ "${resume}" = "1" ] && [ -f "${latest_ckpt}" ]; then
    echo "检测到断点：${latest_ckpt}"
    echo "模式：RESUME=1，执行断点续训"
    run_train append bash tools/dist_train.sh "${cfg}" 1 --seed 0 --resume-from "${latest_ckpt}"
  else
    if [ "${resume}" = "1" ] && [ ! -f "${latest_ckpt}" ]; then
      echo "RESUME=1 但未找到 ${latest_ckpt}，自动改为从头训练。"
    fi
    echo "模式：从头训练（推荐）"
    run_train overwrite bash tools/dist_train.sh "${cfg}" 1 --seed 0
  fi
  
  BEST_AP=$(grep -E "Best AP is" "${log_file}" | tail -n 1 | sed -E 's/.*Best AP is ([0-9.]+).*/\1/' || true)
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
}

# 先执行 Pruned20 蒸馏
run_distill "Pruned20" "experiments/DIST/pruned20_distill_coco.py" "experiments/DIST/work_dirs/pruned20_distill_coco" "experiments/DIST/logs/distill_pruned20_coco.log" "experiments/DIST/logs/distill_pruned20_coco.summary.log"

# 再执行 Pruned30 蒸馏
run_distill "Pruned30" "experiments/DIST/pruned30_distill_coco.py" "experiments/DIST/work_dirs/pruned30_distill_coco" "experiments/DIST/logs/distill_pruned30_coco.log" "experiments/DIST/logs/distill_pruned30_coco.summary.log"

# 保留原脚本文件，以便单独运行

echo "=========================================="
echo "所有蒸馏任务已完成！"
echo "=========================================="
