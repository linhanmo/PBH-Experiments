set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate PoseBH

CFG="${CFG:-vitb_posebh_pruned40.py}"
WORK_DIR="${WORK_DIR:-work_dirs/pruned40_coco/}"
WEIGHT_OUT="${WEIGHT_OUT:-weights/pruned40_base_init.pth}"
LOG_FILE="${LOG_FILE:-logs/prune_pruned40.log}"
RESUME="${RESUME:-0}"

mkdir -p logs
mkdir -p weights
mkdir -p "${WORK_DIR}"

echo "=========================================="
echo "Stage 2 Pruned40 剪枝"
echo "CFG       : ${CFG}"
echo "WORK_DIR  : ${WORK_DIR}"
echo "WEIGHT_OUT: ${WEIGHT_OUT}"
echo "LOG_FILE  : ${LOG_FILE}"
echo "RESUME    : ${RESUME}"
echo "=========================================="

if [ "${RESUME}" = "1" ] && [ -f "${WEIGHT_OUT}" ]; then
  echo "检测到已有权重，跳过剪枝（如需重新剪枝请删除 ${WEIGHT_OUT}）"
  exit 0
fi

python -m experiments.CUT.prune_main \
  --config "${CFG}" \
  --save-weights "${WEIGHT_OUT}" \
  --seed 0 \
  2>&1 | tee "${LOG_FILE}"

echo "剪枝完成: ${WEIGHT_OUT}"