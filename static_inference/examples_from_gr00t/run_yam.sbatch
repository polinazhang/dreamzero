#!/bin/bash
#SBATCH --job-name=gr00t-n16-yam
#SBATCH --output=/coc/testnvme/xzhang3205/tmp/gr00t-n16-yam-%J.out
#SBATCH --error=/coc/testnvme/xzhang3205/tmp/gr00t-n16-yam-%J.err
#SBATCH --partition=kira-lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=a40:1
#SBATCH --qos=long
#SBATCH --time=3-00:00:00
#SBATCH --mem-per-gpu=45G
source ~/.bashrc
set -euo pipefail
REPO_ROOT=/coc/testnvme/xzhang3205/vla-adaptation
ROOT=${REPO_ROOT}/models/gr00t-n1.6
source "${ROOT}/.venv/bin/activate"
cd "${REPO_ROOT}"
export PYTHONPATH=${REPO_ROOT}/static-inference:${REPO_ROOT}:${ROOT}:${PYTHONPATH:-} TMPDIR=/coc/testnvme/xzhang3205/tmp
: "${DATASET_ROOT:?}" "${TRAJECTORY_INDEX:?}" "${OUTPUT_DIR:?}"
mkdir -p "${OUTPUT_DIR}" "${TMPDIR}"
"${ROOT}/.venv/bin/python" "${ROOT}/static_inference/run_yam_static_inference.py" --dataset-root "${DATASET_ROOT}" --trajectory-index "${TRAJECTORY_INDEX}" --output-dir "${OUTPUT_DIR}" --save-meta ${EXTRA_ARGS:-}
