#!/bin/bash
#SBATCH --job-name=gr00t-n16-openarm
#SBATCH --output=/coc/testnvme/xzhang3205/tmp/gr00t-n16-openarm-%J.out
#SBATCH --error=/coc/testnvme/xzhang3205/tmp/gr00t-n16-openarm-%J.err
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
REPO_ROOT=/coc/testnvme/xzhang3205/vla-adaptation; ROOT=${REPO_ROOT}/models/gr00t-n1.6
source "${ROOT}/.venv/bin/activate"; cd "${REPO_ROOT}"
export PYTHONPATH=${REPO_ROOT}/static-inference:${REPO_ROOT}:${ROOT}:${PYTHONPATH:-} TMPDIR=/coc/testnvme/xzhang3205/tmp
: "${MANIFEST:?}" "${EPISODE_SEQUENCE:?}" "${OUTPUT_DIR:?}"; mkdir -p "${OUTPUT_DIR}" "${TMPDIR}"
"${ROOT}/.venv/bin/python" "${ROOT}/static_inference/run_openarm_static_inference.py" --manifest "${MANIFEST}" --episode-sequence "${EPISODE_SEQUENCE}" --output-dir "${OUTPUT_DIR}" --save-meta ${EXTRA_ARGS:-}
