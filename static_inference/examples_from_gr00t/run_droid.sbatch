#!/bin/bash
#SBATCH --job-name=static-gr00t-n16-droid
#SBATCH --output=/coc/testnvme/xzhang3205/tmp/static-gr00t-n16-droid-%J.out
#SBATCH --error=/coc/testnvme/xzhang3205/tmp/static-gr00t-n16-droid-%J.err
#SBATCH --partition=kira-lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos=long
#SBATCH --time=3-00:00:00
#SBATCH --exclude=clippy,cyborg,ig-88,megazord
#SBATCH --mem-per-gpu=45G

source ~/.bashrc
set -euo pipefail
REPO_ROOT=/coc/testnvme/xzhang3205/vla-adaptation
GR00T_ROOT="${REPO_ROOT}/models/gr00t-n1.6"
source "${GR00T_ROOT}/.venv/bin/activate"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/static-inference:${REPO_ROOT}:${GR00T_ROOT}:${PYTHONPATH:-}"
export XDG_CACHE_HOME=/coc/testnvme/xzhang3205/.cache
export HF_HOME=/coc/testnvme/xzhang3205/huggingface
export TRANSFORMERS_CACHE=/coc/testnvme/xzhang3205/huggingface
export TMPDIR=/coc/testnvme/xzhang3205/tmp
export LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
: "${MANIFEST:?}" "${TRAJECTORY_INDEX:?}" "${OUTPUT_DIR:?}"
mkdir -p "${OUTPUT_DIR}" "${TMPDIR}"
"${GR00T_ROOT}/.venv/bin/python" "${GR00T_ROOT}/static_inference/run_droid_static_inference.py" \
  --manifest "${MANIFEST}" --trajectory-index "${TRAJECTORY_INDEX}" \
  --output-dir "${OUTPUT_DIR}" --save-meta ${EXTRA_ARGS:-}
