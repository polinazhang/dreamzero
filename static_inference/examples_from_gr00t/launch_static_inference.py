"""Generate and submit one sbatch job per task for static inference.

Usage:
    # submit task 2
    python static_inference/launch_static_inference.py --task_id 2

    # submit all 18 tasks
    python static_inference/launch_static_inference.py --all

    # pass extra args through to run_static_inference.py
    python static_inference/launch_static_inference.py --task_id 2 --extra_args "--save_meta"

    # only write the sbatch script(s), do not submit
    python static_inference/launch_static_inference.py --task_id 2 --no_submit

Each job requests 1 node / 32 cpus / 1x A40 / qos long / 3 days on kira-lab,
mirroring static_inference/prompts/template.sbatch (GPU count reduced to 1
since the runner is a single-process, single-GPU workload). Logs go to
/coc/testnvme/xzhang3205/vla-adaptation/slurms/.
"""

import argparse
from pathlib import Path
import subprocess

REPO_ROOT = Path("/coc/testnvme/xzhang3205/vla-adaptation")
GR00T_ROOT = REPO_ROOT / "models" / "gr00t-n1.6"
STATIC_DIR = GR00T_ROOT / "static_inference"
SLURM_DIR = REPO_ROOT / "slurms"
GENERATED_DIR = STATIC_DIR / "generated_sbatch"

SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=static-gr00t-task{task_id}
#SBATCH --output={slurm_dir}/static-gr00t-task{task_id}-%J.out
#SBATCH --error={slurm_dir}/static-gr00t-task{task_id}-%J.err
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

REPO_ROOT="{repo_root}"
GR00T_ROOT="${{REPO_ROOT}}/models/gr00t-n1.6"

cd "${{REPO_ROOT}}"
mkdir -p "${{REPO_ROOT}}/slurms"
source "${{GR00T_ROOT}}/.venv/bin/activate"

export PYTHONPATH="${{REPO_ROOT}}:${{GR00T_ROOT}}:${{PYTHONPATH:-}}"
export XDG_CACHE_HOME="/coc/testnvme/xzhang3205/.cache"
export HF_HOME="/coc/testnvme/xzhang3205/huggingface"
export TRANSFORMERS_CACHE="/coc/testnvme/xzhang3205/huggingface"
export LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:${{LD_LIBRARY_PATH:-}}"
export MUJOCO_GL="${{MUJOCO_GL:-egl}}"

"${{GR00T_ROOT}}/.venv/bin/python" "${{GR00T_ROOT}}/static_inference/run_static_inference.py" \\
    --task_id {task_id} {extra_args}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--task_id", type=int, help="Task id, 1..18")
    group.add_argument("--all", action="store_true", help="Launch tasks 1..18")
    parser.add_argument(
        "--extra_args",
        type=str,
        default="",
        help='Extra args forwarded to run_static_inference.py, e.g. "--save_meta"',
    )
    parser.add_argument("--no_submit", action="store_true", help="Only write sbatch scripts")
    args = parser.parse_args()

    task_ids = list(range(1, 19)) if args.all else [args.task_id]
    for task_id in task_ids:
        assert 1 <= task_id <= 18, f"task_id must be in 1..18, got {task_id}"

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    SLURM_DIR.mkdir(parents=True, exist_ok=True)

    for task_id in task_ids:
        script = SBATCH_TEMPLATE.format(
            task_id=task_id,
            slurm_dir=SLURM_DIR,
            repo_root=REPO_ROOT,
            extra_args=args.extra_args,
        )
        script_path = GENERATED_DIR / f"static_inference_task_{task_id}.sbatch"
        script_path.write_text(script)
        print(f"Wrote {script_path}")
        if not args.no_submit:
            result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
            print(result.stdout.strip() or result.stderr.strip())
            result.check_returncode()


if __name__ == "__main__":
    main()
