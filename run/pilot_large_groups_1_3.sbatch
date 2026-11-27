#!/bin/bash -l
# Public template: replace YOUR_ACCOUNT and /path/to/code-switching before submission.
#SBATCH --job-name=cs_pilot_large
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=debug-quad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --chdir=/path/to/code-switching
#SBATCH --output=/path/to/code-switching/logs/pilot_large_%j.out
#SBATCH --error=/path/to/code-switching/logs/pilot_large_%j.err

set -euo pipefail
umask 077
project_dir=/path/to/code-switching
[[ $(pwd -P) == "$project_dir" ]] || exit 2
[[ -n ${CUDA_VISIBLE_DEVICES:-} ]] || { printf 'No allocated GPU detected\n' >&2; exit 2; }
python3 - <<'PY'
import subprocess
from pathlib import Path
root = Path('/path/to/code-switching')
assert root.resolve(strict=True) == root
for name in ['ollama_models', 'logs', '.runtime', 'outputs']:
    path = root / name
    path.resolve().relative_to(root)
    assert not path.is_symlink(), path
memory = subprocess.check_output([
    'nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'
], text=True)
assert memory.strip() and min(int(value) for value in memory.split()) >= 80000, 'An 80GB GPU is required'
PY

module load ollama/0.32.0
export OLLAMA_MODELS="$project_dir/ollama_models"
runtime_dir="$project_dir/.runtime/pilot_large_${SLURM_JOB_ID}"
runtime_home="$runtime_dir/home"
export APPTAINER_CACHEDIR="$runtime_dir/cache"
export APPTAINER_TMPDIR="$runtime_dir/tmp"
mkdir -p "$runtime_home" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
port=$(python3 - <<'PY'
import socket
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 0))
    print(sock.getsockname()[1])
PY
)
server_pid=
cleanup() {
    if [[ -n "$server_pid" ]]; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

printf 'Job %s on %s; CUDA_VISIBLE_DEVICES=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv,noheader
apptainer exec --nv --cleanenv --home "$runtime_home" \
    --bind "$project_dir:$project_dir" \
    --env "OLLAMA_MODELS=$OLLAMA_MODELS,OLLAMA_HOST=127.0.0.1:$port,CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES,OLLAMA_MAX_LOADED_MODELS=1,OLLAMA_NUM_PARALLEL=1" \
    "$OLLAMA_IMG" ollama serve > "$project_dir/logs/ollama_pilot_large_${SLURM_JOB_ID}.log" 2>&1 &
server_pid=$!
python3 -u scripts/run_ollama_pilot.py --port "$port"
