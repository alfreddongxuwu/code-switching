#!/bin/bash -l
# Public template: replace YOUR_ACCOUNT and /path/to/code-switching before submission.
#SBATCH --job-name=cs_gpt_full
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=debug-quad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --chdir=/path/to/code-switching
#SBATCH --output=/path/to/code-switching/logs/openai_full_%j.out
#SBATCH --error=/path/to/code-switching/logs/openai_full_%j.err

set -euo pipefail
umask 077
project_dir=/path/to/code-switching
[[ $(pwd -P) == "$project_dir" ]] || exit 2
python3 -u scripts/run_full_openai.py
