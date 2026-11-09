#!/bin/bash
# Document bootstrap of the spread measures for one extracted run (peak-location intervals).
#   sbatch slurm/bootstrap.sh <run name> [extra bootstrap_spread.py args, e.g. --B 5]
# Reads states directly from $DATA; writes bootstrap_spread.jsonl into the run directory.
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --job-name=bootstrap
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint='gpu_cc:8.0|gpu_cc:8.6|gpu_cc:8.9|gpu_cc:9.0'
#SBATCH --output=logs/%x_%j.out

set -eo pipefail
NAME=$1; shift
source slurm/common.sh
python scripts/bootstrap_spread.py --run "$RUNS/$NAME" "$@"
