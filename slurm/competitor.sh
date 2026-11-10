#!/bin/bash
# E2: endpoint-bank competitor geometry for one extracted run, CPU only.
#   sbatch slurm/competitor.sh <run name> [extra competitor.py args]
# Reads states directly from $DATA; writes <run>/competitor/.
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --job-name=competitor
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/%x_%j.out

set -eo pipefail
NAME=$1; shift
source slurm/common.sh
python scripts/competitor.py --run "$RUNS/$NAME" "$@"
