#!/bin/bash
# Measures for one extracted run, CPU only, one array task per site.
#   sbatch --array=0-33 slurm/measure.sh <run name> [extra measure.py args]
# Llama-3-8B has 34 sites (layer_00..layer_32, final_norm) -> --array=0-33.
# Reads states directly from $DATA (memory-mapped); each task writes its own metrics file.
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --time=06:00:00
#SBATCH --job-name=measure
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/%x_%A_%a.out

set -eo pipefail
NAME=$1; shift
source slurm/common.sh
RUN=$RUNS/$NAME
SITE=$(python -c "import json,sys; print(json.load(open('$RUN/meta.json'))['sites'][int(sys.argv[1])])" "$SLURM_ARRAY_TASK_ID")
echo "site $SITE"
python scripts/measure.py --run "$RUN" --sites "$SITE" --out "metrics_$SITE.jsonl" "$@"
