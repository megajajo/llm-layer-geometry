#!/bin/bash
# Full extraction for one condition. Only after both smoke tests pass.
#   sbatch slurm/extract.sh <name> <sequences.jsonl> [extra extract.py args]
# e.g.
#   sbatch slurm/extract.sh llama3-8b_normal  wikitext103_train_normal_n50000_len20_seed0.jsonl
#   sbatch slurm/extract.sh llama3-8b_shuffled wikitext103_train_shuffled_n50000_len20_seed0.jsonl
#   sbatch slurm/extract.sh llama3-8b-rand0_normal wikitext103_train_normal_n50000_len20_seed0.jsonl --init random --seed 0
#   sbatch slurm/extract.sh llama3-8b-int8_normal  wikitext103_train_normal_n50000_len20_seed0.jsonl --dtype int8 --limit 10000
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --job-name=extract
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint='gpu_cc:8.0|gpu_cc:8.6|gpu_cc:8.9|gpu_cc:9.0'
#SBATCH --output=logs/%x_%j.out
#SBATCH --signal=B:SIGINT@120   # trap fires 2 min before the time limit, results get synced

set -eo pipefail
NAME=$1; SEQFILE=$2; shift 2
source slurm/common.sh
OUT=$SCRATCH/$NAME
DEST=$RUNS/$NAME
if [ -f "$DEST/meta.json" ] && grep -q '"status": "complete"' "$DEST/meta.json"; then
  echo "refusing to overwrite complete run $DEST"; exit 1
fi
trap 'sync_back $OUT $DEST; exit 2' SIGINT SIGTERM

run python scripts/extract.py --model meta-llama/Meta-Llama-3-8B --sequences "$SEQS/$SEQFILE" --out "$OUT" "$@"
sync_back "$OUT" "$DEST"
