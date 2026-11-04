#!/bin/bash
# Stage 1 smoke test (~2 min, devel partition): tiny random Llama with the real tokenizer.
# Tests: env activation, GPU visible, offline HF cache, write to $SCRATCH, sync to $DATA, checker.
#SBATCH --clusters=htc
#SBATCH --partition=devel
#SBATCH --time=00:10:00
#SBATCH --job-name=smoke-tiny
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out

set -eo pipefail
source slurm/common.sh
OUT=$SCRATCH/smoke_tiny
DEST=$RUNS/smoke_tiny_$SLURM_JOB_ID
trap 'sync_back $OUT $DEST; exit 2' SIGINT SIGTERM

python scripts/extract.py --tiny --sequences "$SEQS/wikitext103_train_normal_n50000_len20_seed0.jsonl" \
  --limit 512 --out "$OUT"
python scripts/measure.py --run "$OUT" --partitions 1 --partition-size 512
sync_back "$OUT" "$DEST"
python scripts/check_run.py "$DEST"
