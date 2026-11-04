#!/bin/bash
# Stage 2 smoke test (~15 min): real Llama-3-8B in bf16 on 1,024 normal sequences, all sites,
# spread + ID measures on one partition. Tests: weights load from cache, bf16 GPU, the padding
# and final-norm checks on the real model, fp16 range of real activations, end-to-end timing.
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --time=00:45:00
#SBATCH --job-name=smoke-llama
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='gpu_cc:8.0|gpu_cc:8.6|gpu_cc:8.9|gpu_cc:9.0'
#SBATCH --output=logs/%x_%j.out

set -eo pipefail
source slurm/common.sh
OUT=$SCRATCH/smoke_llama
DEST=$RUNS/smoke_llama_$SLURM_JOB_ID
trap 'sync_back $OUT $DEST; exit 2' SIGINT SIGTERM

python scripts/extract.py --model meta-llama/Meta-Llama-3-8B --dtype bf16 \
  --sequences "$SEQS/wikitext103_train_normal_n50000_len20_seed0.jsonl" --limit 1024 --out "$OUT"
python scripts/measure.py --run "$OUT" --partitions 1 --partition-size 1024
sync_back "$OUT" "$DEST"
python scripts/check_run.py "$DEST"
