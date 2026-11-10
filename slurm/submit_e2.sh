#!/bin/bash
# E2 on Llama-3-8B, FineWeb-256 (agreed 2026-11-10): competitor geometry on the existing runs,
# plus untrained seeds 1-2 at 256 tokens (extract -> competitor, measure array, bootstrap).
# Run on a login node from the repository root:  bash slurm/submit_e2.sh
set -eo pipefail
mkdir -p logs
jobid() { cut -d';' -f1; }   # --parsable prints "<id>;<cluster>"

for name in llama3-8b_fw256_normal llama3-8b_fw256_shuffled llama3-8b-rand0_fw256_normal; do
  echo "$name: competitor $(sbatch --parsable slurm/competitor.sh "$name" | jobid)"
done

for seed in 1 2; do
  name=llama3-8b-rand${seed}_fw256_normal
  ex=$(sbatch --parsable --time=03:00:00 slurm/extract.sh "$name" fineweb10bt_normal_n50000_len256_seed0.jsonl \
       --init random --seed "$seed" --batch-size 32 | jobid)
  co=$(sbatch --parsable --dependency=afterok:$ex slurm/competitor.sh "$name" | jobid)
  me=$(sbatch --parsable --dependency=afterok:$ex --array=0-33 slurm/measure.sh "$name" | jobid)
  bo=$(sbatch --parsable --dependency=afterok:$ex slurm/bootstrap.sh "$name" | jobid)
  echo "$name: extract $ex, competitor $co, measure $me, bootstrap $bo"
done
