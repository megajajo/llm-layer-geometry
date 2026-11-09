#!/bin/bash
# E1 run matrix on Llama-3-8B (agreed 2026-11-09): untrained control (random init, seeds 0-2),
# FineWeb short vs 256-token windows from the same documents, document-bootstrap peak intervals.
# Run on a login node from the repository root, after the smoke tests:
#   FW_SHORT=<N> bash slurm/submit_e1.sh
# N = short FineWeb window in model tokens (median WikiText-20 n_tokens minus BOS), the value
# passed to make_sequences_fineweb.py --short-tokens. Each run: extract -> measure array + bootstrap.
# extract.sh refuses to overwrite a complete run; its dependants then never start (scancel them).
set -eo pipefail
: "${FW_SHORT:?set FW_SHORT to the --short-tokens value used for the FineWeb sequences}"
mkdir -p logs

WT_N=wikitext103_train_normal_n50000_len20_seed0.jsonl
WT_S=wikitext103_train_shuffled_n50000_len20_seed0.jsonl
FW_N=fineweb10bt_normal_n50000_len256_seed0.jsonl
FW_S=fineweb10bt_shuffled_n50000_len256_seed0.jsonl
FWS_N=fineweb10bt_normal_n50000_len${FW_SHORT}_seed0.jsonl
FWS_S=fineweb10bt_shuffled_n50000_len${FW_SHORT}_seed0.jsonl
LONG=(--time=03:00:00)   # sbatch options for 256-token runs

jobid() { cut -d';' -f1; }   # --parsable prints "<id>;<cluster>"

# submit [sbatch options --] <name> <sequences file> [extract.py args]
submit() {
  local opts=()
  if [ "$1" != "${1#--}" ]; then while [ "$1" != "--" ]; do opts+=("$1"); shift; done; shift; fi
  local name=$1 seqs=$2; shift 2
  local ex
  ex=$(sbatch --parsable "${opts[@]}" slurm/extract.sh "$name" "$seqs" "$@" | jobid)
  local me bo
  me=$(sbatch --parsable --dependency=afterok:$ex --array=0-33 slurm/measure.sh "$name" | jobid)
  bo=$(sbatch --parsable --dependency=afterok:$ex slurm/bootstrap.sh "$name" | jobid)
  echo "$name: extract $ex, measure $me, bootstrap $bo"
}

# Untrained control on WikiText-20
for seed in 0 1 2; do
  submit "llama3-8b-rand${seed}_normal"   "$WT_N" --init random --seed "$seed"
  submit "llama3-8b-rand${seed}_shuffled" "$WT_S" --init random --seed "$seed"
done

# FineWeb, same documents at two lengths
submit "llama3-8b_fw-short_normal"   "$FWS_N"
submit "llama3-8b_fw-short_shuffled" "$FWS_S"
submit "${LONG[@]}" -- "llama3-8b_fw256_normal"   "$FW_N" --batch-size 32
submit "${LONG[@]}" -- "llama3-8b_fw256_shuffled" "$FW_S" --batch-size 32
submit "llama3-8b-rand0_fw-short_normal" "$FWS_N" --init random --seed 0
submit "${LONG[@]}" -- "llama3-8b-rand0_fw256_normal" "$FW_N" --init random --seed 0 --batch-size 32

# Bootstrap for the two runs extracted on 2026-11-05
for name in llama3-8b_normal llama3-8b_shuffled; do
  echo "$name: bootstrap $(sbatch --parsable slurm/bootstrap.sh "$name" | jobid)"
done
