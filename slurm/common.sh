# Sourced by every job script: environment, offline Hugging Face, paths, result sync.
# Jobs are submitted from the repository root (wherever it is cloned); data lives under
# THESIS_ROOT, outside the repository (override if needed):
#   $THESIS_ROOT/envs/thesis conda env (+ thesis.tar, its pack: slurm/pack_env.sh)
#   $THESIS_ROOT/hf          Hugging Face cache (model weights, datasets)
#   $THESIS_ROOT/sequences   input sequences (make_sequences.py)
#   $THESIS_ROOT/runs        one directory per run, synced back from $SCRATCH

export THESIS_ROOT=${THESIS_ROOT:-$DATA/thesis}
export HF_HOME=$THESIS_ROOT/hf
export HF_HUB_OFFLINE=1          # compute nodes: never download, fail fast if something is missing
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
SEQS=$THESIS_ROOT/sequences
RUNS=$THESIS_ROOT/runs

ENV=$THESIS_ROOT/envs/thesis
PACK=$ENV.tar

module purge
module load Anaconda3/2025.06-1
# Run from a copy of the env on node-local disk: imports from $DATA are very slow.
if [ -f "$PACK" ] && [ -n "$TMPDIR" ]; then
  if [ -n "$(find "$ENV"/lib/python3*/site-packages -mindepth 1 -maxdepth 1 -not -name __pycache__ -newer "$PACK" -print -quit)" ]; then
    echo "env changed since it was packed: run slurm/pack_env.sh first"; exit 1
  fi
  t=$SECONDS
  mkdir -p "$TMPDIR/env"
  tar -xf "$PACK" -C "$TMPDIR/env"
  source "$TMPDIR/env/bin/activate"
  conda-unpack
  echo "env unpacked to $TMPDIR/env in $((SECONDS - t))s"
else
  echo "no env pack (or no \$TMPDIR): using $ENV directly, imports will be slow"
  source activate "$ENV"
fi
export PYTHONPATH=$PWD/src${PYTHONPATH:+:$PYTHONPATH}   # layergeom is not in the pack

echo "job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "code commit $(git rev-parse --short HEAD 2>/dev/null) $(git status --porcelain 2>/dev/null | grep -q . && echo '(dirty)')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU"

# sync_back <scratch_dir> <dest_dir>: copy results to $DATA; also on the pre-timeout signal
sync_back() { mkdir -p "$2" && rsync -a "$1"/ "$2"/ && echo "synced to $2"; }
# run <cmd...>: run in the background and wait, so a trap fires as soon as the signal
# arrives (bash defers traps until a foreground command exits, i.e. never at a timeout)
run() { "$@" & wait $!; }
