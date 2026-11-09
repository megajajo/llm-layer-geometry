#!/bin/bash
# Pack the conda env into one file that every job unpacks onto node-local disk (common.sh).
# Importing straight from $DATA took 5-8 min per Python start in the smoke jobs (thousands of
# small files over the network filesystem); one large file copies much faster.
# Run on an interactive node after setup_env.sh and again after EVERY change to the env
# (jobs refuse to start if the env is newer than its pack).
set -eo pipefail
export THESIS_ROOT=${THESIS_ROOT:-$DATA/thesis}
ENV=$THESIS_ROOT/envs/thesis

module purge
module load Anaconda3/2025.06-1
source activate "$ENV"
pip install conda-pack
# layergeom is installed editable, so it is left out; jobs import it from the repo (PYTHONPATH)
conda-pack -p "$ENV" --format tar -o "$ENV.partial.tar" --ignore-editable-packages --force
mv "$ENV.partial.tar" "$ENV.tar"     # only a finished pack ever has the name jobs look for
ls -lh "$ENV.tar"
