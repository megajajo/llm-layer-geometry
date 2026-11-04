#!/bin/bash
# One-time setup. NOT an sbatch script: run it step by step on an interactive node,
#   srun -M htc -p interactive --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=03:00:00 --pty /bin/bash
# after copying this repository to $DATA/thesis/code (rsync from the Mac, see README).
set -eo pipefail

export THESIS_ROOT=${THESIS_ROOT:-$DATA/thesis}
export HF_HOME=$THESIS_ROOT/hf
mkdir -p "$THESIS_ROOT"/{envs,hf,sequences,runs}
cd "$THESIS_ROOT/code"

# 1. Environment (ARC's documented PyTorch recipe, ARC.md)
module purge
module load Anaconda3/2025.06-1
if [ ! -d "$THESIS_ROOT/envs/thesis" ]; then
  conda create -y --prefix "$THESIS_ROOT/envs/thesis" python=3.11
fi
source activate "$THESIS_ROOT/envs/thesis"
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
pip install -e ".[extract,dev]" bitsandbytes
pip install "dadapy @ git+https://github.com/sissa-data-science/DADApy@d729b8835e771001ec7ba6ca68938b737b96edae"
mkdir -p env && pip freeze > env/arc-freeze.txt     # commit this file
python -c 'import torch; print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0))'
python /apps/common/bin/pytorch-gpu-cc.py            # confirms which gpu_cc constraints this torch supports
pytest -q

# 2. Hugging Face login (Llama-3 is gated: accept the licence on huggingface.co first).
#    The token is stored under $HF_HOME on $DATA, not in $HOME.
hf auth whoami || hf auth login

# 3. Download weights + dataset and pin their revisions (writes configs/revisions.json: commit it)
python scripts/prefetch.py --model meta-llama/Meta-Llama-3-8B

# 4. Input sequences, shared by every model and condition (~1 min)
python scripts/make_sequences.py --n 50000 --seed 0 --out "$THESIS_ROOT/sequences"
ls -lh "$THESIS_ROOT/sequences"
