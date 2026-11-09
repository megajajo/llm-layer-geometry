# layergeom

Code for the thesis experiments on the layer-wise geometry of decoder-only LM hidden states. Research notes live in the parent `Thesis/` folder; this repository holds only code, configs and small result summaries. Hidden states and model weights live on ARC `$DATA` and are never committed.

## Local setup (Mac)

```bash
uv venv --python 3.11
uv pip install -e ".[dev,notebook,extract]"
# DADApy: the PyPI sdist fails to build against NumPy 2, so install the pinned GitHub commit
uv pip install "dadapy @ git+https://github.com/sissa-data-science/DADApy@d729b8835e771001ec7ba6ca68938b737b96edae"
pytest
```

## Pipeline

1. `scripts/prefetch.py`: download weights and dataset, pin revisions in `configs/revisions.json`.
2. `scripts/make_sequences.py`: Cheng-style 20-token WikiText-103 sequences, normal and corpus-shuffled, shared by every model and condition.
2b. `scripts/make_sequences_fineweb.py`: FineWeb sample-10BT windows stored as Llama token ids, normal and word-shuffled, at two lengths from the same documents (256 tokens, and the last N tokens of the same window).
3. `scripts/extract.py`: last-token states at every site (`layer_00` embeddings … `layer_32` pre-norm, `final_norm`), saved as one float16 `.npy` per site, plus `meta.json` with provenance and sanity checks (padding invariance, final-norm identity).
4. `scripts/measure.py`: spread (participation ratio, eigenvalue entropy) and ID (TwoNN, ABIDE, GRIDE across scales) per site × partition × variant (`raw`, `unit`, `no_outliers`) to JSONL.
4b. `scripts/bootstrap_spread.py`: document bootstrap (default B = 200) of the spread measures at every site, on GPU, for peak-location intervals; `bootstrap_spread.jsonl`.
5. `scripts/check_run.py`: validates a run directory; exits non-zero on any problem.

Every output carries a `status` field that is set to `complete` only at the very end; anything else is unusable.

## On ARC

Clone the repository anywhere under `$DATA` (e.g. `$DATA/llm-layer-geometry`). Environment, Hugging Face cache, sequences and runs go to `$DATA/thesis` (`THESIS_ROOT`), outside the repository.

Then, in order, from the repository root (submit from there; `mkdir -p logs` first):

1. **Setup**, interactively: follow `slurm/setup_env.sh` step by step on an interactive node. Commit the resulting `configs/revisions.json` and `env/arc-freeze.txt`. Setup ends with `slurm/pack_env.sh`, which packs the env into one file that jobs unpack onto node-local disk (imports straight from `$DATA` take minutes). **Re-run `pack_env.sh` after every change to the env**; jobs refuse to start on a stale pack.
2. **Smoke 1**: `sbatch slurm/smoke_tiny.sh`, ~2 min on `devel`. Tiny random model; tests env, GPU, offline cache, writing and syncing results, the checker.
3. **Smoke 2**: `sbatch slurm/smoke_llama.sh`, ~15 min. Real Llama-3-8B in bf16 on 1,024 sequences, all measures on one partition.
4. Look at both logs and `check_run.py` output (`OK` at the end). Only then:
5. **Full runs**: `sbatch slurm/extract.sh <name> <sequences file> [args]` per condition, then `sbatch --array=0-33 slurm/measure.sh <name>`.

6. **E1 completion** (Llama-3-8B: untrained seeds 0–2, FineWeb short/256, bootstrap). On an interactive node: `python scripts/prefetch.py --fineweb-shard sample/10BT/000_00000.parquet`; set N = median of `$THESIS_ROOT/runs/llama3-8b_normal/n_tokens.npy` minus 1 (BOS); `python scripts/make_sequences_fineweb.py --short-tokens N --out $THESIS_ROOT/sequences`. Smoke: `sbatch --time=00:30:00 slurm/extract.sh smoke_fw256 fineweb10bt_normal_n50000_len256_seed0.jsonl --limit 1024 --batch-size 32` and `sbatch slurm/bootstrap.sh llama3-8b_normal --B 5 --out bootstrap_smoke.jsonl`. Then `FW_SHORT=N bash slurm/submit_e1.sh` (login node) submits the whole matrix with dependencies.

Results land in `$DATA/thesis/runs/<name>`; pull the small `metrics*.jsonl` and `meta.json` files back to the Mac with rsync.

## Contents

- `src/layergeom/abide.py`: ABIDE intrinsic-dimension estimator (Di Noia et al., arXiv:2405.15132v6, Sci. Rep. 2026), with BIDE, 2NN start, Fisher-information CI and the Epps–Singleton goodness-of-fit check. The k* step is tested to match DADApy's `compute_kstar` exactly.
- `src/layergeom/data.py`, `extract.py`, `measures.py`, `runinfo.py`: pipeline modules (see docstrings for the choices made where Cheng et al. are silent).
- `notebooks/01_id_estimator_calibration.ipynb`: TwoNN / GRIDE / ABIDE on synthetic data of known dimension at n = 10k, D = 4096.
- `slurm/`: ARC job scripts.
- `tests/`: unit tests, including a tiny-Llama check that our hooks match HF `hidden_states` and that padding does not change last-token states.
