# layergeom

Code for the thesis experiments on the layer-wise geometry of decoder-only LM hidden states. Research notes live in the parent `Thesis/` folder; this repository holds only code, configs and small result summaries. Hidden states and model weights live on ARC `$DATA` and are never committed.

## Setup

```bash
uv venv --python 3.11
uv pip install -e ".[dev]"
# DADApy for cross-checks: the PyPI sdist fails to build against NumPy 2,
# so install from GitHub (pin a commit for experiment runs)
uv pip install "dadapy @ git+https://github.com/sissa-data-science/DADApy@main"
pytest
```

On ARC, follow `../ARC.md` (conda env in `$DATA`, `pip install -e ".[extract]"`).

## Contents

- `src/layergeom/abide.py`: ABIDE intrinsic-dimension estimator (Di Noia et al., arXiv:2405.15132v6, Sci. Rep. 2026), with BIDE, 2NN start, Fisher-information CI and the Epps–Singleton goodness-of-fit check. The k* step is tested to match DADApy's `compute_kstar` exactly.
- `tests/`: unit tests on synthetic data with known intrinsic dimension.
