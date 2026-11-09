"""Document bootstrap of the spread measures at every site, for peak-location intervals.

Each replicate draws documents with replacement (`docs.npy`; shuffled sequences, doc = -1,
are their own unit, since sequences from one article are correlated) and recomputes the
spread measures on the full resampled cloud, weighting each sequence by its document's
count. The same draws are used at every site, so each replicate gives a whole layer
profile and its own peak. b = 0 is the full sample. ID estimators are not bootstrapped:
duplicated points have zero neighbour distances (they keep the 5-partition intervals).

Rows go to `bootstrap_spread.jsonl` in the run; a sidecar `.meta.json` gets status
"complete" at the end, with a float64 CPU check of b = 0 at a few sites.

    python scripts/bootstrap_spread.py --run $DATA/thesis/runs/<run> --B 200
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from layergeom import measures, runinfo

REPO = Path(__file__).resolve().parents[1]

p = argparse.ArgumentParser()
p.add_argument("--run", type=Path, required=True)
p.add_argument("--B", type=int, default=200)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--variants", nargs="+", default=["raw", "unit"], choices=["raw", "unit"])
p.add_argument("--sites", nargs="*", default=None, help="default: all sites in meta.json")
p.add_argument("--check-sites", type=int, default=3, help="sites checked against float64 CPU spread")
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--out", type=Path, default=Path("bootstrap_spread.jsonl"))
args = p.parse_args()

run_meta = json.loads((args.run / "meta.json").read_text())
if run_meta.get("status") != "complete":
    raise SystemExit(f"extraction run is not complete: {args.run}")
sites = args.sites or run_meta["sites"]
out = args.out if args.out.is_absolute() else args.run / args.out
meta = runinfo.base_meta(args, REPO)
meta["extraction_git"] = run_meta["git"]
runinfo.write_json(out.with_suffix(".meta.json"), meta)
t0 = time.time()

# Resampling units: documents; doc = -1 sequences are units of their own.
docs = np.load(args.run / "docs.npy")
n = len(docs)
keys = np.where(docs >= 0, docs, -1 - np.arange(n))
_, unit = np.unique(keys, return_inverse=True)
G = int(unit.max()) + 1
rng = np.random.default_rng(args.seed)
counts = [np.ones(G, dtype=np.int64)] + [np.bincount(rng.integers(0, G, G), minlength=G) for _ in range(args.B)]
weights = torch.tensor(np.stack([c[unit] for c in counts]), dtype=torch.float32, device=args.device)  # (B+1, n)
meta.update(n=n, n_units=G)


checks = {}
check_at = set(sites[:: max(1, len(sites) // args.check_sites)][: args.check_sites])
with open(out, "w") as f:
    for site in sites:
        X_np = np.load(args.run / "states" / f"{site}.npy", mmap_mode="r")
        for variant in args.variants:
            X = torch.tensor(np.asarray(X_np, dtype=np.float32), device=args.device)
            if variant == "unit":
                X = X / X.norm(dim=1, keepdim=True)
            X = X - X.mean(dim=0)
            for b in range(args.B + 1):
                for k, v in measures.spread(measures.weighted_eigenvalues(X, weights[b])).items():
                    f.write(json.dumps({"site": site, "variant": variant, "b": b, "measure": k, "value": v}) + "\n")
                if b == 0 and site in check_at:
                    ref = measures.spread(measures.eigenvalues(measures.preprocess(np.asarray(X_np), variant)[0]))
                    got = measures.spread(measures.weighted_eigenvalues(X, weights[0]))
                    checks[f"{site}/{variant}"] = max(abs(got[k] / ref[k] - 1) for k in ref)
            f.flush()
            del X
        print(f"{site} done ({time.time() - t0:.0f}s)", flush=True)

meta.update(checks_max_rel_err=checks, status="complete", seconds=round(time.time() - t0, 1))
runinfo.write_json(out.with_suffix(".meta.json"), meta)
print(f"checks (max rel err vs float64 CPU): {checks}")
