"""Spread and ID measures per site, partition and preprocessing variant.

Partitions follow Cheng: contiguous blocks of the (already random-order) sequences.
Rows go to a JSONL file; a sidecar `<out>.meta.json` gets status "complete" at the end.

    python scripts/measure.py --run $DATA/thesis/runs/<run> --partitions 5 --partition-size 10000
    python scripts/measure.py --run ... --sites layer_10 layer_11 --out metrics_part3.jsonl  # array task
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from layergeom import measures, runinfo

REPO = Path(__file__).resolve().parents[1]

p = argparse.ArgumentParser()
p.add_argument("--run", type=Path, required=True)
p.add_argument("--sites", nargs="*", default=None, help="default: all sites in meta.json")
p.add_argument("--partitions", type=int, default=5)
p.add_argument("--partition-size", type=int, default=10_000)
p.add_argument("--variants", nargs="+", default=["raw", "unit", "no_outliers"])
p.add_argument("--outlier-factor", type=float, default=10.0)
p.add_argument("--no-id", action="store_true", help="spread measures only (fast)")
p.add_argument("--k-max", type=int, default=350)
p.add_argument("--gride-range", type=int, default=2**13)
p.add_argument("--out", type=Path, default=None)
args = p.parse_args()

run_meta = json.loads((args.run / "meta.json").read_text())
if run_meta.get("status") != "complete":
    raise SystemExit(f"extraction run is not complete: {args.run}")
sites = args.sites or run_meta["sites"]
n = run_meta["n"]
size = min(args.partition_size, n // args.partitions)
out = args.out or args.run / "metrics.jsonl"
if not out.is_absolute():
    out = args.run / out
meta = runinfo.base_meta(args, REPO)
meta.update(partition_size_used=size, extraction_git=run_meta["git"])
runinfo.write_json(out.with_suffix(".meta.json"), meta)
t0 = time.time()

with open(out, "w") as f:
    for site in sites:
        X_all = np.load(args.run / "states" / f"{site}.npy", mmap_mode="r")
        for part in range(args.partitions):
            X = np.asarray(X_all[part * size : (part + 1) * size])
            for variant in args.variants:
                Xv, info = measures.preprocess(X, variant, args.outlier_factor)
                base = {"site": site, "partition": part, "variant": variant, "n": int(Xv.shape[0]), **info}
                rows = [{"measure": k, "value": v} for k, v in measures.spread(measures.eigenvalues(Xv)).items()]
                if not args.no_id:
                    Xu, n_dup = measures.dedupe(Xv)
                    base["n_duplicates"] = n_dup
                    rows += measures.id_estimates(Xu, args.k_max, args.gride_range)
                for r in rows:
                    f.write(json.dumps({**base, **r}) + "\n")
                f.flush()
        print(f"{site} done ({time.time() - t0:.0f}s)", flush=True)

meta.update(status="complete", seconds=round(time.time() - t0, 1))
runinfo.write_json(out.with_suffix(".meta.json"), meta)
