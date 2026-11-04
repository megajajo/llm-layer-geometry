"""Validate a finished run directory before launching anything bigger.

Checks: status flags, array shapes and dtypes, no inf/NaN, no all-zero or
constant sites, sane number of duplicates, metrics file covering every
site x partition x variant. Exits non-zero on any failure.

    python scripts/check_run.py $DATA/thesis/runs/<run>
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

run = Path(sys.argv[1])
errors, notes = [], []

meta = json.loads((run / "meta.json").read_text())
if meta.get("status") != "complete":
    errors.append(f"extraction status = {meta.get('status')}")
n, D, sites = meta.get("n"), meta.get("hidden_size"), meta.get("sites", [])
notes.append(f"n={n} D={D} sites={len(sites)} gpu={meta['env'].get('gpu')} git={meta['git']}")
notes.append(f"checks={meta.get('checks')}")

for site in sites:
    path = run / "states" / f"{site}.npy"
    if not path.exists():
        errors.append(f"missing {path.name}")
        continue
    X = np.load(path, mmap_mode="r")
    if X.shape != (n, D) or X.dtype != np.float16:
        errors.append(f"{site}: shape {X.shape} dtype {X.dtype}")
    Xf = np.asarray(X, dtype=np.float32)
    if not np.isfinite(Xf).all():
        errors.append(f"{site}: non-finite values (fp16 overflow?)")
    if (np.abs(Xf).sum(axis=1) == 0).any():
        errors.append(f"{site}: all-zero rows (unfilled batch?)")
    n_unique = len(np.unique(X, axis=0))
    if site == "layer_00":
        # token embeddings: sequences ending in the same token are identical by construction
        notes.append(f"{site}: {n_unique}/{n} unique (= distinct final tokens)")
    elif n_unique < 0.99 * n:
        errors.append(f"{site}: only {n_unique}/{n} unique states")
    notes.append(f"{site}: mean norm {np.linalg.norm(Xf, axis=1).mean():.1f}, max |x| {np.abs(Xf).max():.1f}")

for docs_file in ("docs.npy", "n_tokens.npy"):
    if not (run / docs_file).exists():
        errors.append(f"missing {docs_file}")

for mfile in sorted(run.glob("metrics*.jsonl")):
    mmeta = json.loads(mfile.with_suffix(".meta.json").read_text())
    if mmeta.get("status") != "complete":
        errors.append(f"{mfile.name}: status {mmeta.get('status')}")
    rows = [json.loads(line) for line in mfile.read_text().splitlines()]
    bad = [r for r in rows if r["value"] is None or not np.isfinite(r["value"])]
    if bad:
        errors.append(f"{mfile.name}: {len(bad)} non-finite values, e.g. {bad[0]}")
    cover = Counter((r["site"], r["partition"], r["variant"]) for r in rows)
    expected = len(mmeta["args"]["sites"] or sites) * mmeta["args"]["partitions"] * len(mmeta["args"]["variants"])
    if len(cover) != expected:
        errors.append(f"{mfile.name}: {len(cover)} site/partition/variant cells, expected {expected}")
    notes.append(f"{mfile.name}: {len(rows)} rows, {len(cover)} cells, {mmeta.get('seconds')}s")

print("\n".join(notes))
if errors:
    print("\nFAILED:\n  " + "\n  ".join(errors))
    sys.exit(1)
print("\nOK")
