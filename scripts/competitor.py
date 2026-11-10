"""E2: endpoint-bank competitor geometry for one extracted run (Mudarisov et al. reimplementation).

The run's contexts are split into disjoint banks of `--bank-size` (their setup: 1,024),
each analysed as in the paper; every context is a query within its own bank. Endpoint =
`layer_<L>` (last block, before the final norm); trajectory = `layer_01` .. `layer_<L>`.

Output directory `<run>/competitor/`:
    <variant>.npz   per (layer, context) arrays: A_, own_, other_, enter_, exit_ for cos / euc,
                    plus `context` (row index into the run) and `bank`
                    variants: raw, centred, centred_no_outliers, token_projection (App. F.3)
    same_token.npz  App. F.2 control: 50 final tokens x 40 contexts, same-token vs
                    size-matched different-token banks of 39 (raw states)
    meta.json       args, choices, dropped dimensions per layer, status "complete" at the end

    python scripts/competitor.py --run $DATA/thesis/runs/llama3-8b_fw256_normal
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from layergeom import competitor as cp
from layergeom import data, runinfo

REPO = Path(__file__).resolve().parents[1]
VARIANTS = ["raw", "centred", "centred_no_outliers", "token_projection"]

p = argparse.ArgumentParser()
p.add_argument("--run", type=Path, required=True)
p.add_argument("--bank-size", type=int, default=1024)
p.add_argument("--banks", type=int, default=None, help="default: as many disjoint banks as fit")
p.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
p.add_argument("--outlier-factor", type=float, default=10.0)
p.add_argument("--sequences", type=Path, default=None, help="for final tokens (default: the run's own file)")
p.add_argument("--same-token", nargs=2, type=int, default=[50, 40], metavar=("TOKENS", "PER_TOKEN"))
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", type=Path, default=Path("competitor"))
args = p.parse_args()

run_meta = json.loads((args.run / "meta.json").read_text())
if run_meta.get("status") != "complete":
    raise SystemExit(f"extraction run is not complete: {args.run}")
sites = [s for s in run_meta["sites"] if s.startswith("layer_") and s != "layer_00"]
L = len(sites)
n_banks = args.banks or run_meta["n"] // args.bank_size
ctx = np.arange(n_banks * args.bank_size)           # sequences are already in random order
bank_of = ctx // args.bank_size
out = args.out if args.out.is_absolute() else args.run / args.out
out.mkdir(parents=True, exist_ok=True)
meta = runinfo.base_meta(args, REPO)
meta.update(extraction_git=run_meta["git"], endpoint=sites[-1], trajectory=[sites[0], sites[-1]],
            n_banks=n_banks, n_contexts=len(ctx), dropped_dims={})
runinfo.write_json(out / "meta.json", meta)
t0 = time.time()


def load(site, rows=ctx):
    return np.asarray(np.load(args.run / "states" / f"{site}.npy", mmap_mode="r")[rows], dtype=np.float32)


Y_all = load(sites[-1])
E_all = load("layer_00")
E_all /= np.linalg.norm(E_all, axis=1, keepdims=True)
stats = {v: {f"{s}_{m}": np.full((L, len(ctx)), np.nan, np.float32)
             for s in ("A", "own", "other", "enter", "exit") for m in cp.METRICS} for v in args.variants}
prev = {}                                             # (variant, metric, bank) -> competitor mask

for li, site in enumerate(sites):
    X_all = Y_all if site == sites[-1] else load(site)
    for v in args.variants:
        X, Y, E = X_all, Y_all, None
        if v in ("centred", "centred_no_outliers"):
            X, Y = X - X.mean(0), Y - Y.mean(0)
            if v == "centred_no_outliers":
                drop = cp.outlier_dims(X_all, Y_all, factor=args.outlier_factor)
                meta["dropped_dims"][site] = drop.tolist()
                X, Y = np.delete(X, drop, axis=1), np.delete(Y, drop, axis=1)
        if v == "token_projection":
            E = E_all
        for b in range(n_banks):
            sl = slice(b * args.bank_size, (b + 1) * args.bank_size)
            r = cp.bank_stats(X[sl], Y[sl], None if E is None else E[sl])
            for m in cp.METRICS:
                for s in ("A", "own", "other"):
                    stats[v][f"{s}_{m}"][li, sl] = r[f"{s}_{m}"]
                if (v, m, b) in prev:
                    stats[v][f"enter_{m}"][li, sl], stats[v][f"exit_{m}"][li, sl] = cp.turnover(prev[v, m, b], r[f"C_{m}"])
                prev[v, m, b] = r[f"C_{m}"]
    print(f"{site} done ({time.time() - t0:.0f}s)", flush=True)

for v in args.variants:
    np.savez_compressed(out / f"{v}.npz", context=ctx, bank=bank_of, sites=np.array(sites), **stats[v])

# App. F.2: same-final-token vs size-matched different-token banks (raw states)
seq_path = args.sequences or Path(run_meta["args"]["sequences"])
_, seqs = data.read_jsonl(seq_path)
if seqs[0].ids is None:
    raise SystemExit(f"{seq_path} has no token ids; the same-token control needs them")
final = np.array([s.ids[-1] for s in seqs[: run_meta["n"]]])
n_tok, per_tok = args.same_token
idx, toks = cp.same_token_sample(final, n_tok, per_tok, seed=args.seed)
same, diff = cp.matched_banks(toks, seed=args.seed)
Y = load(sites[-1], idx)
st = {f"{s}_{bk}_{m}": np.full((L, len(idx)), np.nan, np.float32)
      for s in ("A", "own", "other") for bk in ("same", "diff") for m in cp.METRICS}
for li, site in enumerate(sites):
    X = load(site, idx)
    for bk, banks in (("same", same), ("diff", diff)):
        r = cp.restricted_stats(X, Y, banks)
        for k, val in r.items():
            s, m = k.split("_")
            st[f"{s}_{bk}_{m}"][li] = val
np.savez_compressed(out / "same_token.npz", context=idx, token=toks, sites=np.array(sites), **st)
meta.update(same_token={"tokens": n_tok, "per_token": per_tok, "sequences": str(seq_path)},
            status="complete", seconds=round(time.time() - t0, 1))
runinfo.write_json(out / "meta.json", meta)
print(f"done in {meta['seconds']}s")
