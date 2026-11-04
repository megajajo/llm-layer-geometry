"""Build Cheng-style 20-token WikiText-103 sequences (normal or corpus-shuffled).

Sequences are built once and shared by every model and init, so all conditions
see identical inputs.

    python scripts/make_sequences.py --n 50000 --seed 0 --out $DATA/thesis/sequences
"""

import argparse
import json
from pathlib import Path

from layergeom import data

REPO = Path(__file__).resolve().parents[1]
KEY = "Salesforce/wikitext/wikitext-103-raw-v1"

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=50_000)
p.add_argument("--length", type=int, default=20)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--split", default="train")
p.add_argument("--out", type=Path, required=True)
args = p.parse_args()

revs = json.loads((REPO / "configs" / "revisions.json").read_text())
lines = data.load_wikitext_lines(args.split, revision=revs[KEY])
vocab, docs = data.articles(lines)
print(f"{len(docs)} articles, {sum(map(len, docs))} tokens, vocab {len(vocab)}")

for shuffled in (False, True):
    cond = "shuffled" if shuffled else "normal"
    seqs = data.sample_windows(vocab, docs, args.n, args.length, seed=args.seed, shuffled=shuffled)
    path = args.out / f"wikitext103_{args.split}_{cond}_n{args.n}_len{args.length}_seed{args.seed}.jsonl"
    meta = {**vars(args), "out": str(args.out), "condition": cond, "dataset_revision": revs[KEY]}
    data.write_jsonl(seqs, path, meta)
    print(f"wrote {path}")
