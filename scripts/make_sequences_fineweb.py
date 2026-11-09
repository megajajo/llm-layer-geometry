"""Build FineWeb sample-10BT windows at two context lengths, normal and word-shuffled.

Same documents, two lengths: one window of `--long` model tokens per document, and a
short window = the last `--short-tokens` tokens of that same window, so the two differ
only in how much context precedes the same final token. Windows are stored as token ids
(no BOS; extract.py prepends it). The shuffled control permutes words across the sampled
documents (as Cheng et al. do for whole corpora), re-tokenizes, and cuts non-overlapping
windows from the resulting stream.

Choices where Mudarisov et al. are silent or differ (recorded in each file's meta):
one window per document (Mudarisov: several non-overlapping per document), documents
pre-filtered by FineWeb's GPT-2 `token_count`, then by model-token count.

    python scripts/make_sequences_fineweb.py --n 50000 --seed 0 --short-tokens 26 \
        --out $DATA/thesis/sequences
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from layergeom import data

REPO = Path(__file__).resolve().parents[1]
KEY = "HuggingFaceFW/fineweb/sample-10BT"

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=50_000)
p.add_argument("--long", type=int, default=256)
p.add_argument("--short-tokens", type=int, required=True,
               help="short window length in model tokens (median WikiText-20 n_tokens minus BOS)")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--model", default="meta-llama/Meta-Llama-3-8B", help="tokenizer")
p.add_argument("--parquet", type=Path, default=None, help="local shard (default: the pinned one in the HF cache)")
p.add_argument("--out", type=Path, required=True)
args = p.parse_args()

revs = json.loads((REPO / "configs" / "revisions.json").read_text())
fw = revs.get(KEY, {})
path = args.parquet or Path(hf_hub_download("HuggingFaceFW/fineweb", fw["file"], repo_type="dataset",
                                            revision=fw["revision"]))
table = pq.read_table(path, columns=["id", "text", "token_count"])
src_ids, texts, gpt2_len = (table[c].to_pylist() for c in ("id", "text", "token_count"))
print(f"{len(texts)} documents in {path.name}")

tok = AutoTokenizer.from_pretrained(args.model, revision=revs.get(args.model))
rng = np.random.default_rng(args.seed)
cand = rng.permutation(np.flatnonzero(np.array(gpt2_len) >= args.long))

# Tokenize candidates in random order until n documents have >= long model tokens
# and distinct long windows.
chosen, chosen_ids = [], []
for b in range(0, len(cand), 4096):
    batch = cand[b : b + 4096]
    enc = tok([texts[i] for i in batch], add_special_tokens=False)["input_ids"]
    for i, ids in zip(batch, enc):
        if len(ids) >= args.long:
            chosen.append(int(i))
            chosen_ids.append(ids)
    if len(chosen) >= int(1.05 * args.n):
        break
longs, shorts = data.fineweb_windows(chosen_ids, chosen, args.long, args.short_tokens, seed=args.seed)
seen, keep = set(), []
for k, (_, _, w) in enumerate(longs):
    if tuple(w) not in seen:
        seen.add(tuple(w))
        keep.append(k)
if len(keep) < args.n:
    raise SystemExit(f"only {len(keep)} distinct windows, need {args.n}")
keep = keep[: args.n]
docs_used = [chosen[k] for k in keep]
print(f"{len(docs_used)} documents used ({len(chosen) - len(keep)} spare or duplicate)")

# Shuffled control over the same documents' text
shuf_ids = tok(data.shuffle_words([texts[i] for i in docs_used], seed=args.seed),
               add_special_tokens=False)["input_ids"]
s_longs, s_shorts = data.shuffled_stream_windows(shuf_ids, args.n, args.long, args.short_tokens, seed=args.seed)


def to_seqs(windows, with_source):
    return [data.Sequence(text=tok.decode(w), doc=d, start=s, ids=w,
                          source_id=src_ids[d] if with_source else None) for d, s, w in windows]


base = {**vars(args), "out": str(args.out), "parquet": str(args.parquet) if args.parquet else None,
        "dataset": KEY, "dataset_revision": fw.get("revision"), "dataset_file": fw.get("file"),
        "tokenizer": args.model, "tokenizer_revision": revs.get(args.model), "unit": "model tokens, no BOS"}
for cond, lw, sw in (("normal", [longs[k] for k in keep], [shorts[k] for k in keep]),
                     ("shuffled", s_longs, s_shorts)):
    for length, windows in ((args.long, lw), (args.short_tokens, sw)):
        out = args.out / f"fineweb10bt_{cond}_n{args.n}_len{length}_seed{args.seed}.jsonl"
        meta = {**base, "condition": cond, "length": length}
        data.write_jsonl(to_seqs(windows, cond == "normal"), out, meta)
        print(f"wrote {out}")
