"""Extract last-token states at every site for one (model, init, sequences) run.

Output directory:
    meta.json            args, revisions, versions, host/GPU, checks, status
    states/<site>.npy    (n, D) float16, one file per site (layer_00..layer_L, final_norm)
    docs.npy             article index per sequence (-1 for shuffled), for grouped resampling
    n_tokens.npy         model tokens per sequence (incl. BOS)

`meta.json` gets status "complete" only after every array is flushed; anything
else means the run is unusable.

    python scripts/extract.py --sequences seqs.jsonl --out $SCRATCH/run --limit 512
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaConfig

from layergeom import data, extract, runinfo

REPO = Path(__file__).resolve().parents[1]
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

p = argparse.ArgumentParser()
p.add_argument("--model", default="meta-llama/Meta-Llama-3-8B")
p.add_argument("--init", choices=["pretrained", "random"], default="pretrained")
p.add_argument("--seed", type=int, default=0, help="init seed for --init random")
p.add_argument("--dtype", choices=["bf16", "fp16", "fp32", "int8"], default="bf16")
p.add_argument("--tiny", action="store_true", help="tiny random Llama (pipeline test only)")
p.add_argument("--sequences", type=Path, required=True)
p.add_argument("--limit", type=int, default=None, help="use only the first N sequences")
p.add_argument("--batch-size", type=int, default=128)
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--out", type=Path, required=True)
args = p.parse_args()

revs = json.loads((REPO / "configs" / "revisions.json").read_text())
revision = revs.get(args.model)
meta = runinfo.base_meta(args, REPO)
meta["model_revision"] = revision
runinfo.write_json(args.out / "meta.json", meta)
t0 = time.time()

seq_meta, seqs = data.read_jsonl(args.sequences)
seqs = seqs[: args.limit] if args.limit else seqs
meta["sequences_meta"] = seq_meta

tok = AutoTokenizer.from_pretrained(args.model, revision=revision)
tok.pad_token = tok.pad_token or tok.eos_token
tok.padding_side = "right"  # last real token is found from the mask; left padding would shift RoPE positions

torch.manual_seed(args.seed)
if args.tiny:
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=256, intermediate_size=512, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2)
    model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float32)
elif args.init == "random":
    cfg = AutoConfig.from_pretrained(args.model, revision=revision)
    model = AutoModelForCausalLM.from_config(cfg, torch_dtype=DTYPES[args.dtype])
elif args.dtype == "int8":
    from transformers import BitsAndBytesConfig

    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=revision, quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map="auto"
    )
else:
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=revision, torch_dtype=DTYPES[args.dtype])
if args.dtype != "int8":
    model.to(args.device)
model.eval()

rec = extract.LastTokenRecorder(model)
enc = tok([s.text for s in seqs], padding=True, return_tensors="pt")
ids, mask = enc["input_ids"], enc["attention_mask"]

# Sanity checks on the first sequences before the full pass. A padding bug gives errors
# of order 1; bf16 kernels differ between batch shapes at the ~1e-2 level.
tol = 1e-3 if (args.tiny or args.dtype == "fp32") else 0.1
meta["checks"] = {
    "padding_max_rel_err": extract.check_padding_invariance(model, rec, ids, mask, rtol=tol),
    "final_norm_rel_err": extract.check_final_norm(model, rec, ids, mask),
    "bos_first": bool((ids[:, 0] == tok.bos_token_id).all()) if tok.bos_token_id is not None else None,
}
print("checks:", meta["checks"], flush=True)

n, D = len(seqs), model.config.hidden_size
outs = extract.open_outputs(args.out / "states", rec.names, n, D)
for b in range(0, n, args.batch_size):
    states = extract.run_batch(model, rec, ids[b : b + args.batch_size], mask[b : b + args.batch_size])
    for name, arr in states.items():
        if not torch.isfinite(arr).all():
            raise RuntimeError(f"non-finite values at {name}, batch {b}")
        outs[name][b : b + len(arr)] = arr.numpy().astype(np.float16)
    print(f"{min(b + args.batch_size, n)}/{n}", flush=True)
for m in outs.values():
    m.flush()
np.save(args.out / "docs.npy", np.array([s.doc for s in seqs]))
np.save(args.out / "n_tokens.npy", mask.sum(dim=1).numpy())

fp16_max = max(float(np.abs(np.asarray(m)).max()) for m in outs.values())
meta.update(n=n, hidden_size=D, sites=rec.names, max_abs_value=fp16_max, seconds=round(time.time() - t0, 1),
            status="complete")
runinfo.write_json(args.out / "meta.json", meta)
print(f"done: {n} sequences, {len(rec.names)} sites, {meta['seconds']}s")
