"""Download model and dataset into $HF_HOME and pin their revisions.

Run on an ARC *interactive* node (compute jobs run offline). Writes the resolved
commit hashes to configs/revisions.json; commit that file so every run uses them.

    python scripts/prefetch.py --model meta-llama/Meta-Llama-3-8B
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO = Path(__file__).resolve().parents[1]

p = argparse.ArgumentParser()
p.add_argument("--model", default="meta-llama/Meta-Llama-3-8B")
p.add_argument("--dataset", default="Salesforce/wikitext")
p.add_argument("--dataset-config", default="wikitext-103-raw-v1")
args = p.parse_args()

api = HfApi()
model_sha = api.model_info(args.model).sha
snapshot_download(args.model, revision=model_sha, allow_patterns=["*.json", "*.safetensors", "tokenizer*"])
print(f"model {args.model} @ {model_sha}")

ds_sha = api.dataset_info(args.dataset).sha
from datasets import load_dataset  # noqa: E402

for split in ("train", "validation", "test"):
    load_dataset(args.dataset, args.dataset_config, split=split, revision=ds_sha)
print(f"dataset {args.dataset}/{args.dataset_config} @ {ds_sha}")

path = REPO / "configs" / "revisions.json"
revs = json.loads(path.read_text()) if path.exists() else {}
revs[args.model] = model_sha
revs[f"{args.dataset}/{args.dataset_config}"] = ds_sha
path.parent.mkdir(exist_ok=True)
path.write_text(json.dumps(revs, indent=2) + "\n")
print(f"wrote {path}")
