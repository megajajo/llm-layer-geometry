"""Download model and dataset into $HF_HOME and pin their revisions.

Run on an ARC *interactive* node (compute jobs run offline). Writes the resolved
commit hashes to configs/revisions.json; commit that file so every run uses them.

    python scripts/prefetch.py --model meta-llama/Meta-Llama-3-8B
    python scripts/prefetch.py --fineweb-shard sample/10BT/000_00000.parquet   # FineWeb only
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

REPO = Path(__file__).resolve().parents[1]

p = argparse.ArgumentParser()
p.add_argument("--model", default="meta-llama/Meta-Llama-3-8B")
p.add_argument("--dataset", default="Salesforce/wikitext")
p.add_argument("--dataset-config", default="wikitext-103-raw-v1")
p.add_argument("--fineweb-shard", default=None,
               help="download only this file of HuggingFaceFW/fineweb (e.g. sample/10BT/000_00000.parquet)")
args = p.parse_args()

api = HfApi()
path = REPO / "configs" / "revisions.json"
revs = json.loads(path.read_text()) if path.exists() else {}

if args.fineweb_shard:
    fw_sha = api.dataset_info("HuggingFaceFW/fineweb").sha
    local = hf_hub_download("HuggingFaceFW/fineweb", args.fineweb_shard, repo_type="dataset", revision=fw_sha)
    print(f"fineweb {args.fineweb_shard} @ {fw_sha} -> {local}")
    revs["HuggingFaceFW/fineweb/sample-10BT"] = {"revision": fw_sha, "file": args.fineweb_shard}
    path.write_text(json.dumps(revs, indent=2) + "\n")
    print(f"wrote {path}")
    raise SystemExit(0)

model_sha = api.model_info(args.model).sha
snapshot_download(args.model, revision=model_sha, allow_patterns=["*.json", "*.safetensors", "tokenizer*"])
print(f"model {args.model} @ {model_sha}")

ds_sha = api.dataset_info(args.dataset).sha
from datasets import load_dataset  # noqa: E402

for split in ("train", "validation", "test"):
    load_dataset(args.dataset, args.dataset_config, split=split, revision=ds_sha)
print(f"dataset {args.dataset}/{args.dataset_config} @ {ds_sha}")

revs[args.model] = model_sha
revs[f"{args.dataset}/{args.dataset_config}"] = ds_sha
path.parent.mkdir(exist_ok=True)
path.write_text(json.dumps(revs, indent=2) + "\n")
print(f"wrote {path}")
