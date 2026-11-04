"""Provenance written next to every output: code commit, library versions, host, GPU."""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

PACKAGES = ["numpy", "scipy", "scikit-learn", "torch", "transformers", "datasets", "dadapy", "bitsandbytes"]


def git_state(repo: Path) -> dict:
    def run(*args):
        try:
            return subprocess.check_output(["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def versions() -> dict:
    out = {"python": sys.version.split()[0]}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    return out


def environment() -> dict:
    env = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    try:
        import torch

        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return env


def write_json(path: Path, obj: dict) -> None:
    """Atomic write: a half-written file is never left behind under the final name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)


def base_meta(args, repo: Path) -> dict:
    return {
        "args": vars(args),
        "git": git_state(repo),
        "versions": versions(),
        "env": environment(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": "running",
    }
