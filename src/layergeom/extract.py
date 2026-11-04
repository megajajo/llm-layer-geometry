"""Last-token residual states after every block, captured with our own hooks.

Sites saved per sequence (last non-padding token):
- `layer_00`: token embeddings (input to block 1);
- `layer_ii`: residual stream after block i, *before* the final norm (i = 1..L);
- `final_norm`: output of the model's final norm on the last block.

Cheng et al. use HF `hidden_states`, whose last entry is post-final-norm in Llama;
Mudarisov et al. use the pre-norm residual. Saving both covers each paper. We do
not rely on `output_hidden_states`, whose recording differs across transformers
versions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def decoder_parts(model):
    """(embedding module, list of decoder blocks, final norm) for Llama-like models."""
    base = model.model if hasattr(model, "model") else model
    return base.embed_tokens, list(base.layers), base.norm


class LastTokenRecorder:
    """Forward hooks that keep only each sequence's last non-padding position."""

    def __init__(self, model):
        emb, layers, norm = decoder_parts(model)
        self.modules = [emb, *layers, norm]
        self.names = ["layer_00", *[f"layer_{i:02d}" for i in range(1, len(layers) + 1)], "final_norm"]
        self.last_idx: torch.Tensor | None = None
        self.out: dict[str, torch.Tensor] = {}
        self.handles = [m.register_forward_hook(self._hook(n)) for m, n in zip(self.modules, self.names)]

    def _hook(self, name):
        def fn(_module, _inp, output):
            h = output[0] if isinstance(output, tuple) else output
            rows = torch.arange(h.shape[0], device=h.device)
            self.out[name] = h[rows, self.last_idx.to(h.device)].float().cpu()

        return fn

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def run_batch(model, rec: LastTokenRecorder, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """Right-padded batch -> {site: (batch, D) float32 tensor}."""
    rec.last_idx = attention_mask.sum(dim=1) - 1
    rec.out = {}
    dev = next(model.parameters()).device
    base = model.model if hasattr(model, "model") else model  # skip the LM head: no logits needed
    base(input_ids=input_ids.to(dev), attention_mask=attention_mask.to(dev), use_cache=False)
    return dict(rec.out)


def open_outputs(out_dir: Path, names: list[str], n: int, d: int) -> dict[str, np.memmap]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        name: np.lib.format.open_memmap(out_dir / f"{name}.npy", mode="w+", dtype=np.float16, shape=(n, d))
        for name in names
    }


@torch.no_grad()
def check_padding_invariance(model, rec, input_ids, attention_mask, k: int = 4, rtol: float = 2e-2):
    """Batched (right-padded) states must match running sequences one at a time.

    Returns the worst relative error across sites; raises if above rtol.
    """
    batched = run_batch(model, rec, input_ids[:k], attention_mask[:k])
    worst = 0.0
    for i in range(k):
        L = int(attention_mask[i].sum())
        single = run_batch(model, rec, input_ids[i : i + 1, :L], attention_mask[i : i + 1, :L])
        for name in rec.names:
            a, b = batched[name][i], single[name][0]
            err = float((a - b).norm() / (b.norm() + 1e-12))
            worst = max(worst, err)
    if worst > rtol:
        raise RuntimeError(f"padding changes last-token states: max rel. error {worst:.3g} > {rtol}")
    return worst


@torch.no_grad()
def check_final_norm(model, rec, input_ids, attention_mask, rtol: float = 2e-2):
    """final_norm must equal norm(last block output): confirms the sites are what we think."""
    states = run_batch(model, rec, input_ids[:4], attention_mask[:4])
    _, layers, norm = decoder_parts(model)
    last = states[f"layer_{len(layers):02d}"]
    p = next(norm.parameters())
    renorm = norm(last.to(p.device, p.dtype)).float().cpu()
    err = float((renorm - states["final_norm"]).norm() / states["final_norm"].norm())
    if err > rtol:
        raise RuntimeError(f"final_norm != norm(last block): rel. error {err:.3g}")
    return err
