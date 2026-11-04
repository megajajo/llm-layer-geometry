"""Cheng-style input sequences from WikiText-103.

Cheng et al. (arXiv:2405.15471v4, §3.2) sample 50k distinct 20-token sequences per
corpus, counting "typographic tokens", with no constraint to end at a sentence
boundary; shuffled controls shuffle the token order of the whole corpus first.

Our choices where the paper is silent (record them with every run):
- split: `train` of `wikitext-103-raw-v1`;
- typographic tokens = WikiText's own whitespace-separated tokens (punctuation and
  `@-@` markers are separate tokens), counted before detokenization;
- windows never cross article boundaries; heading lines (` = ... = `) are dropped;
- each window is detokenized (Moses-style spacing undone) before the model sees it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_TITLE = re.compile(r"^ = [^=].* = $")
_HEADING = re.compile(r"^ (= )+.*( =)+ $")


@dataclass
class Sequence:
    text: str
    doc: int  # article index, -1 for shuffled sequences
    start: int  # token offset within the article (or the shuffled corpus)


def articles(lines) -> tuple[list[str], list[np.ndarray]]:
    """Group WikiText lines into articles, dropping headings.

    Returns (vocab, docs): each article is an int32 array of token ids into vocab,
    so the ~100M-token corpus can be shuffled and sliced cheaply.
    """
    ids: dict[str, int] = {}
    docs: list[list[int]] = []
    for line in lines:
        if _TITLE.match(line):
            docs.append([])
            continue
        if not line.strip() or _HEADING.match(line):
            continue
        if not docs:
            docs.append([])
        docs[-1].extend(ids.setdefault(t, len(ids)) for t in line.split())
    vocab = list(ids)
    return vocab, [np.array(d, dtype=np.int32) for d in docs if d]


def detokenize(s: str) -> str:
    """Undo WikiText's Moses-style spacing (after lm-evaluation-harness's wikitext detokenizer)."""
    s = s.replace("s '", "s'")
    s = re.sub(r"/' [0-9]/", r"/'[0-9]/", s)
    s = s.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    s = s.replace(" : ", ": ").replace(" ; ", "; ").replace(" . ", ". ")
    s = s.replace(" ! ", "! ").replace(" ? ", "? ").replace(" , ", ", ")
    s = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", s)
    s = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", s)
    s = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", s)
    s = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', s)
    s = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", s)
    s = s.replace("= = = =", "====").replace("= = =", "===").replace("= =", "==")
    s = s.replace(" " + chr(176) + " ", chr(176)).replace(" \n", "\n").replace("\n ", "\n")
    s = s.replace(" N ", " 1 ").replace(" 's", "'s")
    # trailing punctuation at the end of a window has no following space
    s = re.sub(r" ([.,;:!?])$", r"\1", s)
    return s.strip()


def sample_windows(
    vocab: list[str], docs: list[np.ndarray], n: int, length: int = 20, seed: int = 0, shuffled: bool = False
) -> list[Sequence]:
    """n distinct windows of `length` tokens, uniform over valid start positions."""
    rng = np.random.default_rng(seed)
    if shuffled:
        docs = [rng.permutation(np.concatenate(docs))]
    starts = np.array([max(len(d) - length + 1, 0) for d in docs])
    cum = np.cumsum(starts)
    total = int(cum[-1])
    out: list[Sequence] = []
    seen: set[str] = set()
    while len(out) < n:
        pos = rng.integers(0, total, size=2 * (n - len(out)) + 100)
        for p in pos:
            di = int(np.searchsorted(cum, p, side="right"))
            off = int(p - (cum[di - 1] if di else 0))
            text = detokenize(" ".join(vocab[t] for t in docs[di][off : off + length]))
            if text in seen:
                continue
            seen.add(text)
            out.append(Sequence(text, -1 if shuffled else di, off))
            if len(out) == n:
                break
    return out


def load_wikitext_lines(split: str = "train", revision: str | None = None) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split, revision=revision)
    return ds["text"]


def write_jsonl(seqs: list[Sequence], path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(json.dumps({"meta": meta}) + "\n")
        for s in seqs:
            f.write(json.dumps({"text": s.text, "doc": s.doc, "start": s.start}) + "\n")


def read_jsonl(path: Path) -> tuple[dict, list[Sequence]]:
    with open(path) as f:
        meta = json.loads(f.readline())["meta"]
        seqs = [Sequence(**json.loads(line)) for line in f]
    return meta, seqs
