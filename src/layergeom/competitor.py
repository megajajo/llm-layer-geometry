"""Endpoint-bank competitor geometry (Mudarisov, Burtsev & State, arXiv:2609.37824v1).

For a bank of contexts with states x[l, i] (last position, after block l, before the
final norm) and endpoints y[i] = x[L, i]:
- cosine distance d_C(x, y) = 1 - cos(x, y); Euclidean distance divided by the query's
  endpoint norm R_i = |y_i| (Eqs. 1, 37);
- competitor fraction A[l, i] = share of the other bank endpoints strictly closer to
  x[l, i] than its own endpoint (Eqs. 2, 42-43);
- own and mean-other distances, whose difference is the preference margin (Eq. 41);
- turnover: alternatives entering / leaving the competitor set from layer l-1 to l
  (Fig. 4, App. G), as fractions of the bank.

Our choices where the paper is silent (record them with every run):
- `centred`: subtract the mean over all contexts of the run, per layer, from the states
  and the endpoint mean from the endpoints;
- `centred_no_outliers`: additionally drop, per layer, the dimensions whose RMS exceeds
  `factor` x the median RMS in the states or the endpoints (E1's rule);
- token projection (App. F.3, Eq. 64): the query's final-token embedding direction is
  projected out of the query state, its own endpoint *and every alternative* (one linear
  map per query, so all its comparisons live in the same space).
"""

from __future__ import annotations

import numpy as np

METRICS = ("cos", "euc")


def bank_stats(X: np.ndarray, Y: np.ndarray, E: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Per-query statistics for one layer of one bank.

    X: (M+1, D) states, Y: (M+1, D) endpoints of the same contexts (row i is context i's
    own endpoint; every other row is an alternative). E: optional (M+1, D) unit
    final-token embedding directions to project out per query.
    Returns {f"{stat}_{metric}": (M+1,)} and the competitor masks {f"C_{metric}": (M+1, M+1) bool}.
    """
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    n = X.shape[0]
    G = X @ Y.T                                    # x_i . y_j
    xx = np.einsum("id,id->i", X, X)[:, None]       # |x_i|^2
    yy = np.einsum("jd,jd->j", Y, Y)[None, :]       # |y_j|^2
    if E is not None:
        E = E.astype(np.float64)
        xe = np.einsum("id,id->i", X, E)[:, None]   # x_i . e_i
        ye = Y @ E.T                                # ye[j, i] = y_j . e_i
        ye = ye.T                                   # (i, j)
        G = G - xe * ye                             # P_i x_i . P_i y_j
        xx = xx - xe**2
        yy = yy - ye**2                             # |P_i y_j|^2, now (i, j)
    yy = np.broadcast_to(yy, G.shape)
    own_norm = np.sqrt(np.diag(yy))[:, None]        # R_i (after projection, if any)
    d = {
        "cos": 1.0 - G / np.sqrt(xx * yy),
        "euc": np.sqrt(np.clip(xx + yy - 2.0 * G, 0.0, None)) / own_norm,
    }
    # own distance computed directly: the Gram form loses ~sqrt(eps) to cancellation near zero
    diff = X - Y
    dd = np.einsum("id,id->i", diff, diff)
    if E is not None:
        dd = dd - np.einsum("id,id->i", diff, E) ** 2
    np.fill_diagonal(d["euc"], np.sqrt(np.clip(dd, 0.0, None)) / own_norm[:, 0])
    off = ~np.eye(n, dtype=bool)
    out = {}
    for m, D in d.items():
        own = np.diag(D)
        C = (D < own[:, None]) & off
        out[f"A_{m}"] = C.sum(1) / (n - 1)
        out[f"own_{m}"] = own
        out[f"other_{m}"] = (D * off).sum(1) / (n - 1)
        out[f"C_{m}"] = C
    return out


def turnover(prev: np.ndarray, cur: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fractions of the bank entering and leaving the competitor set between two layers."""
    m = prev.shape[1] - 1
    return (cur & ~prev).sum(1) / m, (prev & ~cur).sum(1) / m


def outlier_dims(*arrays: np.ndarray, factor: float = 10.0) -> np.ndarray:
    """Union over arrays of dimensions whose RMS exceeds factor x the median RMS."""
    drop = set()
    for A in arrays:
        rms = np.sqrt(np.mean(np.asarray(A, dtype=np.float64) ** 2, axis=0))
        drop |= set(np.flatnonzero(rms > factor * np.median(rms)).tolist())
    return np.array(sorted(drop), dtype=int)


def same_token_sample(final_tokens: np.ndarray, n_tokens: int = 50, per_token: int = 40, seed: int = 0):
    """App. F.2: `per_token` contexts for each of the `n_tokens` most frequent final tokens.

    Returns (index array of sampled contexts, their token ids), grouped by token. Raises if
    fewer than `n_tokens` tokens have `per_token` contexts.
    """
    rng = np.random.default_rng(seed)
    toks, counts = np.unique(final_tokens, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    eligible = toks[order][counts[order] >= per_token][:n_tokens]
    if len(eligible) < n_tokens:
        raise ValueError(f"only {len(eligible)} final tokens with >= {per_token} contexts")
    idx = np.concatenate([rng.choice(np.flatnonzero(final_tokens == t), per_token, replace=False) for t in eligible])
    return idx, final_tokens[idx]


def matched_banks(tokens: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """For each query in a same-token sample: its same-token bank (all others with its token)
    and a size-matched different-token bank drawn once (fixed across layers)."""
    rng = np.random.default_rng(seed)
    n = len(tokens)
    same, diff = [], []
    for i in range(n):
        s = np.flatnonzero((tokens == tokens[i]) & (np.arange(n) != i))
        same.append(s)
        diff.append(rng.choice(np.flatnonzero(tokens != tokens[i]), len(s), replace=False))
    return np.stack(same), np.stack(diff)


def restricted_stats(X: np.ndarray, Y: np.ndarray, banks: np.ndarray) -> dict[str, np.ndarray]:
    """Competitor fraction and distances for query i against the endpoints banks[i] only."""
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    G = X @ Y.T                                   # all pairs in the sample, then restricted to each bank
    xx = np.einsum("id,id->i", X, X)[:, None]
    yy = np.einsum("jd,jd->j", Y, Y)[None, :]
    R = np.sqrt(yy).T
    D = {"cos": 1.0 - G / np.sqrt(xx * yy), "euc": np.sqrt(np.clip(xx + yy - 2.0 * G, 0.0, None)) / R}
    np.fill_diagonal(D["euc"], np.linalg.norm(X - Y, axis=1) / R[:, 0])   # exact own distance
    rows = np.arange(len(X))[:, None]
    out = {}
    for m in METRICS:
        own, alt = np.diag(D[m]), D[m][rows, banks]
        out[f"A_{m}"] = (alt < own[:, None]).mean(1)
        out[f"own_{m}"] = own
        out[f"other_{m}"] = alt.mean(1)
    return out
