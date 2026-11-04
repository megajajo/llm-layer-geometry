"""Per-layer spread and intrinsic-dimension measures on a cloud of states (n, D).

Spread measures use the covariance eigenvalues of centred states:
- participation ratio PR = (sum l)^2 / sum l^2;
- eigenvalue entropy H = -sum p log p with p = l / sum l, and its exponent exp(H)
  (the effective number of directions; Schulte & Ruegamer's variance-spread reading).

ID estimators: TwoNN and ABIDE (layergeom.abide), GRIDE (DADApy) across scales.

Preprocessing variants:
- `raw`;
- `unit`: every vector rescaled to unit length (removes norm growth);
- `no_outliers`: drop dimensions whose RMS exceeds `factor` x the median RMS
  (a provisional rule for massive-activation dimensions; the factor is a run parameter).
"""

from __future__ import annotations

import numpy as np

from layergeom import abide as ab


def unit_rows(X: np.ndarray) -> np.ndarray:
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def outlier_dims(X: np.ndarray, factor: float) -> np.ndarray:
    rms = np.sqrt(np.mean(X.astype(np.float64) ** 2, axis=0))
    return np.flatnonzero(rms > factor * np.median(rms))


def preprocess(X: np.ndarray, variant: str, outlier_factor: float = 10.0) -> tuple[np.ndarray, dict]:
    X = X.astype(np.float64)
    if variant == "raw":
        return X, {}
    if variant == "unit":
        return unit_rows(X), {}
    if variant == "no_outliers":
        drop = outlier_dims(X, outlier_factor)
        return np.delete(X, drop, axis=1), {"dropped_dims": drop.tolist()}
    raise ValueError(variant)


def eigenvalues(X: np.ndarray) -> np.ndarray:
    """Covariance eigenvalues of the centred cloud, descending (via the smaller Gram side)."""
    Xc = X - X.mean(axis=0)
    n, d = Xc.shape
    G = Xc.T @ Xc if d <= n else Xc @ Xc.T
    lam = np.linalg.eigvalsh(G / (n - 1))[::-1]
    return np.clip(lam, 0.0, None)


def spread(lam: np.ndarray) -> dict:
    p = lam / lam.sum()
    nz = p[p > 0]
    H = float(-(nz * np.log(nz)).sum())
    return {
        "participation_ratio": float(lam.sum() ** 2 / (lam**2).sum()),
        "eig_entropy": H,
        "eff_rank": float(np.exp(H)),
        "total_variance": float(lam.sum()),
    }


def dedupe(X: np.ndarray) -> tuple[np.ndarray, int]:
    U = np.unique(X, axis=0)
    return U, X.shape[0] - U.shape[0]


def id_estimates(X: np.ndarray, k_max: int = 350, gride_range: int = 2**13) -> list[dict]:
    """TwoNN, ABIDE and GRIDE rows for one (deduplicated) cloud."""
    from dadapy import Data

    n = X.shape[0]
    k_max = min(k_max, n - 2)
    dist, idx = ab.neighbours(X, k_max)
    r = ab.abide(dist=dist, idx=idx)
    rows = [
        {"measure": "twonn", "value": r.id_2nn},
        {
            "measure": "abide",
            "value": r.id,
            "se": r.se,
            "kstar_mean": float(r.kstar.mean()),
            "kstar_median": float(np.median(r.kstar)),
            "trajectory": r.trajectory,
        },
    ]
    rng_max = 2 ** int(np.floor(np.log2(min(gride_range, n - 1))))
    data = Data(X, maxk=rng_max, verbose=False)
    ids, errs, ranks = data.return_id_scaling_gride(range_max=rng_max, return_ranks=True)
    rows += [
        {"measure": "gride", "rank": int(k), "value": float(v), "se": float(e)}
        for v, e, k in zip(ids, errs, ranks)
    ]
    return rows
