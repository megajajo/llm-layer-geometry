"""Adaptive Binomial Intrinsic Dimension Estimator (ABIDE).

Implements Di Noia, Macocco, Glielmo, Laio & Mira, "Scale adaptive and robust
intrinsic dimension estimation via optimal neighbourhood identification",
arXiv:2405.15132v6 (28 Jul 2026), Sci. Rep. 16:18559 (2026). Section and
equation numbers below refer to v6.

Idea: for each point, grow the neighbourhood until a likelihood-ratio test says
the density is no longer constant (k*_i, Eq. 4). Estimate the ID with the
binomial estimator (BIDE, Eq. 1) on those per-point neighbourhoods. Because the
test needs the ID to compute ball volumes, iterate from the 2NN estimate until
the ID converges (Alg. 1).

Conventions (checked against DADApy's `_compute_kstar`, main branch, accessed
2026-11-04): neighbour column 0 is the point itself; the first test uses k = 3;
k*_i is the first k whose test statistic exceeds the threshold, or k_max if
none does.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats
from sklearn.neighbors import NearestNeighbors

C_STAR = 0.2032  # optimal tau = C_STAR**(1/d) (§2.3, from Macocco et al. 2023)
K_MIN = 3  # first k tested, as in DADApy


@dataclass
class AbideResult:
    id: float
    se: float  # 1 / sqrt(n * I(d*)), Eq. 5
    kstar: np.ndarray  # per-point optimal neighbourhood size, computed with the final ID
    tau: float  # radius ratio used in the final BIDE step
    k_a: np.ndarray  # per-point inner-ball counts at the final step
    k_b: np.ndarray  # per-point outer-ball counts (k* - 1) at the final step
    id_2nn: float  # starting value
    trajectory: list[float] = field(default_factory=list)  # ID after each iteration
    converged: bool = False

    def ci(self, level: float = 0.95) -> tuple[float, float]:
        """Normal-approximation confidence interval (§2.3). Within-sample only."""
        z = stats.norm.ppf(0.5 + level / 2)
        return self.id - z * self.se, self.id + z * self.se


def neighbours(X: np.ndarray, k_max: int, n_jobs: int = -1) -> tuple[np.ndarray, np.ndarray]:
    """Sorted Euclidean distances and indices to the k_max nearest neighbours.

    Returns arrays of shape (n, k_max + 1); column 0 is the point itself.
    """
    n = X.shape[0]
    if k_max + 1 > n:
        raise ValueError(f"k_max={k_max} needs more than {k_max} points, got n={n}")
    nn = NearestNeighbors(n_neighbors=k_max + 1, n_jobs=n_jobs).fit(X)
    dist, idx = nn.kneighbors(X)
    return dist, idx


def id_2nn(dist: np.ndarray) -> float:
    """Maximum-likelihood 2NN estimate: n / sum log(r2 / r1) (§2.3)."""
    return dist.shape[0] / np.sum(np.log(dist[:, 2] / dist[:, 1]))


def compute_kstar(dist: np.ndarray, idx: np.ndarray, d: float, d_thr: float) -> np.ndarray:
    """Per-point optimal neighbourhood size k*_i (Eq. 4).

    For k = 3, ..., k_max - 1, compare the density in the k-NN ball of point i
    with that in the k-NN ball of i's (k+1)-th neighbour j. Volumes enter only
    through r^d (the unit-ball constant cancels), so the statistic is

        D_ik = -2k (x + y - 2 logaddexp(x, y) + log 4),
        x = d log r_{i,k},  y = d log r_{j,k}.
    """
    n, cols = dist.shape
    k_max = cols - 1
    ks = np.arange(K_MIN, k_max)  # tested k values
    with np.errstate(divide="ignore"):  # column 0 is the point itself
        log_r = np.log(dist)
    x = d * log_r[:, ks]  # (n, len(ks))
    j = idx[:, ks + 1]  # (k+1)-th neighbour of i
    y = d * log_r[j, ks[None, :]]
    D = -2.0 * ks * (x + y - 2.0 * np.logaddexp(x, y) + np.log(4.0))
    fails = D > d_thr
    first = np.argmax(fails, axis=1)
    return np.where(fails.any(axis=1), ks[first], k_max)


def binomial_counts(dist: np.ndarray, k: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Outer and inner ball counts for BIDE with per-point outer radius r_{i,k_i}.

    k_B = k_i - 1 points lie strictly inside the open ball of radius r_{i,k_i}
    (excluding i); k_A counts neighbours strictly closer than tau * r_{i,k_i}
    (Alg. 1, lines 6-9).
    """
    n = dist.shape[0]
    t_b = dist[np.arange(n), k]
    t_a = tau * t_b
    cols = np.arange(1, dist.shape[1])
    inside = (dist[:, 1:] < t_a[:, None]) & (cols[None, :] < k[:, None])
    return inside.sum(axis=1), k - 1


def bide(k_a: np.ndarray, k_b: np.ndarray, tau: float) -> float:
    """Binomial ID estimator, Eq. 1."""
    return float(np.log(k_a.sum() / k_b.sum()) / np.log(tau))


def fisher_se(d: float, tau: float, k_b: np.ndarray) -> float:
    """Standard error from the observed Fisher information, Eq. 5."""
    n = k_b.shape[0]
    p = tau**d
    info = np.log(tau) ** 2 * p * k_b.mean() / (1.0 - p)
    return float(1.0 / np.sqrt(n * info))


def abide(
    X: np.ndarray | None = None,
    *,
    dist: np.ndarray | None = None,
    idx: np.ndarray | None = None,
    k_max: int = 350,
    alpha: float = 0.01,
    max_iter: int = 5,
    tol: float = 1e-4,
    n_jobs: int = -1,
) -> AbideResult:
    """Estimate the intrinsic dimension with ABIDE (Alg. 1).

    Pass either the data X (n, D) or precomputed `dist`, `idx` from
    `neighbours`. Defaults follow §3: k_max = 350, alpha = 0.01
    (D_thr = 6.635), at most 5 iterations (the paper's practice, §2.3).
    Because k* is integer-valued the ID can oscillate at the ~1e-3 level, so
    a small `tol` may never be reached; `converged` records whether it was.

    Duplicate points must be removed beforehand (zero distances break the
    estimator; the paper's App. notes the same for discrete data).
    """
    if dist is None:
        if X is None:
            raise ValueError("pass X or (dist, idx)")
        dist, idx = neighbours(np.asarray(X, dtype=np.float64), k_max, n_jobs=n_jobs)
    if np.any(dist[:, 1:] <= 0):
        raise ValueError("zero neighbour distances: remove duplicate points first")

    d_thr = stats.chi2.ppf(1 - alpha, df=1)
    d0 = id_2nn(dist)
    d_cur = d0
    trajectory = []
    converged = False
    for _ in range(max_iter):
        tau = C_STAR ** (1.0 / d_cur)
        kstar = compute_kstar(dist, idx, d_cur, d_thr)
        k_a, k_b = binomial_counts(dist, kstar, tau)
        d_next = bide(k_a, k_b, tau)
        trajectory.append(d_next)
        if abs(d_cur - d_next) < tol:
            converged = True
            d_cur = d_next
            break
        d_cur = d_next

    d_final = d_cur
    tau = C_STAR ** (1.0 / d_final)
    kstar = compute_kstar(dist, idx, d_final, d_thr)  # Alg. 1, lines 18-20
    k_a, k_b = binomial_counts(dist, kstar, tau)
    return AbideResult(
        id=d_final,
        se=fisher_se(d_final, tau, k_b),
        kstar=kstar,
        tau=tau,
        k_a=k_a,
        k_b=k_b,
        id_2nn=d0,
        trajectory=trajectory,
        converged=converged,
    )


def gof_pvalue(result: AbideResult, n_draws: int = 100_000, seed: int = 0) -> float:
    """Model-validation p-value (§2.3): Epps-Singleton test of observed k_A
    against the binomial mixture implied by the k_B and the estimated ID.

    A relative measure for comparing fits, not a calibrated test (the paper's
    caveat): with large n it rejects for small departures.
    """
    rng = np.random.default_rng(seed)
    kb = rng.choice(result.k_b, size=n_draws, replace=True)
    sim = rng.binomial(kb, result.tau**result.id)
    return float(stats.epps_singleton_2samp(result.k_a, sim).pvalue)
