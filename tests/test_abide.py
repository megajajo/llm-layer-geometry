import numpy as np
import pytest
from scipy import stats

from layergeom import abide as ab


def gaussian(n, d, D, noise, seed=0):
    """§3.1 setting: d-dim standard Gaussian in D dims plus isotropic noise."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, D))
    X[:, :d] = rng.standard_normal((n, d))
    return X + noise * rng.standard_normal((n, D))


def kstar_loop(dist, idx, d, d_thr):
    """Direct port of DADApy `_compute_kstar` (per-point while loop)."""
    n, maxk = dist.shape
    out = np.empty(n, dtype=int)
    for i in range(n):
        j = 4
        while j < maxk:
            ksel = j - 1
            vvi = dist[i, ksel] ** d
            vvj = dist[idx[i, j], ksel] ** d
            dl = -2.0 * ksel * (np.log(vvi) + np.log(vvj) - 2.0 * np.log(vvi + vvj) + np.log(4))
            if dl > d_thr:
                break
            j += 1
        out[i] = j - 1
    return out


def test_kstar_matches_reference_loop():
    X = gaussian(800, 3, 20, 1e-3)
    dist, idx = ab.neighbours(X, k_max=60)
    d_thr = stats.chi2.ppf(0.99, df=1)
    for d in (2.0, 3.0, 7.5):
        np.testing.assert_array_equal(
            ab.compute_kstar(dist, idx, d, d_thr), kstar_loop(dist, idx, d, d_thr)
        )


@pytest.mark.parametrize("d", [2, 5])
def test_recovers_clean_gaussian(d):
    res = ab.abide(gaussian(5000, d, 100, 1e-4))
    assert abs(res.id - d) < 0.15 * d
    assert abs(res.trajectory[-1] - res.trajectory[-2]) < 0.01


def test_more_robust_to_noise_than_2nn():
    # §3.1 / Fig. 2: with noise, 2NN overestimates; ABIDE should be closer to the truth.
    res = ab.abide(gaussian(5000, 2, 100, 3e-2))
    assert res.id_2nn > 2.5
    assert abs(res.id - 2) < abs(res.id_2nn - 2)


def test_ci_and_counts_consistent():
    res = ab.abide(gaussian(3000, 4, 50, 1e-3))
    lo, hi = res.ci()
    assert lo < res.id < hi
    assert np.all(res.k_a <= res.k_b)
    assert np.all(res.k_b == res.kstar - 1)
    assert 0.0 <= ab.gof_pvalue(res, n_draws=20_000) <= 1.0


def test_duplicates_rejected():
    X = gaussian(500, 2, 10, 1e-3)
    X[1] = X[0]
    with pytest.raises(ValueError, match="duplicate"):
        ab.abide(X, k_max=50)


def test_matches_dadapy_kstar():
    dadapy = pytest.importorskip("dadapy")
    X = gaussian(1000, 3, 20, 1e-3, seed=1)
    data = dadapy.Data(X, maxk=101, verbose=False)
    data.compute_distances(maxk=101)
    data.set_id(3.0)
    data.compute_kstar(alpha=0.01)
    dist, idx = ab.neighbours(X, k_max=101)
    ours = ab.compute_kstar(dist, idx, 3.0, stats.chi2.ppf(0.99, df=1))
    np.testing.assert_array_equal(ours, data.kstar)
