import numpy as np

from layergeom import competitor as cp


def brute(X, Y, E=None):
    n = len(X)
    out = {f"{s}_{m}": np.zeros(n) for s in ("A", "own", "other") for m in cp.METRICS}
    for i in range(n):
        P = np.eye(X.shape[1]) if E is None else np.eye(X.shape[1]) - np.outer(E[i], E[i])
        x, ys = P @ X[i], Y @ P.T
        R = np.linalg.norm(ys[i])
        d = {"cos": np.array([1 - x @ y / (np.linalg.norm(x) * np.linalg.norm(y)) for y in ys]),
             "euc": np.linalg.norm(ys - x, axis=1) / R}
        for m in cp.METRICS:
            others = np.delete(d[m], i)
            out[f"A_{m}"][i] = (others < d[m][i]).mean()
            out[f"own_{m}"][i] = d[m][i]
            out[f"other_{m}"][i] = others.mean()
    return out


def test_bank_stats_match_brute_force_with_and_without_projection():
    rng = np.random.default_rng(0)
    X, Y = rng.standard_normal((40, 16)), rng.standard_normal((40, 16)) + 0.5
    E = rng.standard_normal((40, 16))
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    for e in (None, E):
        got, ref = cp.bank_stats(X, Y, e), brute(X, Y, e)
        for k in ref:
            np.testing.assert_allclose(got[k], ref[k], atol=1e-10, err_msg=k)


def test_endpoint_layer_has_no_competitors():
    Y = np.random.default_rng(1).standard_normal((30, 8))
    r = cp.bank_stats(Y, Y)
    for m in cp.METRICS:
        assert (r[f"A_{m}"] == 0).all() and np.allclose(r[f"own_{m}"], 0)


def test_straight_path_gives_nested_competitor_sets():
    # App. A.1: monotone straight-line convergence to the own endpoint never adds competitors
    rng = np.random.default_rng(2)
    X0, Y = rng.standard_normal((50, 12)), rng.standard_normal((50, 12))
    prev = None
    for t in np.linspace(0, 1, 12):
        r = cp.bank_stats((1 - t) * X0 + t * Y, Y)
        if prev is not None:
            for m in cp.METRICS:
                enter, _ = cp.turnover(prev[f"C_{m}"], r[f"C_{m}"])
                assert (enter == 0).all(), m
        prev = r


def test_restricted_banks_match_full_bank_and_token_sampling():
    rng = np.random.default_rng(3)
    X, Y = rng.standard_normal((20, 6)), rng.standard_normal((20, 6))
    full = cp.bank_stats(X, Y)
    banks = np.stack([np.delete(np.arange(20), i) for i in range(20)])
    r = cp.restricted_stats(X, Y, banks)
    for k in r:
        np.testing.assert_allclose(r[k], full[k], atol=1e-12, err_msg=k)

    final = np.repeat(np.arange(8), [50, 45, 41, 40, 40, 39, 3, 1])
    idx, toks = cp.same_token_sample(final, n_tokens=5, per_token=40, seed=0)
    assert len(idx) == 200 and len(set(idx)) == 200 and (final[idx] == toks).all()
    same, diff = cp.matched_banks(toks, seed=0)
    assert same.shape == diff.shape == (200, 39)
    rows = np.arange(200)[:, None]
    assert (toks[same] == toks[:, None]).all() and (same != rows).all()
    assert (toks[diff] != toks[:, None]).all()


def test_outlier_dims_union():
    A = np.ones((10, 6)); A[:, 2] = 50
    B = np.ones((10, 6)); B[:, 4] = 50
    assert cp.outlier_dims(A, B).tolist() == [2, 4]
