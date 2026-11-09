import numpy as np
import pytest

from layergeom import data, measures

LINES = [
    " = Article One = \n",
    " \n",
    " The cat sat on the mat @-@ like , as usual . \n",
    " = = Section = = \n",
    " It was 1 @,@ 000 years old ( roughly ) . \n",
    " = Article Two = \n",
    " Another short text with words in it . \n",
]


def test_articles_split_and_drop_headings():
    vocab, docs = data.articles(LINES)
    assert len(docs) == 2
    assert "Section" not in [vocab[t] for t in docs[0]] and vocab[docs[0][0]] == "The"
    assert vocab[docs[1][0]] == "Another"


def test_detokenize():
    assert data.detokenize("mat @-@ like , as usual .") == "mat-like, as usual."
    assert data.detokenize("1 @,@ 000 years ( roughly )") == "1,000 years (roughly)"


def test_windows_stay_inside_articles_and_are_distinct():
    vocab, docs = data.articles([" = A = ", " ".join(f"a{i}" for i in range(30)), " = B = ",
                                 " ".join(f"b{i}" for i in range(25))])
    seqs = data.sample_windows(vocab, docs, 15, length=5, seed=0)
    assert len({s.text for s in seqs}) == 15
    for s in seqs:
        words = s.text.split()
        assert len(words) == 5 and len({w[0] for w in words}) == 1


def test_shuffled_windows_mix_articles():
    vocab, docs = data.articles([" = A = ", " ".join(f"a{i}" for i in range(200)), " = B = ",
                                 " ".join(f"b{i}" for i in range(200))])
    seqs = data.sample_windows(vocab, docs, 20, length=10, seed=0, shuffled=True)
    assert all(s.doc == -1 for s in seqs)
    assert any(len({w[0] for w in s.text.split()}) == 2 for s in seqs)


def test_spread_known_spectrum():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20000, 8)) * np.array([1, 1, 1, 1, 0, 0, 0, 0])
    s = measures.spread(measures.eigenvalues(X))
    assert abs(s["participation_ratio"] - 4) < 0.1
    assert abs(s["eff_rank"] - 4) < 0.1


def test_extraction_sites_and_padding():
    torch = pytest.importorskip("torch")
    from transformers import AutoModelForCausalLM, LlamaConfig

    from layergeom import extract

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=100, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2)
    model = AutoModelForCausalLM.from_config(cfg).eval()
    rec = extract.LastTokenRecorder(model)
    assert rec.names == ["layer_00", "layer_01", "layer_02", "layer_03", "final_norm"]
    ids = torch.randint(1, 100, (6, 12))
    mask = torch.ones_like(ids)
    for i, L in enumerate([12, 9, 5, 12, 7, 3]):
        mask[i, L:] = 0
        ids[i, L:] = 0
    assert extract.check_padding_invariance(model, rec, ids, mask, k=6, rtol=1e-4) < 1e-4
    assert extract.check_final_norm(model, rec, ids, mask, rtol=1e-5) < 1e-5
    # our layer_ii must equal HF hidden_states[i] for i < L (the pre-norm entries)
    states = extract.run_batch(model, rec, ids, mask)
    hs = model.model(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states
    last = mask.sum(1) - 1
    for i in range(3):
        ref = hs[i][torch.arange(6), last]
        torch.testing.assert_close(states[f"layer_{i:02d}"], ref)


def test_ids_round_trip(tmp_path):
    seqs = [data.Sequence("a b", 3, 7, ids=[5, 6, 7], source_id="x"), data.Sequence("c", 4, 0, ids=[8])]
    data.write_jsonl(seqs, tmp_path / "s.jsonl", {"k": 1})
    meta, back = data.read_jsonl(tmp_path / "s.jsonl")
    assert meta == {"k": 1} and back == seqs


def test_fineweb_short_is_suffix_of_long_one_per_doc():
    rng = np.random.default_rng(0)
    docs = [list(rng.integers(0, 1000, L)) for L in (40, 55, 300)]
    longs, shorts = data.fineweb_windows(docs, [10, 11, 12], long=32, short=6, seed=0)
    assert [d for d, _, _ in longs] == [10, 11, 12] == [d for d, _, _ in shorts]
    for (d, s, w), (_, ss, sw), ids in zip(longs, shorts, docs):
        assert len(w) == 32 and w == list(ids[s : s + 32])
        assert sw == w[-6:] and ss == s + 26


def test_shuffled_stream_windows_mix_documents():
    texts = [" ".join(f"a{i}" for i in range(200)), " ".join(f"b{i}" for i in range(200))]
    shuf = data.shuffle_words(texts, seed=0)
    assert sorted(" ".join(shuf).split()) == sorted(" ".join(texts).split())
    assert any(w.startswith("b") for w in shuf[0].split())
    toks = [[int(w[1:]) + (1000 if w[0] == "b" else 0) for w in t.split()] for t in shuf]
    longs, shorts = data.shuffled_stream_windows(toks, 10, long=32, short=5, seed=0)
    starts = sorted(s for _, s, _ in longs)
    assert all(d == -1 for d, _, _ in longs) and all(b - a >= 32 for a, b in zip(starts, starts[1:]))
    assert all(sw == w[-5:] for (_, _, w), (_, _, sw) in zip(longs, shorts))
    assert any(len({t >= 1000 for t in w}) == 2 for _, _, w in longs)


def test_weighted_eigenvalues_match_and_duplicate():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 12)) * np.linspace(3, 0.1, 12)
    Xt = torch.tensor(X - X.mean(0))
    np.testing.assert_allclose(measures.weighted_eigenvalues(Xt, torch.ones(500, dtype=torch.float64)),
                               measures.eigenvalues(X), rtol=1e-10, atol=1e-12)
    w = torch.tensor(rng.integers(0, 3, 500), dtype=torch.float64)  # weight k = row repeated k times
    Xrep = np.repeat(X, w.numpy().astype(int), axis=0)
    np.testing.assert_allclose(measures.weighted_eigenvalues(Xt, w), measures.eigenvalues(Xrep),
                               rtol=1e-8, atol=1e-10)
