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
