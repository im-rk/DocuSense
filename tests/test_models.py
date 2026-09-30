"""Model code smoke tests (CPU, tiny data). These check the code is CORRECT; real training happens on Kaggle."""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from data.build_common import read_jsonl
from data.synth_invoices import build_synth, make_invoice
from models.classifier.architecture import DocClassifierResNet, PlainCNN
from models.tagger.architecture import DocTaggerNet, build_tagger, count_params
from models.tagger.dataset import TaggerDataset, collate
from models.tagger.labels import LABEL_ID as L, N_CLS
from models.tagger.loss import class_weights, partial_ce
from models.tagger.predict import build_inputs
from models.tagger.vocab import build_vocab


def test_synth_invoice_labels():
    im, words = make_invoice(seed=1)
    labs = {w["label"] for w in words}
    assert {L["VENDOR"], L["DATE"], L["INVOICE_NO"], L["ITEM_NAME"], L["ITEM_PRICE"], L["SUBTOTAL"], L["TAX"], L["TOTAL"]} <= labs
    assert im.mode == "L" and all(w["box"][2] > w["box"][0] for w in words)


@pytest.fixture(scope="module")
def synth_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    build_synth(str(d), n_train=16, n_val=6, audit=0)
    return str(d)


def test_tagger_shapes_params_and_padding_invariance():
    torch.manual_seed(0)
    m = build_tagger(dict(vocab=100)).eval()
    n = count_params(m)
    assert 1.5e6 < n < 2.6e6, n
    B, N, L_ = 2, 7, 24
    ch = torch.randint(2, 100, (B, N, L_)); ch[:, :, 15:] = 0
    cr, pos = torch.rand(B, N, 1, 32, 96), torch.rand(B, N, 10)
    pad = torch.zeros(B, N, dtype=torch.bool); pad[1, 5:] = True
    ch[pad] = 0
    with torch.no_grad():
        o = m(ch, cr, pos, pad)
        assert o.shape == (B, N, N_CLS)
        o2 = m(torch.cat([ch, torch.zeros(B, 3, L_, dtype=torch.long)], 1), torch.cat([cr, torch.zeros(B, 3, 1, 32, 96)], 1),
               torch.cat([pos, torch.zeros(B, 3, 10)], 1), torch.cat([pad, torch.ones(B, 3, dtype=torch.bool)], 1))[:, :N]
    assert torch.allclose(o[~pad], o2[~pad], atol=1e-5)            # extra padding must not change real outputs


@pytest.mark.parametrize("flags", [dict(use_text=False), dict(use_look=False), dict(use_pos=False), dict(use_lstm=False),
                                   dict(lstm_layers=1)])
def test_tagger_ablation_flags_run(flags):
    m = build_tagger(dict(vocab=50, **flags))
    out = m(torch.randint(1, 50, (1, 4, 24)), torch.rand(1, 4, 1, 32, 96), torch.rand(1, 4, 10), torch.zeros(1, 4, dtype=torch.bool))
    assert out.shape == (1, 4, N_CLS)


def test_partial_loss_semantics():
    C = N_CLS
    w = torch.ones(C)
    logits = torch.full((1, C), -10.0)
    logits[0, L["VENDOR"]] = 10.0                                    # model is sure this token is a VENDOR
    lab = torch.tensor([0])                                          # dataset says OTHER
    un_cord = torch.tensor([[i in (L["VENDOR"], L["DATE"], L["INVOICE_NO"]) for i in range(C)]])
    assert partial_ce(logits, lab, un_cord, w) < 1e-3                # CORD never labels vendors -> no penalty
    assert partial_ce(logits, lab, torch.zeros_like(un_cord), w) > 5  # naive CE punishes it
    lab2 = torch.tensor([L["TOTAL"]])                                # a real label is still ordinary CE
    assert partial_ce(logits, lab2, un_cord, w) > 5
    cw = class_weights([1000, 10, 0, 100, 50, 50, 10, 10, 10, 0])
    assert cw[1] > cw[0] and cw[2] == 1.0


def test_dataset_and_parity_with_serving(synth_dir):
    recs = read_jsonl(f"{synth_dir}/synth_train.jsonl")
    vocab = {ch: i + 2 for i, ch in enumerate(build_vocab((w["text"] for r in recs for w in r["words"]), 1))}
    ds = TaggerDataset(synth_dir, ["synth_train"], vocab, augment=False)
    item = ds[0]
    n = len(recs[0]["words"])
    assert item["chars"].shape == (n, 24) and item["crops"].shape == (n, 1, 32, 96) and item["pos"].shape == (n, 10)
    assert not item["unannot"].any()                                 # synthetic annotates every class
    # train/serve parity: rebuild the same page through the serving path and compare tensors
    im, words = make_invoice(seed=1234)                              # build_synth uses seed + i, i=0 -> 1234
    from data.reading_order import order_words
    ordered = order_words(words)
    chars, crops, pos = build_inputs(ordered, np.asarray(im), vocab)
    assert [w["text"] for w in ordered][:n] == [w["text"] for w in recs[0]["words"]]
    assert torch.equal(chars, item["chars"]) and torch.allclose(crops, item["crops"]) and torch.allclose(pos, item["pos"], atol=1e-3)
    aug = TaggerDataset(synth_dir, ["synth_train"], vocab, augment=True)[0]   # augmentation keeps shapes consistent
    assert aug["chars"].shape[0] == aug["crops"].shape[0] == aug["pos"].shape[0] == aug["labels"].shape[0]
    b = collate([ds[0], ds[1]])
    assert b["pad"].shape == b["labels"].shape and b["pad"].any() != (len(ds[0]["labels"]) == len(ds[1]["labels"]))


def test_tagger_training_end_to_end_cpu(synth_dir, tmp_path):
    from models.tagger.predict import load_tagger, tag_words
    from models.tagger.train import sanity_overfit, train_tagger
    acc = sanity_overfit(synth_dir.replace("\\", "/"), n_docs=4, steps=150) if False else None   # needs cord/sroie names
    out = str(tmp_path / "out")
    m = train_tagger(synth_dir, out, epochs=2, batch_size=4, num_workers=0, train_sets=("synth_train",),
                     val_sets=("synth_val",), test_sets=("synth_val",), synth_frac=0.5, patience=5, tag="t")
    assert m["params"] > 1e6 and "ALL_TEST" in m["test"] and 0.0 <= m["test"]["ALL_TEST"]["tokens"]["macro_f1"] <= 1.0
    model, vocab, _ = load_tagger(f"{out}/t_v1.pth")
    im, words = make_invoice(seed=99)
    raw = [{"text": w["text"], "box": w["box"]} for w in words]
    tagged = tag_words(model, vocab, np.asarray(im), raw)
    assert len(tagged) == len(words) and {"label", "label_name", "prob", "line"} <= set(tagged[0])
    # resume path: rerun with the same out dir must reuse the last checkpoint without error
    train_tagger(synth_dir, out, epochs=2, batch_size=4, num_workers=0, train_sets=("synth_train",),
                 val_sets=("synth_val",), test_sets=("synth_val",), synth_frac=0.5, tag="t")


def test_loss_decreases_when_overfitting(synth_dir):
    from models.tagger.train import _loss
    recs = read_jsonl(f"{synth_dir}/synth_train.jsonl")
    vocab = {ch: i + 2 for i, ch in enumerate(build_vocab((w["text"] for r in recs for w in r["words"]), 1))}
    ds = TaggerDataset(synth_dir, ["synth_train"], vocab)
    b = collate([ds[i] for i in range(2)])
    torch.manual_seed(0)
    m = build_tagger(dict(vocab=len(vocab) + 2, p=0.0))
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
    w = torch.ones(N_CLS)
    first = last = None
    for s in range(40):
        loss = _loss(m, b, w, True)
        first = first if first is not None else loss.item()
        opt.zero_grad(); loss.backward(); opt.step(); last = loss.item()
    assert last < 0.5 * first, (first, last)


def test_classifier_models(tmp_path):
    for cls in (DocClassifierResNet, PlainCNN):
        m = cls(5)
        assert m(torch.rand(2, 1, 256, 256)).shape == (2, 5)
    n = sum(p.numel() for p in DocClassifierResNet(5).parameters())
    assert 2.5e6 < n < 4.5e6, n


def test_classifier_training_end_to_end_cpu(tmp_path):
    from models.classifier.predict import classify, load_classifier
    from models.classifier.train import train_classifier
    d = tmp_path / "cls"; d.mkdir()
    rng = np.random.default_rng(0)
    for split, n in (("train", 40), ("val", 10), ("test", 10)):
        y = np.arange(n) % 5
        x = (rng.random((n, 96, 96)) * 60 + y[:, None, None] * 40).astype(np.uint8)   # class-dependent brightness
        np.save(d / f"cls_{split}_x.npy", x); np.save(d / f"cls_{split}_y.npy", y.astype(np.int64))
    json.dump({"mean": 0.5, "std": 0.25}, open(d / "cls_stats.json", "w"))
    out = str(tmp_path / "out")
    m = train_classifier(str(d), out, epochs=2, batch_size=8, augment=False, num_workers=0, tag="c")
    assert 0.0 <= m["test_acc"] <= 1.0 and m["params"] > 2e6
    model, meta = load_classifier(f"{out}/c_v1.pth")
    name, conf, probs = classify(model, meta, np.full((300, 200), 200, np.uint8))
    assert name in meta["classes"] and 0 <= conf <= 1 and abs(sum(probs.values()) - 1) < 1e-4
