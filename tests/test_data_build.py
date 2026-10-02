import json
import os

import numpy as np
from PIL import Image

from data.build_common import SplitBuilder, read_jsonl
from data.build_cord import cord_words
from data.build_sroie import build_sroie, label_sroie, parse_box_file, split_line
from models.tagger.labels import LABEL_ID as L, LABELS, annotated_classes, unannot_mask


def quad(x0, y0, x1, y1):
    return {"x1": x0, "y1": y0, "x2": x1, "y2": y0, "x3": x1, "y3": y1, "x4": x0, "y4": y1}


def test_cord_parse():
    gt = {"meta": {"image_size": {"width": 200, "height": 100}},
          "valid_line": [
              {"category": "menu.nm", "words": [{"quad": quad(10, 10, 60, 20), "text": "Nasi", "is_key": 0}]},
              {"category": "menu.price", "words": [{"quad": quad(150, 10, 190, 20), "text": "10.000", "is_key": 0}]},
              {"category": "total.total_price", "words": [{"quad": quad(10, 50, 60, 60), "text": "TOTAL", "is_key": 1},
                                                          {"quad": quad(150, 50, 190, 60), "text": "10.000", "is_key": 0}]},
              {"category": "menu.unitprice", "words": [{"quad": quad(100, 10, 120, 20), "text": "10.000"}]}]}
    sample = {"image": Image.new("RGB", (400, 200), "white"), "ground_truth": json.dumps(gt)}   # image is 2x annotation
    _, gray, words = cord_words(sample)
    assert gray.shape == (200, 400)
    got = [(w["text"], LABELS[w["label"]]) for w in words]
    assert got == [("Nasi", "ITEM_NAME"), ("10.000", "ITEM_PRICE"), ("TOTAL", "OTHER"), ("10.000", "TOTAL"), ("10.000", "OTHER")]
    assert words[0]["box"] == [20, 20, 120, 40]                                                  # scaled x2


def test_split_line_and_boxfile(tmp_path):
    parts = split_line("AB CDEF", [0, 0, 70, 10])
    assert [p[0] for p in parts] == ["AB", "CDEF"] and parts[0][1][2] == 20 and parts[1][1][0] == 30
    f = tmp_path / "x.txt"
    f.write_text("1,2,50,2,50,12,1,12,HELLO, WORLD\nbad line\n", encoding="utf-8")
    assert parse_box_file(str(f)) == [{"text": "HELLO, WORLD", "box": [1, 2, 50, 12]}]


def line(text, y, x0=0, x1=200):
    return {"text": text, "box": [x0, y, x1, y + 10]}


def test_label_sroie_ok():
    lines = [line("ACME TRADING SDN BHD", 0), line("DATE: 25/12/2018 10:30", 20), line("NASI GORENG 9.00", 40),
             line("TOTAL 9.00", 60), line("CASH 10.00", 80)]
    words, flags = label_sroie(lines, {"company": "ACME TRADING SDN BHD", "date": "25/12/2018", "total": "9.00"})
    assert all(flags.values())
    lab = {(w["text"], i): LABELS[w["label"]] for i, w in enumerate(words) if w["label"]}
    assert [v for v in lab.values()].count("VENDOR") == 4
    assert sum(1 for w in words if w["label"] == L["DATE"]) == 1
    tot = [w for w in words if w["label"] == L["TOTAL"]]
    assert len(tot) == 1 and tot[0]["box"][1] == 60                      # the TOTAL line, not the item line


def test_label_sroie_drops_when_key_missing():
    words, flags = label_sroie([line("HELLO", 0)], {"company": "ACME", "date": "25/12/2018", "total": "9.00"})
    assert words is None and not any(flags.values())


def test_build_sroie_end_to_end(tmp_path):
    root = tmp_path / "SROIE" / "train"
    (root / "img").mkdir(parents=True); (root / "box").mkdir(); (root / "entities").mkdir()
    Image.new("RGB", (300, 120), "white").save(root / "img" / "r1.jpg")
    (root / "box" / "r1.txt").write_text(
        "0,0,200,0,200,10,0,10,ACME TRADING\n0,20,200,20,200,30,0,30,DATE 25/12/2018\n0,40,200,40,200,50,0,50,TOTAL 9.00\n")
    (root / "entities" / "r1.txt").write_text(json.dumps({"company": "ACME TRADING", "date": "25/12/2018", "total": "9.00",
                                                          "address": "X"}))
    out = tmp_path / "out"
    res = build_sroie(str(out), str(tmp_path / "SROIE"), audit=1)
    assert res["coverage"] == 1.0
    recs = read_jsonl(str(out / "sroie_train.jsonl")) + read_jsonl(str(out / "sroie_val.jsonl"))
    assert len(recs) == 1 and recs[0]["source"] == "sroie"
    n = len(recs[0]["words"])
    crops = np.load(out / ("sroie_train_crops.npy" if (out / "sroie_train.jsonl").read_text().strip() else "sroie_val_crops.npy"))
    assert crops.shape == (n, 32, 96)


def test_unannot_masks():
    assert unannot_mask("cord")[L["VENDOR"]] and not unannot_mask("cord")[L["TOTAL"]]
    assert annotated_classes("sroie") == [L["VENDOR"], L["DATE"], L["TOTAL"]]
    assert not any(unannot_mask("synth"))


def test_rvl_class_mapping_is_by_name_not_id():
    from data.build_rvl_cache import ORIGINAL_NAMES, _targets
    alpha = ["Advertisement", "Budget", "Email", "File Folder", "Form", "Handwritten", "Invoice", "Letter", "Memo",
             "News Article", "Presentation", "Questionnaire", "Resume", "Scientific Publication", "Scientific Report",
             "Specification"]
    for names, inv_id in ((ORIGINAL_NAMES, 11), (alpha, 6)):        # same classes, different id orders
        t = _targets(names, 100)
        assert t[inv_id] == ("INVOICE", 100)
        assert sorted(v[0] for v in t.values()).count("OTHER") == 4 and all(v[1] == 25 for v in t.values() if v[0] == "OTHER")
        assert len(t) == 7                                          # invoice, form, letter + 4 OTHER sources


def test_label_sroie_vendor_fuzzy_and_extra_text():
    # vendor line has a registration number and an OCR typo; company name spans 2 lines
    lines = [line("GARDENIA BAKERIES (KL) SDN BHD (139386-X)", 0), line("LOT 3 JALAN PELABUR", 12),
             line("DATE 25/12/2018", 30), line("TOTAL 9.00", 60)]
    words, flags = label_sroie(lines, {"company": "GARDENIA BAKERIES (KL) SDN BHD", "date": "25/12/2018", "total": "9.00"})
    assert flags["vendor"] and words is not None
    vend = [w["text"] for w in words if w["label"] == L["VENDOR"]]
    assert vend == ["GARDENIA", "BAKERIES", "(KL)", "SDN", "BHD"]                    # not the registration number


def test_label_sroie_keeps_doc_without_vendor():
    lines = [line("SOMETHING ELSE", 0), line("DATE 25/12/2018", 20), line("TOTAL 9.00", 40)]
    words, flags = label_sroie(lines, {"company": "ACME TRADING SDN BHD", "date": "25/12/2018", "total": "9.00"})
    assert words is not None and not flags["vendor"] and flags["date"] and flags["total"]
    assert not any(w["label"] == L["VENDOR"] for w in words)
    assert unannot_mask("sroie_nv")[L["VENDOR"]] and not unannot_mask("sroie_nv")[L["TOTAL"]]


def test_rvl_val_split_name_fallback(monkeypatch):
    import sys, types
    from data import build_rvl_cache as rc
    from PIL import Image
    calls = []
    class DS:
        features = {"label": types.SimpleNamespace(names=rc.ORIGINAL_NAMES)}
        def cast_column(self, *a, **k): return self
        def shuffle(self, **k): return iter([{"label": 11, "image": Image.new("L", (40, 60), 255)}] * 5)
    def load_dataset(hf_id, split, streaming, **kw):
        calls.append(split)
        if split == "validation":
            raise ValueError("Bad split: validation. Available splits: ['train', 'test', 'val']")
        return DS()
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=load_dataset, Image=lambda **k: None))
    X, Y = rc.collect_hf("x/rvl", "validation", quota=2)
    assert calls == ["validation", "val"] and len(X) == 2 and set(Y) == {rc.CID["INVOICE"]}


def test_rvl_decode_skips_corrupt_images():
    import io
    from PIL import Image
    from data.build_rvl_cache import _decode
    buf = io.BytesIO(); Image.new("L", (300, 200), 128).save(buf, "TIFF")
    assert _decode({"bytes": buf.getvalue(), "path": None}).shape == (256, 256)
    assert _decode({"bytes": b"II*\x00corrupt", "path": None}) is None
