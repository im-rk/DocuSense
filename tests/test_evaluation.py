import json
import os

import pytest

pytest.importorskip("torch")                     # evaluation.run_evaluation imports the pipeline


def test_run_evaluation_scores_fields(tmp_path, monkeypatch):
    import evaluation.run_evaluation as ev
    fake = {"document_type": "RECEIPT", "validation": {"needs_review": False},
            "fields": {"vendor": "Acme Store", "date": "2018-12-25", "total": 15.9, "tax": 1.0}}
    monkeypatch.setattr(ev, "process", lambda path: dict(fake))
    monkeypatch.chdir(tmp_path)
    os.makedirs("evaluation")
    json.dump([{"file": "a.jpg", "ground_truth": {"vendor": "ACME STORE!", "date": "25/12/2018", "total": 15.9, "tax": 0.9,
                                                  "colour": "ignored"}}], open("own_eval.json", "w"))
    ev.main("own_eval.json")
    res = json.load(open("evaluation/results.json"))
    assert res["field_accuracy"] == {"vendor": 1.0, "date": 1.0, "tax": 0.0, "total": 1.0}
    assert res["n_documents"] == 1 and res["overall"] == 0.75
