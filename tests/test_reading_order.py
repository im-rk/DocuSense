import numpy as np

from data.crops import make_crop, make_crops
from data.reading_order import order_words
from models.tagger.features import MAX_CHARS, crops_to_float, encode_chars, pos_features


def box(x, y, w=40, h=10):
    return [x, y, x + w, y + h]


def test_order_words_lines_and_flags():
    words = [{"text": "c", "box": box(100, 52)}, {"text": "a", "box": box(0, 0)},
             {"text": "d", "box": box(200, 50)}, {"text": "b", "box": box(100, 2)}]
    out = order_words(words)
    assert [w["text"] for w in out] == ["a", "b", "c", "d"]
    assert [w["line"] for w in out] == [0, 0, 1, 1]
    assert [w["is_first"] for w in out] == [1, 0, 1, 0]
    assert [w["is_last"] for w in out] == [0, 1, 0, 1]
    assert order_words([]) == []


def test_pos_features_and_chars():
    f = pos_features(np.array([[10, 20, 60, 40]]), 100, 200, [1], [0])
    assert f.shape == (1, 10)
    assert np.allclose(f[0, :8], [0.1, 0.1, 0.6, 0.2, 0.5, 0.1, 0.35, 0.15]) and f[0, 8] == 1 and f[0, 9] == 0
    ids = encode_chars(["ab", "zzz"], {"a": 2, "b": 3})
    assert ids.shape == (2, MAX_CHARS) and ids[0, :3].tolist() == [2, 3, 0] and ids[1, :3].tolist() == [1, 1, 1]


def test_crops():
    page = np.full((100, 200), 255, np.uint8)
    page[40:50, 20:80] = 0
    c = make_crop(page, [20, 38, 80, 52])
    assert c.shape == (32, 96) and c.dtype == np.uint8 and c.min() < 50
    assert make_crop(page, [500, 500, 510, 510]).shape == (32, 96)          # out of page -> blank, no crash
    assert make_crops(page, []).shape == (0, 32, 96)
    x = crops_to_float(make_crops(page, [[20, 38, 80, 52]]))
    assert x.shape == (1, 1, 32, 96) and 0.0 <= x.min() and x.max() <= 1.0
