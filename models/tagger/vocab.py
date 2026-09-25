"""Character vocabulary for the text branch. Index 0 = padding, 1 = unknown."""
import json
from collections import Counter

PAD, UNK = 0, 1


def build_vocab(texts, min_count: int = 2) -> list:
    """texts: iterable of word strings (training set only). Returns list of characters (index = position + 2)."""
    cnt = Counter()
    for t in texts:
        cnt.update(t)
    return sorted(ch for ch, n in cnt.items() if n >= min_count)


def save_vocab(chars: list, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"chars": chars}, f, ensure_ascii=False)


def load_vocab(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        chars = json.load(f)["chars"]
    return {ch: i + 2 for i, ch in enumerate(chars)}


def vocab_size(vocab: dict) -> int:
    return len(vocab) + 2
