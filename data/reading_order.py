"""Reading order for words on a page. Shared by dataset builders (training) and pipeline (serving)."""


def order_words(words: list, tol: float = 0.5) -> list:
    """words: list of dicts with 'box' = [x0,y0,x1,y1] (pixels). Returns NEW dicts in reading order
    (top-to-bottom lines, left-to-right within a line) with 'line', 'is_first', 'is_last' added."""
    if not words:
        return []
    heights = sorted(w["box"][3] - w["box"][1] for w in words)
    med_h = max(heights[len(heights) // 2], 1.0)
    by_y = sorted(words, key=lambda w: (w["box"][1] + w["box"][3]) / 2)

    lines = []                                   # each: {"yc": running mean, "words": [...]}
    for w in by_y:
        yc = (w["box"][1] + w["box"][3]) / 2
        if lines and abs(yc - lines[-1]["yc"]) <= tol * med_h:
            cur = lines[-1]
            cur["words"].append(w)
            cur["yc"] += (yc - cur["yc"]) / len(cur["words"])
        else:
            lines.append({"yc": yc, "words": [w]})

    out = []
    for li, line in enumerate(lines):
        row = sorted(line["words"], key=lambda w: w["box"][0])
        for j, w in enumerate(row):
            d = dict(w)
            d["line"], d["is_first"], d["is_last"] = li, int(j == 0), int(j == len(row) - 1)
            out.append(d)
    return out
