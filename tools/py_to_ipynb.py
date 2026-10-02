"""Convert the `# %%`-cell scripts in notebooks/kaggle_*.py into .ipynb files that Kaggle can import
(Kaggle: New Notebook -> File -> Import Notebook).   Run from the repo root:   python tools/py_to_ipynb.py
"""
import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def convert(py_path: str) -> str:
    src = open(py_path, encoding="utf-8").read()
    parts = re.split(r"(?m)^(# %%.*)$", src)            # [pre, delimiter, body, delimiter, body, ...]
    chunks = [(parts[i], parts[i + 1].strip("\n")) for i in range(1, len(parts) - 1, 2)]
    cells = []
    for delim, c in chunks:
        if not c.strip():
            continue
        if "[markdown]" in delim:                       # only cells explicitly marked become markdown
            text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in c.splitlines())
            cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                          "source": c.splitlines(True)})
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}}
    out = os.path.splitext(py_path)[0] + ".ipynb"
    json.dump(nb, open(out, "w", encoding="utf-8"), indent=1)
    return out


if __name__ == "__main__":
    for p in sorted(glob.glob(os.path.join(ROOT, "notebooks", "kaggle_*.py"))):
        print("wrote", convert(p))
