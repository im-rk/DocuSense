"""Copy the training-relevant code into kaggle_bundle/docusense-code/ so you can upload that ONE folder to Kaggle as a
private Dataset named `docusense-code`.   Run from the repo root:   python tools/make_kaggle_bundle.py

Included: models/ (code only, no checkpoints) and data/ (code only, no processed data).
Kaggle notebooks import them with `from models... import ...` / `from data... import ...`.
"""
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "kaggle_bundle", "docusense-code")
SKIP_DIRS = {"__pycache__", "checkpoints", "processed", ".pytest_cache"}
SKIP_EXT = {".pth", ".pt", ".npy", ".pyc"}


def copy_tree(name):
    for dp, dirs, files in os.walk(os.path.join(ROOT, name)):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        rel = os.path.relpath(dp, ROOT)
        os.makedirs(os.path.join(OUT, rel), exist_ok=True)
        for f in files:
            if os.path.splitext(f)[1] not in SKIP_EXT:
                shutil.copy2(os.path.join(dp, f), os.path.join(OUT, rel, f))


if __name__ == "__main__":
    shutil.rmtree(OUT, ignore_errors=True)
    for pkg in ("models", "data"):
        copy_tree(pkg)
    n = sum(len(f) for _, _, f in os.walk(OUT))
    print(f"Bundled {n} files into {OUT}\nNow upload the contents of that folder to Kaggle as a Dataset named 'docusense-code'.")
