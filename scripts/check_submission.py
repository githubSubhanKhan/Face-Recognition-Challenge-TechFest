"""Check a Face Embedding Challenge submission (.npz) the way the teacher grades it, before uploading.

    python scripts/check_submission.py group05.npz --data <data_dir> [--key practice_key.csv]

Format rules from the student pack README: keys train_ids, train_emb [N,D], test_ids, test_emb [N,D]; same D;
no NaNs; ID arrays exactly as make_submission.py writes them (sorted os.listdir of train/ and test/).
Scoring: each test embedding is matched to the closest train embedding by cosine similarity; score = top-1
accuracy, top-5 breaks ties. Scores are only computed when the answer key is given (practice).
"""
import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ("train_ids", "train_emb", "test_ids", "test_emb")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--data", default="", help="folder with train/ and test/: checks the ID arrays against it")
    ap.add_argument("--key", default="", help="answer CSV with test_id,person: prints top-1 / top-5 accuracy")
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=False)
    errors, warnings = [], []
    missing = [k for k in KEYS if k not in z.files]
    if missing:
        sys.exit(f"FAIL: missing arrays {missing} (has {z.files})")
    tr_ids, tr, te_ids, te = (z[k] for k in KEYS)
    print(f"train_emb {tr.shape} {tr.dtype} | test_emb {te.shape} {te.dtype} | extra arrays: "
          f"{[k for k in z.files if k not in KEYS] or 'none'}")

    if tr.ndim != 2 or te.ndim != 2:
        errors.append("embeddings must be 2-D [N, D]")
    elif tr.shape[1] != te.shape[1]:
        errors.append(f"train D={tr.shape[1]} != test D={te.shape[1]}")
    if len(tr_ids) != len(tr) or len(te_ids) != len(te):
        errors.append("number of IDs and embeddings differ")
    for name, e in (("train_emb", tr), ("test_emb", te)):
        if not np.issubdtype(e.dtype, np.floating):
            errors.append(f"{name} is {e.dtype}, not float")
        elif not np.isfinite(e).all():
            errors.append(f"{name} has {int((~np.isfinite(e)).sum())} NaN/inf values")
        else:
            n = np.linalg.norm(e, axis=1)
            if (n < 1e-6).any():
                errors.append(f"{name} has {int((n < 1e-6).sum())} all-zero vectors (cosine undefined)")
            elif not np.allclose(n, 1, atol=1e-3):
                warnings.append(f"{name} is not unit length (norms {n.min():.3f}-{n.max():.3f}): fine for cosine, "
                                "but a plain dot-product grader would rank differently")

    if args.data:
        for name, ids in (("train", tr_ids), ("test", te_ids)):
            listing = sorted(os.listdir(Path(args.data) / name))  # exactly what make_submission.py uses
            if list(ids) != listing:
                extra = sorted(set(listing) - set(ids))
                errors.append(f"{name}_ids differ from sorted(os.listdir({name}/))"
                              + (f"; not in the npz: {extra[:5]}" if extra else ""))

    for w in warnings:
        print("WARNING:", w)
    if errors:
        for e in errors:
            print("FAIL:", e)
        sys.exit(1)
    print("OK: format matches the submission rules")

    if args.key:
        key = pd.read_csv(args.key)
        truth = dict(zip(key["test_id"].astype(str), key["person"].astype(str)))
        known = set(truth.values())
        person = np.array([s if s in known else re.sub(r"[_\-\s]*\d+$", "", s)
                           for s in (Path(str(i)).stem for i in tr_ids)])
        trn = tr / np.linalg.norm(tr, axis=1, keepdims=True)
        ten = te / np.linalg.norm(te, axis=1, keepdims=True)
        order = np.argsort(-(ten @ trn.T), axis=1)
        y = np.array([truth[str(i)] for i in te_ids])
        top1 = np.mean(person[order[:, 0]] == y)
        top5 = np.mean([t in person[o[:5]] for t, o in zip(y, order)])
        print(f"top-1 accuracy: {top1 * 100:.2f}%   top-5 accuracy: {top5 * 100:.2f}%   ({len(y)} test images)")


if __name__ == "__main__":
    main()
