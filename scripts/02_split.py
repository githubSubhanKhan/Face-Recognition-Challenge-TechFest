"""Identity-disjoint train / val / test split + verification pairs.

By default val and test identities come ONLY from your own consented Pakistani
collection (--eval-sources own_collection), because:
  * that is the population the hidden benchmark is drawn from;
  * public celebrity sets may already be inside the pretrained models' training data
    (WebFace / Glint360K are scraped from the web), which would inflate your numbers.

--own-train-frac (default 0.4) of those own-collection identities stay in TRAIN, because the problem
statement requires fine-tuning on your own collected data; the rest are split val/test.

    python scripts/02_split.py --eval-sources own_collection
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from pakface.data import assert_disjoint, identity_disjoint_split, make_pairs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--out", default="data")
    ap.add_argument("--val-frac", type=float, default=None,
                    help="default 0.4 of eval-source identities, or 0.15 of all identities without eval sources")
    ap.add_argument("--test-frac", type=float, default=None,
                    help="default 0.6 of eval-source identities, or 0.15 of all identities without eval sources")
    ap.add_argument("--eval-sources", nargs="*", default=["own_collection"],
                    help="sources whose identities may be used for val/test (empty = any source)")
    ap.add_argument("--own-train-frac", type=float, default=0.4,
                    help="fraction of eval-source identities kept for TRAINING (fine-tuning must use your own "
                         "collected data); the rest is split val/test by --val-frac/--test-frac")
    ap.add_argument("--genuine-per-id", type=int, default=30)
    ap.add_argument("--impostor-ratio", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    eval_sources = [s for s in args.eval_sources if s in set(df["source"])] or None
    if args.eval_sources and not eval_sources:
        print(f"WARNING: none of {args.eval_sources} found; val/test will be drawn from all sources "
              "(fine for a practice run, but celebrity test numbers are inflated - don't report them)")
    # with eval sources the fractions split THOSE identities (the rest of train comes from other sources);
    # without them they are fractions of ALL identities, so most must stay in train
    dv, dt = (0.4, 0.6) if eval_sources else (0.15, 0.15)
    args.val_frac = dv if args.val_frac is None else args.val_frac
    args.test_frac = dt if args.test_frac is None else args.test_frac
    df = identity_disjoint_split(df, args.val_frac, args.test_frac, args.seed, test_sources=eval_sources,
                                 eval_source_train_frac=args.own_train_frac)
    assert_disjoint(df)
    df.to_csv(Path(args.out) / "splits.csv", index=False)

    hard = ("gender",) if "gender" in df.columns else ()
    for split in ["val", "test"]:
        sub = df[df["split"] == split]
        pairs = make_pairs(sub, args.genuine_per_id, args.impostor_ratio, seed=args.seed + (split == "test"),
                           hard_negative_cols=hard)
        pairs.to_csv(Path(args.out) / f"pairs_{split}.csv", index=False)
        print(f"{split:5s}: {sub['identity'].nunique():4d} identities, {len(sub):6d} images, "
              f"{int(pairs['label'].sum())} genuine + {int((pairs['label'] == 0).sum())} impostor pairs")
    tr = df[(df["split"] == "train") & df["train_usable"]]
    print(f"train: {tr['identity'].nunique():4d} identities, {len(tr):6d} usable images")
    if eval_sources:
        own = tr[tr["source"].isin(eval_sources)]
        print(f"       of which {own['identity'].nunique()} identities / {len(own)} images from {eval_sources}")
    # leakage check across splits (identity names)
    s = df.groupby("split")["identity"].apply(set)
    for a in s.index:
        for b in s.index:
            if a < b:
                assert not (s[a] & s[b]), f"identity overlap between {a} and {b}"
    print("OK: no identity appears in more than one split")


if __name__ == "__main__":
    main()
