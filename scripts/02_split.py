"""Identity-disjoint train / val / test split + verification pairs.

By default val and test identities come ONLY from your own consented Pakistani
collection (--eval-sources own_collection), because:
  * that is the population the hidden benchmark is drawn from;
  * public celebrity sets may already be inside the pretrained models' training data
    (WebFace / Glint360K are scraped from the web), which would inflate your numbers.

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
    ap.add_argument("--val-frac", type=float, default=0.4)
    ap.add_argument("--test-frac", type=float, default=0.6)
    ap.add_argument("--eval-sources", nargs="*", default=["own_collection"],
                    help="sources whose identities may be used for val/test (empty = any source)")
    ap.add_argument("--genuine-per-id", type=int, default=30)
    ap.add_argument("--impostor-ratio", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    eval_sources = [s for s in args.eval_sources if s in set(df["source"])] or None
    if args.eval_sources and not eval_sources:
        print(f"WARNING: none of {args.eval_sources} found; val/test will be drawn from all sources")
    df = identity_disjoint_split(df, args.val_frac, args.test_frac, args.seed, test_sources=eval_sources)
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
    # leakage check across splits (identity names)
    s = df.groupby("split")["identity"].apply(set)
    for a in s.index:
        for b in s.index:
            if a < b:
                assert not (s[a] & s[b]), f"identity overlap between {a} and {b}"
    print("OK: no identity appears in more than one split")


if __name__ == "__main__":
    main()
