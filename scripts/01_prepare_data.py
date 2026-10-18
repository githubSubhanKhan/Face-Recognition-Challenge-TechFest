"""Prepare the merged dataset: scan sources, de-duplicate, detect + align + quality-score every image.

Outputs
  data/manifest.csv          one row per usable image (raw path, aligned crop path, identity, source, quality...)
  data/aligned/<id>/*.png    112x112 aligned crops (used by fine-tuning)
  data/report_prepare.txt    counts, detection-stage breakdown, quality flags
  data/label_conflicts.csv   near-identical images filed under DIFFERENT identities (fix these by hand!)

    python scripts/01_prepare_data.py --raw data/raw --identities data/identities.csv --images data/images.csv
"""
import argparse
import hashlib
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from pakface.config import PipelineConfig, default_config  # noqa: E402
from pakface.data import attach_metadata, dhash, find_duplicates, scan  # noqa: E402
from pakface.pipeline import FacePipeline, load_image  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data")
    ap.add_argument("--identities", default="data/identities.csv")
    ap.add_argument("--images", default="data/images.csv")
    ap.add_argument("--config", default="configs/pipeline.json")
    ap.add_argument("--min-train-quality", type=float, default=0.15,
                    help="crops below this are marked unusable for TRAINING (still allowed in val/test)")
    ap.add_argument("--dedupe-dist", type=int, default=4,
                    help="max dHash Hamming distance treated as duplicate (0 disables); burst shots are dropped")
    ap.add_argument("--ctx", type=int, default=0)
    args = ap.parse_args()

    df = scan(args.raw)
    df = attach_metadata(df, args.identities, args.images)
    print(f"scanned {len(df)} images, {df['identity'].nunique()} identities, sources: {dict(Counter(df['source']))}")

    # ---- de-duplication
    hashes, ok = [], []
    for p in tqdm(df["path"], desc="hashing"):
        try:
            g = cv2.cvtColor(load_image(p), cv2.COLOR_BGR2GRAY)
            hashes.append(dhash(cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)))
            ok.append(True)
        except Exception as e:  # unreadable / corrupt file
            print(f"  unreadable {p}: {e}")
            hashes.append(0)
            ok.append(False)
    df = df[ok].reset_index(drop=True)
    hashes = [h for h, k in zip(hashes, ok) if k]
    drop, conflicts = find_duplicates(df, hashes, args.dedupe_dist) if args.dedupe_dist > 0 else (set(), [])
    pd.DataFrame(conflicts, columns=["path_a", "path_b"]).to_csv(Path(args.out) / "label_conflicts.csv", index=False)
    df = df.drop(index=list(drop)).reset_index(drop=True)
    print(f"removed {len(drop)} duplicate images; {len(conflicts)} cross-identity near-duplicates -> label_conflicts.csv")

    # ---- detect / align / quality
    cfg = PipelineConfig.from_json(args.config) if os.path.exists(args.config) else default_config()
    cfg.ctx_id = args.ctx
    pipe = FacePipeline(cfg, load_embedders=False)
    aligned_root = Path(args.out) / "aligned"
    rows = []
    for r in tqdm(df.itertuples(), total=len(df), desc="detect+align"):
        crop, q, flags, stage = pipe.preprocess(r.path)
        dst = aligned_root / r.identity / (Path(r.path).stem + f"_{hashlib.md5(r.path.encode()).hexdigest()[:8]}.png")
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), crop)
        rows.append({"aligned_path": str(dst), "quality": q, "flags": "|".join(flags), "det_stage": stage})
    pipe.save_det_cache()
    df = pd.concat([df, pd.DataFrame(rows)], axis=1)
    df["train_usable"] = (df["quality"] >= args.min_train_quality) & (df["det_stage"] != "no_detection")
    df.to_csv(Path(args.out) / "manifest.csv", index=False)

    lines = [
        f"images: {len(df)}   identities: {df['identity'].nunique()}",
        f"per source: {dict(Counter(df['source']))}",
        f"detection stages: {dict(Counter(df['det_stage']))}",
        f"quality flags: {dict(Counter(f for fl in df['flags'].fillna('') for f in fl.split('|') if f))}",
        f"train-usable: {int(df['train_usable'].sum())} / {len(df)}",
        f"images per identity: min {df.groupby('identity').size().min()}, median {df.groupby('identity').size().median()}",
    ]
    Path(args.out, "report_prepare.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
