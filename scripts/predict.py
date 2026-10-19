"""Run the frozen pipeline on the HIDDEN BENCHMARK (or any new data).

Three modes, so you are ready whatever format the TechFest team uses:

1) Verification pairs (most likely):
     python scripts/predict.py pairs --csv benchmark/pairs.csv --root benchmark/images --out submission.csv
   The CSV needs two image columns (auto-detected: img1/img2, image1/image2, path_a/path_b, left/right, a/b,
   or simply the first two columns). Output adds `score` (cosine) and `prediction` (1 = same person).
   If the CSV also has a ground-truth column (label/is_same/same/match), metrics are printed.

2) Identification (gallery of known people + probe images):
     python scripts/predict.py identify --gallery benchmark/gallery --probes benchmark/probes --out ids.csv [--open-set]
   gallery/<person>/*.jpg ; probes/*.jpg (or probes/<person>/*.jpg to get accuracy).

3) Single pair:
     python scripts/predict.py one --img1 a.jpg --img2 b.jpg

Uses results/calibration.json (threshold + fusion weights fitted on your validation identities).
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pakface.config import PipelineConfig, default_config  # noqa: E402
from pakface.data import IMG_EXT  # noqa: E402
from pakface.matching import Calibration, build_templates, fused_embeddings, identify  # noqa: E402
from pakface.metrics import full_report  # noqa: E402
from pakface.pipeline import FacePipeline  # noqa: E402

PAIR_COLS = [("img1", "img2"), ("image1", "image2"), ("path_a", "path_b"), ("left", "right"), ("a", "b"),
             ("image_1", "image_2"), ("img_1", "img_2"), ("probe", "reference"), ("query", "reference")]
LABEL_COLS = ["label", "is_same", "same", "match", "issame", "ground_truth", "gt"]


def load(args):
    cfg = PipelineConfig.from_json(args.config) if os.path.exists(args.config) else default_config()
    cfg.ctx_id = args.ctx
    cfg.cache_dir = args.cache
    calib = Calibration.load(args.calibration) if os.path.exists(args.calibration) else None
    if calib is None:
        print(f"WARNING: {args.calibration} not found -> using an uncalibrated default threshold of 0.30")
        calib = Calibration(global_threshold=0.30, use_bins=False)
    return FacePipeline(cfg), calib


def resolve(p, root):
    p = str(p).strip()
    return p if (os.path.isabs(p) or not root) else os.path.join(root, p)


def cmd_pairs(args):
    df = pd.read_csv(args.csv)
    lower = {c.lower(): c for c in df.columns}
    ca, cb = next(((lower[a], lower[b]) for a, b in PAIR_COLS if a in lower and b in lower), (df.columns[0], df.columns[1]))
    pa = [resolve(p, args.root) for p in df[ca]]
    pb = [resolve(p, args.root) for p in df[cb]]
    missing = [p for p in set(pa + pb) if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"{len(missing)} images not found, e.g. {missing[:3]} - check --root")
    pipe, calib = load(args)
    store = pipe.build_store(pa + pb)
    s, _, q = store.subset_scores(pa, pb, calib.weights or None)
    thr = calib.thresholds_for(q, len(s))
    df["score"] = np.round(s, 6)
    df["prediction"] = (s > thr).astype(int)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(df)} pairs, {int(df['prediction'].sum())} predicted same-person")
    lab = next((lower[c] for c in LABEL_COLS if c in lower), None)
    if lab:
        y = df[lab].astype(str).str.strip().str.lower().map(
            {"1": 1, "1.0": 1, "true": 1, "same": 1, "yes": 1, "0": 0, "0.0": 0, "false": 0, "different": 0, "no": 0}
        ).to_numpy()
        rep = full_report(s, y, thr)
        print({k: round(v, 4) for k, v in rep.items() if isinstance(v, float)})


def list_images(d):
    return sorted(str(p) for p in Path(d).rglob("*") if p.suffix.lower() in IMG_EXT)


def cmd_identify(args):
    pipe, calib = load(args)
    gal = list_images(args.gallery)
    gal_ids = [Path(p).parent.name for p in gal]
    probes = list_images(args.probes)
    store = pipe.build_store(gal + probes)
    w = calib.weights or None

    ids, T = build_templates(fused_embeddings(store, gal, w), gal_ids, store.quality[[store.index[p] for p in gal]])
    # open-set rejection reuses the calibrated verification threshold (probe-vs-template cosine)
    preds, scores = identify(fused_embeddings(store, probes, w), ids, T,
                             calib.global_threshold if args.open_set else None)
    out = pd.DataFrame({"probe": probes, "predicted_identity": preds, "score": np.round(scores, 6)})
    probe_parent = [Path(p).parent.name for p in probes]
    if set(probe_parent) & set(ids):
        out["true_identity"] = probe_parent
        acc = (out["true_identity"] == out["predicted_identity"]).mean()
        print(f"rank-1 identification accuracy: {acc*100:.2f}%")
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}")


def cmd_one(args):
    pipe, calib = load(args)
    store = pipe.build_store([args.img1, args.img2])
    s, per, q = store.subset_scores([args.img1], [args.img2], calib.weights or None)
    t = calib.thresholds_for(q, 1)[0]
    print(f"score={s[0]:.4f} threshold={t:.4f} -> {'SAME person' if s[0] > t else 'DIFFERENT people'}")
    print({m: round(float(v[0]), 4) for m, v in per.items()}, "pair quality", round(float(q[0]), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pipeline.json")
    ap.add_argument("--calibration", default="results/calibration.json")
    ap.add_argument("--cache", default="cache/predict")
    ap.add_argument("--ctx", type=int, default=0)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pairs")
    p.add_argument("--csv", required=True)
    p.add_argument("--root", default="")
    p.add_argument("--out", default="submission.csv")
    i = sub.add_parser("identify")
    i.add_argument("--gallery", required=True)
    i.add_argument("--probes", required=True)
    i.add_argument("--open-set", action="store_true")
    i.add_argument("--out", default="identification.csv")
    o = sub.add_parser("one")
    o.add_argument("--img1", required=True)
    o.add_argument("--img2", required=True)
    args = ap.parse_args()
    {"pairs": cmd_pairs, "identify": cmd_identify, "one": cmd_one}[args.cmd](args)


if __name__ == "__main__":
    main()
