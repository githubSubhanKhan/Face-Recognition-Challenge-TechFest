"""Run the frozen pipeline on the HIDDEN BENCHMARK (or any new data).

Four modes, so you are ready whatever format the TechFest team uses:

1) Verification pairs:
     python scripts/predict.py pairs --csv benchmark/pairs.csv --root benchmark/images --out submission.csv
   The CSV needs two image columns (auto-detected: img1/img2, image1/image2, path_a/path_b, left/right, a/b,
   or simply the first two columns). Output adds `score` (cosine) and `prediction` (1 = same person).
   If the CSV also has a ground-truth column (label/is_same/same/match), metrics are printed.

2) Identification (gallery of known people + probe images):
     python scripts/predict.py identify --gallery benchmark/gallery --probes benchmark/probes --out ids.csv [--open-set]
   gallery/<person>/*.jpg ; probes/*.jpg (or probes/<person>/*.jpg to get accuracy).

3) Search: find one person in a set of photos ("pick a single face out of an entire school"):
     python scripts/predict.py search --query person.jpg --photos school_photos --out matches.csv --draw found/
   --query may be one image or a folder of images of the same person.

4) Single pair:
     python scripts/predict.py one --img1 a.jpg --img2 b.jpg

Group photos (--multi-face auto, the default): an image with 2+ faces is split into one entry per face;
single-person images keep the normal single-face pipeline, so their results are unchanged.
  pairs    : score = best-matching face combination ("is the person in A also in B?")
  identify : one output row per face; faces in group photos below the threshold become "unknown"
  search   : every face of every photo is compared with the query person
Gallery images are always treated as single-person enrolment photos.

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
from pakface.pipeline import FacePipeline, load_image, split_face_key  # noqa: E402

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


def list_images(d):
    return sorted(str(p) for p in Path(d).rglob("*") if p.suffix.lower() in IMG_EXT)


def faces_for(pipe, paths, args, force=False):
    """{image: [face keys]} - one key per face for group photos, [image] otherwise."""
    if args.multi_face == "off" and not force:
        return {p: [p] for p in dict.fromkeys(paths)}, {}
    return pipe.expand_faces(paths, args.min_face_rel)


def face_index(key):
    k = split_face_key(key)[1]
    return "" if k is None else k


def bbox_str(pipe, key, boxes):
    """x1 y1 x2 y2 of the face, from the group-photo detections or the single-face detection cache."""
    if key in boxes:
        b = boxes[key]
    else:
        d = pipe.det_cache.get(os.path.abspath(key))
        b = d["bbox"] if d else None
    return "" if b is None else " ".join(str(int(round(v))) for v in b)


def best_face_pairs(store, faces, pa, pb, weights):
    """Score every face in A against every face in B and keep the best combination per pair.
    For two single-person images this is exactly the ordinary pair score."""
    rows, ka, kb = [], [], []
    for r, (a, b) in enumerate(zip(pa, pb)):
        for x in faces[a]:
            for y in faces[b]:
                rows.append(r)
                ka.append(x)
                kb.append(y)
    s, _, q = store.subset_scores(ka, kb, weights)
    best = pd.Series(s).groupby(np.array(rows)).idxmax().to_numpy()
    return s[best], q[best], [ka[i] for i in best], [kb[i] for i in best]


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
    faces, _ = faces_for(pipe, pa + pb, args)
    store = pipe.build_store([k for ks in faces.values() for k in ks])
    s, q, fa, fb = best_face_pairs(store, faces, pa, pb, calib.weights or None)
    thr = calib.thresholds_for(q, len(s))
    df["score"] = np.round(s, 6)
    df["prediction"] = (s > thr).astype(int)
    n_group = sum(len(faces[a]) > 1 or len(faces[b]) > 1 for a, b in zip(pa, pb))
    if n_group:
        df["faces_a"], df["faces_b"] = [len(faces[a]) for a in pa], [len(faces[b]) for b in pb]
        df["matched_face_a"], df["matched_face_b"] = [face_index(k) for k in fa], [face_index(k) for k in fb]
        print(f"{n_group} pairs involve group photos (scored by their best-matching face)")
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(df)} pairs, {int(df['prediction'].sum())} predicted same-person")
    lab = next((lower[c] for c in LABEL_COLS if c in lower), None)
    if lab:
        y = df[lab].astype(str).str.strip().str.lower().map(
            {"1": 1, "1.0": 1, "true": 1, "same": 1, "yes": 1, "0": 0, "0.0": 0, "false": 0, "different": 0, "no": 0}
        ).to_numpy()
        rep = full_report(s, y, thr)
        print({k: round(v, 4) for k, v in rep.items() if isinstance(v, float)})


def cmd_identify(args):
    pipe, calib = load(args)
    gal = list_images(args.gallery)
    gal_ids = [Path(p).parent.name for p in gal]
    probes = list_images(args.probes)
    faces, boxes = faces_for(pipe, probes, args)
    keys = [k for p in probes for k in faces[p]]
    store = pipe.build_store(gal + keys)
    w = calib.weights or None

    ids, T = build_templates(fused_embeddings(store, gal, w), gal_ids, store.quality[[store.index[p] for p in gal]])
    preds, scores = identify(fused_embeddings(store, keys, w), ids, T)
    # open-set rejection reuses the calibrated verification threshold (probe-vs-template cosine). Faces in group
    # photos always use it: bystanders who are not enrolled must come out as "unknown", not as their look-alike.
    group = np.array([split_face_key(k)[1] is not None for k in keys], dtype=bool)
    reject = (scores <= calib.global_threshold) & (group | args.open_set)
    preds = ["unknown" if r else p for p, r in zip(preds, reject)]
    images = [split_face_key(k)[0] for k in keys]
    out = pd.DataFrame({"probe": images, "predicted_identity": preds, "score": np.round(scores, 6)})
    if group.any():
        out.insert(1, "face", [face_index(k) for k in keys])
        out.insert(2, "bbox", [bbox_str(pipe, k, boxes) for k in keys])
    probe_parent = [Path(p).parent.name for p in images]
    if set(probe_parent) & set(ids):
        out["true_identity"] = probe_parent
        # per image: correct if the true person is among the faces identified in it
        hit = (out["true_identity"] == out["predicted_identity"]).groupby(out["probe"]).any()
        print(f"rank-1 identification accuracy: {hit.mean()*100:.2f}% of {len(hit)} probe images")
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}" + (f" ({int(group.sum())} faces from group photos)" if group.any() else ""))


def cmd_search(args):
    pipe, calib = load(args)
    query = [args.query] if os.path.isfile(args.query) else list_images(args.query)
    photos = [p for p in list_images(args.photos) if p not in set(query)]
    faces, boxes = faces_for(pipe, photos, args, force=True)
    keys = [k for p in photos for k in faces[p]]
    store = pipe.build_store(query + keys)
    w = calib.weights or None
    _, T = build_templates(fused_embeddings(store, query, w), ["query"] * len(query),
                           store.quality[[store.index[p] for p in query]])
    s = fused_embeddings(store, keys, w) @ T[0]
    thr = calib.global_threshold
    out = pd.DataFrame({"photo": [split_face_key(k)[0] for k in keys], "face": [face_index(k) for k in keys],
                        "bbox": [bbox_str(pipe, k, boxes) for k in keys], "score": np.round(s, 6),
                        "match": (s > thr).astype(int)}).sort_values("score", ascending=False)
    out.to_csv(args.out, index=False)
    found = out[out["match"] == 1]["photo"].nunique()
    print(f"person found in {found} of {len(photos)} photos ({len(keys)} faces searched, threshold {thr:.4f})")
    print(out.head(10).to_string(index=False))
    if args.draw:
        draw_matches(out, args.draw)
        print(f"annotated photos -> {args.draw}")
    print(f"wrote {args.out}")


def draw_matches(out: pd.DataFrame, out_dir: str):
    """Green box + score on the best-matching face of every photo where the person was found; grey on the rest."""
    import cv2

    os.makedirs(out_dir, exist_ok=True)
    for photo, d in out.groupby("photo"):
        if not d["match"].any():
            continue
        img = load_image(photo)
        lw, best = max(2, img.shape[1] // 400), d["score"].idxmax()
        for i, r in d.iterrows():
            if not r["bbox"]:
                continue
            x1, y1, x2, y2 = map(int, r["bbox"].split())
            color = (0, 200, 0) if i == best else (160, 160, 160)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, lw)
            if i == best:
                cv2.putText(img, f"{r['score']:.2f}", (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                            max(0.5, img.shape[1] / 1500), color, lw)
        cv2.imwrite(str(Path(out_dir) / f"{Path(photo).stem}_match.jpg"), img)


def cmd_one(args):
    pipe, calib = load(args)
    faces, _ = faces_for(pipe, [args.img1, args.img2], args)
    store = pipe.build_store([k for ks in faces.values() for k in ks])
    s, q, fa, fb = best_face_pairs(store, faces, [args.img1], [args.img2], calib.weights or None)
    t = calib.thresholds_for(q, 1)[0]
    print(f"score={s[0]:.4f} threshold={t:.4f} -> {'SAME person' if s[0] > t else 'DIFFERENT people'}")
    _, per, _ = store.subset_scores(fa, fb, calib.weights or None)
    print({m: round(float(v[0]), 4) for m, v in per.items()}, "pair quality", round(float(q[0]), 3))
    if len(faces[args.img1]) > 1 or len(faces[args.img2]) > 1:
        print(f"faces: {len(faces[args.img1])} vs {len(faces[args.img2])}; best match = face "
              f"{face_index(fa[0]) or 0} vs face {face_index(fb[0]) or 0} (left to right)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pipeline.json")
    ap.add_argument("--calibration", default="results/calibration.json")
    ap.add_argument("--cache", default="cache/predict")
    ap.add_argument("--ctx", type=int, default=0)
    ap.add_argument("--multi-face", choices=["auto", "off"], default="auto",
                    help="auto: images with 2+ faces are split into one entry per face (group photos)")
    ap.add_argument("--min-face-rel", type=float, default=0.25,
                    help="ignore faces smaller than this fraction of the image's biggest face (bystanders, posters)")
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
    s = sub.add_parser("search")
    s.add_argument("--query", required=True, help="image of the person, or a folder of their images")
    s.add_argument("--photos", required=True, help="folder of photos to search (group, classroom, CCTV frames)")
    s.add_argument("--out", default="matches.csv")
    s.add_argument("--draw", default="", help="folder for annotated copies of the photos where the person was found")
    o = sub.add_parser("one")
    o.add_argument("--img1", required=True)
    o.add_argument("--img2", required=True)
    args = ap.parse_args()
    {"pairs": cmd_pairs, "identify": cmd_identify, "search": cmd_search, "one": cmd_one}[args.cmd](args)


if __name__ == "__main__":
    main()
