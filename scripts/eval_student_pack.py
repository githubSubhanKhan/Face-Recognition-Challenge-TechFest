"""Evaluate the pipeline on the teacher's student pack (Face Embedding Challenge).

The pack has <data>/train (one photo per person, named by person) and <data>/test (anonymous IDs).
"Training" here is enrolment: every train image is embedded with the frozen pipeline and becomes the
gallery. Each test image is then assigned to its closest train image by cosine similarity, which is exactly
how the teacher scores the submission (top-1 accuracy, top-5 breaks ties).

    python scripts/eval_student_pack.py --data student_pack/student_pack/student_pack/practice \
        --key student_pack/student_pack/student_pack/practice_key.csv --group group05

--key (test_id,person) prints accuracy / precision / recall / F1; without it (lab day) only predictions
are written. --group also saves <group>.npz in the format make_submission.py produces, ready to upload.

Group photos (--multi-face auto, the default): a test image with 2+ faces is split into faces and the face that
best matches an enrolled person becomes that image's embedding (the submission needs one vector per image).
Train images are always treated as single-person enrolment photos.
"""
import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import accuracy_score, precision_recall_fscore_support  # noqa: E402

from pakface.config import PipelineConfig, default_config  # noqa: E402
from pakface.data import IMG_EXT  # noqa: E402
from pakface.matching import Calibration, fused_embeddings  # noqa: E402
from pakface.pipeline import FacePipeline  # noqa: E402


def person_of(train_id, known):
    """Train images are named by person: 'maryam_nawaz.jpg', possibly 'maryam_nawaz_2.jpg' with several photos."""
    stem = Path(train_id).stem
    return stem if stem in known else re.sub(r"[_\-\s]*\d+$", "", stem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder containing train/ and test/")
    ap.add_argument("--key", default="", help="answer CSV with test_id,person (practice_key.csv)")
    ap.add_argument("--group", default="", help="also save <group>.npz in the submission format")
    ap.add_argument("--out", default="student_pack_predictions.csv")
    ap.add_argument("--config", default="configs/pipeline.json")
    ap.add_argument("--calibration", default="results/calibration.json")
    ap.add_argument("--cache", default="cache/student_pack")
    ap.add_argument("--ctx", type=int, default=0, help="GPU id, -1 for CPU")
    ap.add_argument("--multi-face", choices=["auto", "off"], default="auto",
                    help="auto: test images with 2+ faces use the face that best matches an enrolled person")
    ap.add_argument("--min-face-rel", type=float, default=0.25,
                    help="ignore faces smaller than this fraction of the image's biggest face")
    args = ap.parse_args()

    # same file lists and order as make_submission.py, so the .npz IDs match what the teacher expects
    tr_dir, te_dir = Path(args.data) / "train", Path(args.data) / "test"
    tr_ids = sorted(f for f in os.listdir(tr_dir) if Path(f).suffix.lower() in IMG_EXT)
    te_ids = sorted(f for f in os.listdir(te_dir) if Path(f).suffix.lower() in IMG_EXT)
    tr_paths = [str((tr_dir / f).resolve()) for f in tr_ids]
    te_paths = [str((te_dir / f).resolve()) for f in te_ids]
    print(f"train: {len(tr_ids)} images | test: {len(te_ids)} images")

    cfg = PipelineConfig.from_json(args.config) if os.path.exists(args.config) else default_config()
    cfg.ctx_id = args.ctx
    cfg.cache_dir = args.cache
    calib = Calibration.load(args.calibration) if os.path.exists(args.calibration) else None
    weights = calib.weights if calib and calib.weights else None

    pipe = FacePipeline(cfg)
    if args.multi_face == "auto":
        faces, _ = pipe.expand_faces(te_paths, args.min_face_rel)
    else:
        faces = {p: [p] for p in te_paths}
    store = pipe.build_store(tr_paths + [k for p in te_paths for k in faces[p]])
    tr_emb = fused_embeddings(store, tr_paths, weights).astype(np.float32)
    te_emb, n_group = [], 0
    for p in te_paths:
        e = fused_embeddings(store, faces[p], weights)
        n_group += len(e) > 1
        te_emb.append(e[(e @ tr_emb.T).max(1).argmax()])   # face closest to any enrolled person
    te_emb = np.stack(te_emb).astype(np.float32)
    if n_group:
        print(f"{n_group} test images are group photos (scored by their best-matching face)")

    if args.group:
        np.savez(f"{args.group}.npz", train_ids=np.array(tr_ids), train_emb=tr_emb,
                 test_ids=np.array(te_ids), test_emb=te_emb)
        print(f"saved {args.group}.npz {tr_emb.shape} {te_emb.shape}")

    key = pd.read_csv(args.key) if args.key else None
    known = set(key["person"]) if key is not None else set()
    tr_person = np.array([person_of(i, known) for i in tr_ids])

    sims = te_emb @ tr_emb.T                       # cosine: fused vectors are unit length
    order = np.argsort(-sims, axis=1)
    pred = tr_person[order[:, 0]]
    top5 = [list(dict.fromkeys(tr_person[o]))[:5] for o in order]   # 5 closest distinct people
    out = pd.DataFrame({"test_id": te_ids, "predicted": pred,
                        "score": np.round(sims[np.arange(len(te_ids)), order[:, 0]], 4),
                        "second_best": [t[1] if len(t) > 1 else "" for t in top5]})

    if key is not None:
        truth = dict(zip(key["test_id"].astype(str), key["person"].astype(str)))
        missing = [t for t in te_ids if t not in truth]
        if missing:
            raise SystemExit(f"{len(missing)} test images not in {args.key}, e.g. {missing[:3]}")
        y = np.array([truth[t] for t in te_ids])
        out.insert(1, "true", y)
        out["correct"] = (out["true"] == out["predicted"]).map({True: "yes", False: "NO"})
        labels = sorted(set(y) | set(pred))
        p, r, f1, _ = precision_recall_fscore_support(y, pred, labels=labels, average="macro", zero_division=0)
        pw, rw, f1w, _ = precision_recall_fscore_support(y, pred, labels=labels, average="weighted", zero_division=0)
        top5_acc = np.mean([t in c for t, c in zip(y, top5)])
        print("\n" + out.to_string(index=False))
        print("\n================ RESULTS ================")
        print(f"Test images         : {len(y)}  ({len(set(tr_person))} enrolled people)")
        print(f"Top-1 accuracy      : {accuracy_score(y, pred) * 100:6.2f} %")
        print(f"Top-5 accuracy      : {top5_acc * 100:6.2f} %")
        print(f"Precision (macro)   : {p * 100:6.2f} %    (weighted {pw * 100:6.2f} %)")
        print(f"Recall    (macro)   : {r * 100:6.2f} %    (weighted {rw * 100:6.2f} %)")
        print(f"F1 score  (macro)   : {f1 * 100:6.2f} %    (weighted {f1w * 100:6.2f} %)")
        print("=========================================")
    else:
        print("\n" + out.to_string(index=False))
        print("\n(no --key given, so no metrics: predictions only)")

    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
