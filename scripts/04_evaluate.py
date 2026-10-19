"""Full evaluation of the pipeline on the held-out TEST identities.

* thresholds + fusion weights are fitted on VALIDATION pairs only (never on test)
* reports accuracy, precision, recall, F1, FAR, FRR, EER, AUC, TAR@FAR
* ROC, DET and score-distribution plots
* per-group breakdown (gender, age group, skin tone, region, headwear, glasses, mask, lighting, quality flags)
* 1:N identification on the test identities: rank-1 / rank-5, open-set DIR / FPIR
* writes results/calibration.json, which predict.py uses on the hidden benchmark

    python scripts/04_evaluate.py --name full_pipeline
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from pakface.config import PipelineConfig, default_config  # noqa: E402
from pakface.evaluation import (  # noqa: E402
    calibrate_and_score, group_breakdown, identification_markdown, identification_report, report_markdown,
)
from pakface.matching import Calibration  # noqa: E402
from pakface.metrics import full_report, plot_all  # noqa: E402
from pakface.pipeline import FacePipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pipeline.json")
    ap.add_argument("--pairs-val", default="data/pairs_val.csv")
    ap.add_argument("--pairs-test", default="data/pairs_test.csv")
    ap.add_argument("--splits", default="data/splits.csv", help="for per-group metadata")
    ap.add_argument("--method", choices=["max_f1", "max_accuracy", "eer"], default="max_f1",
                    help="threshold objective; max_f1 because F1 is the organisers' ranking metric")
    ap.add_argument("--no-bins", action="store_true", help="single global threshold")
    ap.add_argument("--name", default="full_pipeline")
    ap.add_argument("--out", default="results")
    ap.add_argument("--ctx", type=int, default=0)
    args = ap.parse_args()

    cfg = PipelineConfig.from_json(args.config) if os.path.exists(args.config) else default_config()
    cfg.ctx_id = args.ctx
    out = Path(args.out) / args.name
    out.mkdir(parents=True, exist_ok=True)

    pv, pt = pd.read_csv(args.pairs_val), pd.read_csv(args.pairs_test)
    pipe = FacePipeline(cfg)
    store = pipe.build_store(pd.concat([pv["path_a"], pv["path_b"], pt["path_a"], pt["path_b"]]).tolist())

    use_bins = cfg.quality_binned_threshold and not args.no_bins
    calib, rep, s, q, thr, per_model = calibrate_and_score(store, pv, pt, args.method, use_bins)
    calib.save(Path(args.out) / "calibration.json")
    calib.save(out / "calibration.json")

    y = pt["label"].to_numpy()
    md = [report_markdown(rep, f"{args.name} - test identities (threshold fitted on val)")]
    _, per_model_val, _ = store.subset_scores(pv["path_a"].tolist(), pv["path_b"].tolist())
    for m, sm in per_model.items():
        c = Calibration(method=args.method, use_bins=False).fit(per_model_val[m], pv["label"].to_numpy())
        md.append(report_markdown(full_report(sm, y, c.global_threshold), f"single model: {m}"))
    plots = plot_all(s, y, thr, str(out / "test"), title=args.name)

    meta = pd.read_csv(args.splits) if os.path.exists(args.splits) else None
    groups = group_breakdown(pt, s, thr, meta, store)
    groups.to_csv(out / "per_group.csv", index=False)
    if len(groups):
        md.append("### Per-group breakdown (test)\n\n" + groups.to_markdown(index=False, floatfmt=".4f"))

    ident = identification_report(store, pd.concat([pt["path_a"], pt["path_b"]]).tolist(),
                                  pd.concat([pt["id_a"], pt["id_b"]]).tolist(), calib.weights, calib.global_threshold)
    if ident:
        rep["identification"] = ident
        md.append(identification_markdown(ident))
    pd.DataFrame({"path_a": pt["path_a"], "path_b": pt["path_b"], "label": y, "score": s, "pair_quality": q,
                  "threshold": thr, "pred": (s > thr).astype(int)}).to_csv(out / "test_scores.csv", index=False)
    # hardest errors for the "challenges" slide
    errs = pd.read_csv(out / "test_scores.csv")
    errs = errs[errs["pred"] != errs["label"]].assign(margin=lambda d: (d["score"] - d["threshold"]).abs())
    errs.sort_values("margin", ascending=False).head(40).to_csv(out / "worst_errors.csv", index=False)

    (out / "metrics.json").write_text(json.dumps(rep, indent=2))
    (out / "summary.md").write_text("\n\n".join(md))
    print("\n\n".join(md))
    print(f"\nplots: {plots}\ncalibration -> {Path(args.out) / 'calibration.json'}")


if __name__ == "__main__":
    main()
