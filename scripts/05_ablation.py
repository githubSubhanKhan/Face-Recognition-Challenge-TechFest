"""Ablation study + robustness stress tests (the "how much does each stage contribute" table).

Every row is evaluated identically: thresholds/weights fitted on VAL identities, metrics on TEST identities.

Rows
  naive baseline        buffalo_l ArcFace R50 only, no retries, no TTA, no enhancement, single threshold
  FULL PIPELINE         everything on
  - flip TTA            full minus test-time flip augmentation
  - enhancement         full minus adaptive low-light/contrast enhancement
  - alignment           full minus 5-point alignment (plain bbox crop)
  - detection retries   full minus pad/upscale/brighten retry cascade
  - quality thresholds  full with one global threshold instead of quality-binned thresholds
  only AdaFace          single backbone
  only Glint R100       single backbone
  + fine-tuned AdaFace  (with --finetuned path) full pipeline using the fine-tuned checkpoint

Stress tests: image B of every test pair is degraded (blur, low-res, low light, JPEG, CCTV, grayscale)
for the naive baseline and the full pipeline.

    python scripts/05_ablation.py [--finetuned checkpoints/adaface_ir101_finetuned.pt] [--skip-stress]
"""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pakface.augment import STRESS_TESTS  # noqa: E402
from pakface.config import ModelSpec, PipelineConfig, default_config  # noqa: E402
from pakface.evaluation import calibrate_and_score  # noqa: E402
from pakface.metrics import compute_eer, confusion_metrics  # noqa: E402
from pakface.pipeline import FacePipeline  # noqa: E402

COLS = ["accuracy", "f1", "precision", "recall", "far", "frr", "eer", "auc", "tar@far=1e-3"]


class ModelCache:
    def __init__(self):
        self.detectors, self.embedders = {}, {}

    def pipeline(self, cfg):
        from pakface.detect import FaceDetector
        from pakface.embedders import build_embedders

        dkey = (cfg.detector_pack, cfg.det_size, cfg.det_thresh, cfg.pad_retry, cfg.upscale_retry, cfg.enhance_for_detection)
        if dkey not in self.detectors:
            self.detectors[dkey] = FaceDetector(cfg.detector_pack, cfg.det_size, cfg.det_thresh, cfg.ctx_id,
                                                cfg.pad_retry, cfg.upscale_retry, cfg.enhance_for_detection)
        embs = []
        for spec in cfg.models:
            k = (spec.name, spec.kind, spec.path, spec.channel_order, spec.finetuned_ckpt)
            if k not in self.embedders:
                one = copy.deepcopy(cfg)
                one.models = [spec]
                self.embedders[k] = build_embedders(one)[0]
            embs.append(self.embedders[k])
        return FacePipeline(cfg, detector=self.detectors[dkey], embedders=embs)


def naive_cfg(base: PipelineConfig) -> PipelineConfig:
    c = copy.deepcopy(base)
    c.models = [ModelSpec(name="arcface_r50_buffalo", kind="insightface_onnx",
                          path="~/.insightface/models/buffalo_l/w600k_r50.onnx")]
    c.pad_retry = c.upscale_retry = c.enhance_for_detection = False
    c.det_thresh = 0.5
    c.use_flip_tta = c.use_enhancement = False
    c.quality_binned_threshold = False
    return c


def off(**kw):
    def f(c):
        for k, v in kw.items():
            setattr(c, k, v)
    return f


def two_store_scores(store_a, store_b, pa, pb, weights):
    ia = np.array([store_a.index[p] for p in pa])
    ib = np.array([store_b.index[p] for p in pb])
    per = {m: np.sum(store_a.emb[m][ia] * store_b.emb[m][ib], axis=1) for m in store_a.emb}
    wsum = sum(weights.get(m, 0) for m in per) or 1.0
    fused = sum(weights.get(m, 0) * s for m, s in per.items()) / wsum
    return fused, np.minimum(store_a.quality[ia], store_b.quality[ib])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pipeline.json")
    ap.add_argument("--pairs-val", default="data/pairs_val.csv")
    ap.add_argument("--pairs-test", default="data/pairs_test.csv")
    ap.add_argument("--finetuned", default="", help="checkpoint from 03_finetune.py")
    ap.add_argument("--method", choices=["max_f1", "max_accuracy", "eer"], default="max_f1",
                    help="threshold objective; max_f1 because F1 is the organisers' ranking metric")
    ap.add_argument("--skip-stress", action="store_true")
    ap.add_argument("--out", default="results/ablation")
    ap.add_argument("--ctx", type=int, default=0)
    args = ap.parse_args()

    base = PipelineConfig.from_json(args.config) if os.path.exists(args.config) else default_config()
    base.ctx_id = args.ctx
    for m in base.models:  # the ablation's FULL row is the pretrained pipeline; fine-tuned gets its own row
        m.finetuned_ckpt = ""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pv, pt = pd.read_csv(args.pairs_val), pd.read_csv(args.pairs_test)
    paths = pd.concat([pv["path_a"], pv["path_b"], pt["path_a"], pt["path_b"]]).tolist()
    cache = ModelCache()

    names = [m.name for m in base.models]
    variants = [
        ("naive baseline (ArcFace R50, no extras)", None, {}),
        ("FULL PIPELINE", off(), {}),
        ("- flip TTA", off(use_flip_tta=False), {}),
        ("- enhancement", off(use_enhancement=False), {}),
        ("- alignment (bbox crop)", off(use_alignment=False), {}),
        ("- detection retry cascade", off(pad_retry=False, upscale_retry=False, enhance_for_detection=False), {}),
        ("- quality-binned thresholds", off(quality_binned_threshold=False), {}),
    ]
    variants += [(f"only {n}", off(), {"weights": {k: float(k == n) for k in names}}) for n in names]
    if args.finetuned:
        def ft(c):
            for m in c.models:
                if m.kind == "adaface_hf":
                    m.finetuned_ckpt = args.finetuned
        variants.append(("+ fine-tuned AdaFace", ft, {}))

    rows, calibs, stores = [], {}, {}
    for name, mutate, extra in variants:
        cfg = naive_cfg(base) if mutate is None else copy.deepcopy(base)
        if mutate is not None:
            mutate(cfg)
        print(f"\n=== {name}")
        store = cache.pipeline(cfg).build_store(paths)
        calib, rep, *_ = calibrate_and_score(store, pv, pt, args.method, cfg.quality_binned_threshold,
                                             weights=extra.get("weights"))
        rows.append({"variant": name, **{k: rep.get(k, np.nan) for k in COLS}, "weights": json.dumps(calib.weights)})
        calibs[name], stores[name] = calib, (cfg, store)
        print({k: round(rep.get(k, np.nan), 5) for k in COLS})

    tab = pd.DataFrame(rows)
    full_acc = tab.loc[tab["variant"] == "FULL PIPELINE", "accuracy"].iloc[0]
    full_eer = tab.loc[tab["variant"] == "FULL PIPELINE", "eer"].iloc[0]
    tab["d_accuracy_vs_full"] = tab["accuracy"] - full_acc
    tab["d_eer_vs_full"] = tab["eer"] - full_eer
    tab.to_csv(out / "ablation.csv", index=False)
    pct = tab.copy()
    for c in COLS:
        pct[c] = pct[c].map(lambda v: f"{v:.4f}") if c == "auc" else (pct[c] * 100).map(lambda v: f"{v:.2f}")
    for c in ["d_accuracy_vs_full", "d_eer_vs_full"]:
        pct[c] = (pct[c] * 100).map(lambda v: f"{v:+.2f}")
    md = ["## Ablation (test identities, %; thresholds fitted on val)", "",
          pct.drop(columns=["weights"]).to_markdown(index=False)]

    if not args.skip_stress:
        srows = []
        pa, pb, y = pt["path_a"].tolist(), pt["path_b"].tolist(), pt["label"].to_numpy()
        for vname in ["naive baseline (ArcFace R50, no extras)", "FULL PIPELINE"]:
            cfg, clean = stores[vname]
            calib = calibs[vname]
            pipe = cache.pipeline(cfg)
            for sname, fn in STRESS_TESTS.items():
                print(f"--- stress {sname} on {vname}")
                deg = pipe.build_store(pb, degrade_fn=fn, tag=f"stress_{sname}")
                s, q = two_store_scores(clean, deg, pa, pb, calib.weights)
                r = confusion_metrics(s, y, calib.thresholds_for(q, len(s)))
                srows.append({"pipeline": "naive" if "naive" in vname else "full", "degradation": sname,
                              "accuracy": r["accuracy"], "far": r["far"], "frr": r["frr"], "eer": compute_eer(s, y)})
        st = pd.DataFrame(srows)
        st.to_csv(out / "stress_tests.csv", index=False)
        piv = st.pivot(index="degradation", columns="pipeline", values=["accuracy", "eer"]) * 100
        piv.columns = [f"{a}_{b} (%)" for a, b in piv.columns]
        md += ["", "## Robustness stress tests (image B degraded; %)", "", piv.round(2).to_markdown()]

    (out / "ablation.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
