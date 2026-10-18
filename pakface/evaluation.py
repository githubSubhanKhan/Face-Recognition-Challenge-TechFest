"""Shared evaluation logic used by 04_evaluate.py and 05_ablation.py."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .matching import Calibration, tune_fusion_weights
from .metrics import compute_eer, confusion_metrics, full_report

DEMOGRAPHIC_ATTRS = ["gender", "age_group", "skin_tone", "region", "ethnicity"]
CONDITION_ATTRS = ["headwear", "glasses", "mask", "facial_hair", "lighting", "device"]
NEGATIVE_VALUES = {"", "none", "no", "0", "false", "nan", "normal"}


def calibrate_and_score(store, pairs_val: pd.DataFrame, pairs_test: pd.DataFrame, method="max_accuracy",
                        use_bins=True, weights: dict | None = None, tune_weights=True):
    _, pm_val, q_val = store.subset_scores(pairs_val["path_a"].tolist(), pairs_val["path_b"].tolist())
    y_val = pairs_val["label"].to_numpy()
    if weights is None:
        weights = tune_fusion_weights(pm_val, y_val) if tune_weights else {m: 1.0 for m in pm_val}
    s_val, _, _ = store.subset_scores(pairs_val["path_a"].tolist(), pairs_val["path_b"].tolist(), weights)
    calib = Calibration(method=method, use_bins=use_bins, weights=weights).fit(s_val, y_val, q_val)

    s_test, pm_test, q_test = store.subset_scores(pairs_test["path_a"].tolist(), pairs_test["path_b"].tolist(), weights)
    thr = calib.thresholds_for(q_test, len(s_test))
    y_test = pairs_test["label"].to_numpy()
    rep = full_report(s_test, y_test, thr)
    rep["threshold_global"] = calib.global_threshold
    rep["val_eer"] = compute_eer(s_val, y_val)
    return calib, rep, s_test, q_test, thr, pm_test


def _attr_lookup(meta: pd.DataFrame, col: str) -> dict:
    return dict(zip(meta["path"], meta[col].astype(str).str.strip().str.lower()))


def group_breakdown(pairs: pd.DataFrame, scores, thresholds, meta: pd.DataFrame | None, store=None) -> pd.DataFrame:
    """Per-group metrics. Demographic groups: both images in the group. Conditions: either image has it."""
    y = pairs["label"].to_numpy()
    rows = []

    def add(group, value, mask):
        if mask.sum() < 20:
            return
        r = confusion_metrics(scores[mask], y[mask], thresholds[mask])
        r_eer = compute_eer(scores[mask], y[mask]) if 0 < y[mask].sum() < mask.sum() else float("nan")
        rows.append({"group": group, "value": value, "n_pairs": int(mask.sum()), "accuracy": r["accuracy"],
                     "far": r["far"], "frr": r["frr"], "f1": r["f1"], "eer": r_eer})

    if meta is not None:
        for col in DEMOGRAPHIC_ATTRS:
            if col in meta.columns:
                lk = _attr_lookup(meta, col)
                va = pairs["path_a"].map(lk).fillna("nan").to_numpy()
                vb = pairs["path_b"].map(lk).fillna("nan").to_numpy()
                for v in sorted(set(va) - {"nan"}):
                    add(col, v, (va == v) & (vb == v))
        for col in CONDITION_ATTRS:
            if col in meta.columns:
                lk = _attr_lookup(meta, col)
                va = pairs["path_a"].map(lk).fillna("nan").to_numpy()
                vb = pairs["path_b"].map(lk).fillna("nan").to_numpy()
                for v in sorted((set(va) | set(vb)) - NEGATIVE_VALUES):
                    add(col, v, (va == v) | (vb == v))

    if store is not None:  # automatic quality-flag groups, always available
        fl = {p: set(store.flags[i]) for p, i in store.index.items()}
        for flag in ["dark", "blurry", "low_res", "profile", "low_contrast", "no_detection"]:
            m = np.array([flag in fl.get(a, set()) or flag in fl.get(b, set())
                          for a, b in zip(pairs["path_a"], pairs["path_b"])])
            add("auto_quality_flag", flag, m)
    return pd.DataFrame(rows)


def report_markdown(rep: dict, title: str) -> str:
    keys = ["accuracy", "precision", "recall", "f1", "far", "frr", "eer", "auc",
            "tar@far=1e-2", "tar@far=1e-3", "tar@far=1e-4", "threshold_global", "val_eer"]
    lines = [f"### {title}", "", "| metric | value |", "|---|---|"]
    for k in keys:
        if k in rep:
            v = rep[k]
            pct = k not in ("auc", "threshold_global")
            lines.append(f"| {k} | {v*100:.3f}% |" if pct else f"| {k} | {v:.4f} |")
    lines.append(f"| pairs | {rep['n_pairs']} ({rep['n_genuine']} genuine / {rep['n_impostor']} impostor) |")
    lines.append(f"| confusion | TP {rep['tp']}  TN {rep['tn']}  FP {rep['fp']}  FN {rep['fn']} |")
    return "\n".join(lines)
