"""Shared evaluation logic used by 04_evaluate.py and 05_ablation.py."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .matching import Calibration, fused_embeddings, tune_fusion_weights
from .metrics import compute_eer, confusion_metrics, full_report

DEMOGRAPHIC_ATTRS = ["gender", "age_group", "skin_tone", "region", "ethnicity"]
CONDITION_ATTRS = ["photo_type", "pose", "headwear", "glasses", "mask", "facial_hair", "lighting", "device"]
NEGATIVE_VALUES = {"", "none", "no", "0", "false", "nan", "normal", "frontal", "phone"}


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


def identification_report(store, paths: list, identities: list, weights: dict | None = None,
                          threshold: float | None = None, unknown_frac: float = 0.3, seed: int = 0) -> dict:
    """1:N identification ("pick one face out of a whole school") on held-out identities.

    Each person is enrolled with ONE image (their highest-quality one); every other image is a probe.
    Closed set: rank-1 / rank-5 accuracy. Open set (if threshold given): unknown_frac of the people are
    removed from the gallery; DIR = known probes identified correctly AND above threshold,
    FPIR = unknown probes wrongly accepted as someone in the gallery.
    """
    df = pd.DataFrame({"path": paths, "identity": [str(i) for i in identities]}).drop_duplicates("path")
    df["q"] = store.quality[[store.index[p] for p in df["path"]]]
    df = df[df.groupby("identity")["path"].transform("size") >= 2]
    if df["identity"].nunique() < 2:
        return {}
    gal_idx = df.groupby("identity")["q"].idxmax()
    gal, probes = df.loc[gal_idx], df.drop(gal_idx)
    gids, truth = gal["identity"].to_numpy(), probes["identity"].to_numpy()
    sims = fused_embeddings(store, probes["path"].tolist(), weights) @ fused_embeddings(store, gal["path"].tolist(), weights).T
    ranked = gids[np.argsort(-sims, axis=1)]
    rep = {"n_gallery_ids": len(gal), "n_probes": len(probes),
           "rank1": float((ranked[:, 0] == truth).mean()),
           "rank5": float((ranked[:, :5] == truth[:, None]).any(1).mean())}
    if threshold is not None and len(gal) >= 3:
        rng = np.random.default_rng(seed)
        unknown = rng.choice(gids, max(1, int(round(len(gids) * unknown_frac))), replace=False)
        keep, known = ~np.isin(gids, unknown), ~np.isin(truth, unknown)
        s = sims[:, keep]
        best = s.argmax(1)
        accepted = s[np.arange(len(s)), best] > threshold
        rep.update({"open_set_threshold": float(threshold), "n_unknown_ids": len(unknown),
                    "dir": float(((gids[keep][best] == truth) & accepted)[known].mean()),
                    "fpir": float(accepted[~known].mean()) if (~known).any() else float("nan")})
    return rep


def identification_markdown(rep: dict) -> str:
    lines = ["### Identification (1:N) - one enrolment image per person", "", "| metric | value |", "|---|---|",
             f"| gallery identities | {rep['n_gallery_ids']} |", f"| probes | {rep['n_probes']} |",
             f"| rank-1 accuracy | {rep['rank1']*100:.3f}% |", f"| rank-5 accuracy | {rep['rank5']*100:.3f}% |"]
    if "dir" in rep:
        lines += [f"| open-set: unknown identities | {rep['n_unknown_ids']} (threshold {rep['open_set_threshold']:.4f}) |",
                  f"| DIR (known, correctly identified) | {rep['dir']*100:.3f}% |",
                  f"| FPIR (unknown, wrongly accepted) | {rep['fpir']*100:.3f}% |"]
    return "\n".join(lines)


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
