"""Verification metrics required by the problem statement + plots.

Accuracy, precision, recall, F1, FAR, FRR, EER, ROC/AUC, TAR@FAR, DET.
Convention: label 1 = same person (genuine), 0 = different (impostor);
predict "same" when score > threshold.
"""
from __future__ import annotations

import numpy as np


def roc(scores, labels):
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(np.asarray(labels).astype(int), np.asarray(scores), drop_intermediate=False)
    return fpr, tpr, thr


def compute_eer(scores, labels) -> float:
    fpr, tpr, _ = roc(scores, labels)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    # linear interpolation between the two closest points for a stable estimate
    if i + 1 < len(fpr):
        d0, d1 = fnr[i] - fpr[i], fnr[i + 1] - fpr[i + 1]
        if d0 != d1 and np.sign(d0) != np.sign(d1):
            a = d0 / (d0 - d1)
            return float(fpr[i] + a * (fpr[i + 1] - fpr[i]))
    return float((fpr[i] + fnr[i]) / 2)


def eer_threshold(scores, labels) -> float:
    fpr, tpr, thr = roc(scores, labels)
    i = int(np.nanargmin(np.abs((1 - tpr) - fpr)))
    t = thr[i]
    return float(t if np.isfinite(t) else np.max(scores))


def tar_at_far(scores, labels, far: float) -> float:
    fpr, tpr, _ = roc(scores, labels)
    ok = fpr <= far
    return float(tpr[ok].max()) if ok.any() else 0.0


def confusion_metrics(scores, labels, thresholds) -> dict:
    scores, labels = np.asarray(scores), np.asarray(labels).astype(int)
    thresholds = np.broadcast_to(np.asarray(thresholds, dtype=float), scores.shape)
    pred = (scores > thresholds).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    n = len(labels)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n_pairs": n, "n_genuine": int(labels.sum()), "n_impostor": int(n - labels.sum()),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": (tp + tn) / n if n else 0.0,
        "precision": precision, "recall": recall, "f1": f1,
        "far": fp / (fp + tn) if fp + tn else 0.0,      # false accept rate
        "frr": fn / (fn + tp) if fn + tp else 0.0,      # false reject rate
    }


def full_report(scores, labels, thresholds) -> dict:
    from sklearn.metrics import roc_auc_score

    scores, labels = np.asarray(scores), np.asarray(labels).astype(int)
    rep = confusion_metrics(scores, labels, thresholds)
    if 0 < labels.sum() < len(labels):
        rep.update({
            "auc": float(roc_auc_score(labels, scores)),
            "eer": compute_eer(scores, labels),
            "tar@far=1e-2": tar_at_far(scores, labels, 1e-2),
            "tar@far=1e-3": tar_at_far(scores, labels, 1e-3),
            "tar@far=1e-4": tar_at_far(scores, labels, 1e-4),
        })
    rep["genuine_mean"] = float(scores[labels == 1].mean()) if labels.sum() else float("nan")
    rep["impostor_mean"] = float(scores[labels == 0].mean()) if (labels == 0).sum() else float("nan")
    return rep


# ---------------------------------------------------------------------------
def plot_all(scores, labels, threshold, out_prefix: str, title: str = "") -> list[str]:
    """ROC (log-FAR), DET, and genuine/impostor score histogram. Returns saved paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    scores, labels = np.asarray(scores), np.asarray(labels).astype(int)
    fpr, tpr, _ = roc(scores, labels)
    eer = compute_eer(scores, labels)
    paths = []

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(np.clip(fpr, 1e-5, 1), tpr, lw=2, color="#2563eb", label=f"ROC (EER={eer*100:.2f}%)")
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1)
    ax.set_ylim(0.5, 1.001)
    ax.set_xlabel("False Accept Rate (log)")
    ax.set_ylabel("True Accept Rate")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower right")
    ax.set_title(title or "ROC")
    fig.tight_layout()
    paths.append(f"{out_prefix}_roc.png")
    fig.savefig(paths[-1], dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    fnr = 1 - tpr
    eps = 1e-5
    ax.plot(norm.ppf(np.clip(fpr, eps, 1 - eps)), norm.ppf(np.clip(fnr, eps, 1 - eps)), lw=2, color="#dc2626")
    ticks = [0.001, 0.01, 0.05, 0.2, 0.5]
    ax.set_xticks(norm.ppf(ticks))
    ax.set_xticklabels([f"{t*100:g}%" for t in ticks])
    ax.set_yticks(norm.ppf(ticks))
    ax.set_yticklabels([f"{t*100:g}%" for t in ticks])
    ax.set_xlabel("FAR")
    ax.set_ylabel("FRR")
    ax.grid(alpha=0.3)
    ax.set_title((title + " - " if title else "") + "DET")
    fig.tight_layout()
    paths.append(f"{out_prefix}_det.png")
    fig.savefig(paths[-1], dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(min(scores.min(), -0.2), max(scores.max(), 1.0), 80)
    ax.hist(scores[labels == 0], bins=bins, alpha=0.6, color="#dc2626", label="impostor (different people)", density=True)
    ax.hist(scores[labels == 1], bins=bins, alpha=0.6, color="#16a34a", label="genuine (same person)", density=True)
    thr = float(np.median(np.atleast_1d(threshold)))
    ax.axvline(thr, color="k", ls="--", lw=1.5, label=f"threshold = {thr:.3f}")
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    ax.set_title((title + " - " if title else "") + "score distributions")
    fig.tight_layout()
    paths.append(f"{out_prefix}_hist.png")
    fig.savefig(paths[-1], dpi=160)
    plt.close(fig)
    return paths
