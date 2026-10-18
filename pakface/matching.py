"""Matching: similarity metric, threshold calibration, quality-binned thresholds, identification.

Why cosine similarity: every backbone here is trained with an angular-margin loss
(ArcFace / AdaFace) on L2-normalised features, so identity is encoded in the angle
between embeddings. Cosine is the metric the training objective optimises;
Euclidean distance on normalised vectors is a monotone function of it
(||a-b||^2 = 2 - 2cos), so the two give identical rankings and ROC curves.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .metrics import eer_threshold


def best_accuracy_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Threshold maximising accuracy (exhaustive over sorted midpoints, O(n log n))."""
    order = np.argsort(scores)
    s, y = scores[order], labels[order].astype(int)
    n_pos = y.sum()
    # predict positive if score > t; sweeping t past s[i] moves sample i to the negative side
    fn = np.cumsum(y)                 # positives at or below index i
    tn = np.cumsum(1 - y)             # negatives at or below index i
    correct = tn + (n_pos - fn)
    i = int(np.argmax(correct))
    t = (s[i] + s[i + 1]) / 2 if i + 1 < len(s) else s[i] + 1e-6
    best = correct[i] / len(s)
    # also consider "everything positive"
    if n_pos / len(s) > best:
        return float(s[0] - 1e-6), float(n_pos / len(s))
    return float(t), float(best)


@dataclass
class Calibration:
    method: str = "max_accuracy"        # or "eer"
    global_threshold: float = 0.3
    use_bins: bool = True
    bin_edges: list = field(default_factory=list)       # quality edges, len = n_bins + 1
    bin_thresholds: list = field(default_factory=list)
    weights: dict = field(default_factory=dict)          # per-model fusion weights

    # ------------------------------------------------------------------
    def fit(self, scores, labels, pair_q=None, n_bins: int = 3, min_per_bin: int = 200,
            shrink: float = 0.5) -> "Calibration":
        scores, labels = np.asarray(scores), np.asarray(labels)
        fit_fn = (lambda s, y: best_accuracy_threshold(s, y)[0]) if self.method == "max_accuracy" \
            else (lambda s, y: eer_threshold(s, y))
        self.global_threshold = float(fit_fn(scores, labels))
        self.bin_edges, self.bin_thresholds = [], []
        if self.use_bins and pair_q is not None:
            pair_q = np.asarray(pair_q)
            edges = np.quantile(pair_q, np.linspace(0, 1, n_bins + 1))
            edges[0], edges[-1] = -np.inf, np.inf
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (pair_q > lo) & (pair_q <= hi)
                if m.sum() >= min_per_bin and 0 < labels[m].sum() < m.sum():
                    t = fit_fn(scores[m], labels[m])
                    # shrink towards the global threshold so small bins cannot overfit
                    t = shrink * t + (1 - shrink) * self.global_threshold
                else:
                    t = self.global_threshold
                self.bin_thresholds.append(float(t))
            self.bin_edges = [float(e) for e in edges]
        return self

    def thresholds_for(self, pair_q=None, n: int | None = None) -> np.ndarray:
        if not self.use_bins or not self.bin_edges or pair_q is None:
            return np.full(n if n is not None else len(pair_q), self.global_threshold)
        idx = np.clip(np.searchsorted(self.bin_edges, np.asarray(pair_q), side="left") - 1, 0, len(self.bin_thresholds) - 1)
        return np.array(self.bin_thresholds)[idx]

    def decide(self, scores, pair_q=None) -> np.ndarray:
        scores = np.asarray(scores)
        return (scores > self.thresholds_for(pair_q, len(scores))).astype(int)

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path) -> "Calibration":
        return cls(**json.loads(Path(path).read_text()))


# ---------------------------------------------------------------------------
def tune_fusion_weights(per_model: dict, labels: np.ndarray, grid=np.linspace(0, 1, 11)) -> dict:
    """Grid-search a 2-model fusion weight on validation pairs by EER (lower is better)."""
    from .metrics import compute_eer

    names = list(per_model)
    if len(names) == 1:
        return {names[0]: 1.0}
    if len(names) > 2:
        return {n: 1.0 for n in names}
    best, best_w = 1e9, 0.5
    for w in grid:
        s = w * per_model[names[0]] + (1 - w) * per_model[names[1]]
        e = compute_eer(s, labels)
        if e < best:
            best, best_w = e, w
    return {names[0]: float(best_w), names[1]: float(1 - best_w)}


# ---------------------------------------------------------------------------
def build_templates(emb: np.ndarray, identities: list, quality: np.ndarray | None = None) -> tuple[list, np.ndarray]:
    """One template per identity: quality-weighted mean of its embeddings (renormalised)."""
    ids = sorted(set(identities))
    identities = np.asarray(identities)
    q = np.ones(len(emb)) if quality is None else np.clip(quality, 0.05, 1.0)
    T = []
    for i in ids:
        m = identities == i
        v = (emb[m] * q[m, None]).sum(0)
        T.append(v / (np.linalg.norm(v) + 1e-12))
    return ids, np.stack(T)


def identify(probe_emb: np.ndarray, gallery_ids: list, templates: np.ndarray, threshold: float | None = None):
    """Closed-set (threshold=None) or open-set identification. Returns (pred_ids, best_scores)."""
    sims = probe_emb @ templates.T
    best = sims.argmax(1)
    scores = sims[np.arange(len(sims)), best]
    preds = [gallery_ids[b] for b in best]
    if threshold is not None:
        preds = [p if s > threshold else "unknown" for p, s in zip(preds, scores)]
    return preds, scores
