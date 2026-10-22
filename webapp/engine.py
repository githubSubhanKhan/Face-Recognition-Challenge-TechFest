"""Face analysis for the demo app: the same pipeline as predict.py, run on in-memory images.

Webcam captures and uploads never touch the feature / detection caches on disk. Each image goes through
detection (group photos are split into faces, single-person photos keep the retry cascade) -> alignment ->
quality -> enhancement fusion -> AdaFace + ArcFace embeddings, and pairs are scored with the same weighted
cosine fusion and calibrated threshold that the benchmark predictions use.
"""
from __future__ import annotations

import os
import tempfile
import threading
from dataclasses import dataclass

import cv2
import numpy as np

from pakface.config import PipelineConfig, default_config
from pakface.matching import Calibration
from pakface.pipeline import FacePipeline

MODEL_NAMES = {"adaface_ir101": "AdaFace IR-101", "glint_r100": "ArcFace R100 (Glint360K)"}
DEFAULT_THRESHOLD = 0.30  # same fallback as scripts/predict.py when results/calibration.json is missing


class NoFaceError(Exception):
    pass


@dataclass
class Face:
    bbox: np.ndarray       # x1, y1, x2, y2 in the input image
    crop: np.ndarray       # aligned 112x112 BGR, what the backbones actually see
    quality: float
    flags: list
    emb: dict              # model name -> unit embedding


@dataclass
class Match:
    score: float
    threshold: float
    per_model: dict
    faces_a: list
    faces_b: list
    ia: int                # index of the best-matching face in faces_a
    ib: int

    @property
    def same(self) -> bool:
        return self.score > self.threshold


@dataclass
class Sweep:
    query: Face
    crowd: list
    scores: np.ndarray
    thresholds: np.ndarray

    @property
    def matches(self) -> np.ndarray:
        return self.scores > self.thresholds


class FaceEngine:
    def __init__(self, config: str = "configs/pipeline.json", calibration: str = "results/calibration.json",
                 ctx: int = 0):
        cfg = PipelineConfig.from_json(config) if os.path.exists(config) else default_config()
        cfg.ctx_id = ctx
        cfg.cache_dir = os.path.join(tempfile.gettempdir(), "pakface_app")  # never written: everything is in memory
        self.calibrated = os.path.exists(calibration)
        self.calib = (Calibration.load(calibration) if self.calibrated
                      else Calibration(global_threshold=DEFAULT_THRESHOLD, use_bins=False))
        self.pipe = FacePipeline(cfg)
        self.model_names = [e.name for e in self.pipe.embedders]
        self.weights = self.calib.weights or {m: 1.0 for m in self.model_names}
        self.lock = threading.Lock()  # one request at a time on the shared models

    def faces(self, rgb: np.ndarray, multi: bool = True, min_rel: float = 0.25) -> list[Face]:
        """All faces in a group photo (left to right), or the single primary face otherwise."""
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        with self.lock:
            dets = self.pipe.detector.detect_all(img, min_rel) if multi else []
            if len(dets) < 2:
                dets = [self.pipe.detector.detect(img)]
                if dets[0].stage == "no_detection":
                    return []
            parts = [self.pipe.crop_and_assess(img, d) for d in dets]
            embs, _ = self.pipe.embed_crops([c for c, _, _ in parts], [f for _, _, f in parts])
        return [Face(d.bbox, c, q.score, f, {m: e[i] for m, e in embs.items()})
                for i, (d, (c, q, f)) in enumerate(zip(dets, parts))]

    def score(self, a: Face, b: Face) -> tuple[float, dict]:
        per = {m: float(a.emb[m] @ b.emb[m]) for m in self.model_names}
        tot = sum(self.weights.get(m, 0.0) for m in per) or 1.0
        return sum(self.weights.get(m, 0.0) * s for m, s in per.items()) / tot, per

    def threshold(self, a: Face, b: Face) -> float:
        return float(self.calib.thresholds_for(np.array([min(a.quality, b.quality)]), 1)[0])

    def compare(self, rgb_a: np.ndarray, rgb_b: np.ndarray) -> Match:
        """Same person or not; with group photos, the best-matching face combination decides."""
        fa, fb = self.faces(rgb_a), self.faces(rgb_b)
        for faces, name in ((fa, "Face 1"), (fb, "Face 2")):
            if not faces:
                raise NoFaceError(name)
        best = max(((i, j) for i in range(len(fa)) for j in range(len(fb))),
                   key=lambda ij: self.score(fa[ij[0]], fb[ij[1]])[0])
        a, b = fa[best[0]], fb[best[1]]
        s, per = self.score(a, b)
        return Match(s, self.threshold(a, b), per, fa, fb, *best)

    def sweep(self, query_rgb: np.ndarray, crowd_rgb: np.ndarray) -> Sweep:
        """Find the query person among every face of a group photo."""
        q = self.faces(query_rgb, multi=False)
        if not q:
            raise NoFaceError("the face to find")
        crowd = self.faces(crowd_rgb)
        if not crowd:
            raise NoFaceError("the group photo")
        scores = np.array([self.score(q[0], f)[0] for f in crowd])
        thr = np.array([self.threshold(q[0], f) for f in crowd])
        return Sweep(q[0], crowd, scores, thr)
