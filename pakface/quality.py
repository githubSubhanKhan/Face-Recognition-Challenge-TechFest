"""Image-quality assessment for face crops.

Quality is used in four places (never to *reject* a benchmark pair, because a
verification benchmark expects a decision for every pair):
  1. Cleaning training data (drop unusable crops before fine-tuning).
  2. Routing: deciding whether to run low-light / contrast enhancement.
  3. Template building: quality-weighted averaging of gallery images.
  4. Quality-binned thresholds: low-quality pairs get their own calibrated threshold.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass
class QualityReport:
    sharpness: float      # variance of Laplacian on the 112x112 crop (higher = sharper)
    brightness: float     # mean luminance 0..255
    contrast: float       # luminance std
    face_px: float        # min side of the detected face box in the ORIGINAL image (pixels)
    det_score: float      # detector confidence
    yaw_proxy: float      # 0 = frontal, ~1 = profile (from landmark geometry)
    score: float          # combined quality in [0, 1]
    flags: list

    def as_dict(self) -> dict:
        return asdict(self)


def _sigmoid01(x: float, mid: float, scale: float) -> float:
    return float(1.0 / (1.0 + np.exp(-(x - mid) / scale)))


def yaw_from_kps(kps: np.ndarray | None) -> float:
    if kps is None:
        return 0.5
    kps = np.asarray(kps, dtype=np.float32)
    le, re, nose = kps[0], kps[1], kps[2]
    eye_mid = (le + re) / 2
    inter_eye = np.linalg.norm(re - le) + 1e-6
    return float(min(1.0, abs(nose[0] - eye_mid[0]) / (0.5 * inter_eye)))


def assess(crop_bgr: np.ndarray, face_px: float, det_score: float, kps: np.ndarray | None) -> QualityReport:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    bright = float(gray.mean())
    contrast = float(gray.std())
    yaw = yaw_from_kps(kps)

    flags = []
    if sharp < 60:
        flags.append("blurry")
    if bright < 60:
        flags.append("dark")
    if bright > 200:
        flags.append("overexposed")
    if contrast < 25:
        flags.append("low_contrast")
    if face_px < 40:
        flags.append("low_res")
    if yaw > 0.6:
        flags.append("profile")
    if det_score < 0.5:
        flags.append("weak_detection")

    # Each factor mapped to [0,1]; geometric mean so one terrible factor drags the score down.
    factors = [
        _sigmoid01(np.log1p(sharp), np.log1p(80), 0.5),
        1.0 - abs(bright - 127) / 127,
        _sigmoid01(contrast, 30, 8),
        _sigmoid01(face_px, 48, 12),
        float(np.clip(det_score, 0, 1)),
        1.0 - 0.6 * yaw,
    ]
    factors = np.clip(np.array(factors), 0.02, 1.0)
    score = float(np.exp(np.log(factors).mean()))
    return QualityReport(sharp, bright, contrast, float(face_px), float(det_score), yaw, score, flags)
