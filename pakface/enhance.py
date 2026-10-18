"""Adaptive, identity-safe enhancement.

Deliberately conservative: only photometric fixes (gamma, CLAHE, mild denoise).
Generative face restoration (GFPGAN / CodeFormer) is NOT used by default because it
hallucinates facial detail and can make two different people look alike, which
raises the false-accept rate. It can be tested as an extra ablation row if time allows.
"""
from __future__ import annotations

import cv2
import numpy as np


def adaptive_gamma(img: np.ndarray, target: float = 0.5) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = float(np.clip(gray.mean() / 255.0, 0.02, 0.98))
    gamma = float(np.clip(np.log(target) / np.log(mean), 0.4, 2.5))
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)


def clahe_lab(img: np.ndarray, clip: float = 2.0, grid: int = 4) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def needs_enhancement(flags: list) -> bool:
    return any(f in flags for f in ("dark", "overexposed", "low_contrast"))


def enhance_crop(crop: np.ndarray, flags: list) -> np.ndarray:
    out = crop
    if "dark" in flags or "overexposed" in flags:
        out = adaptive_gamma(out)
    if "dark" in flags or "low_contrast" in flags or "overexposed" in flags:
        out = clahe_lab(out)
    if "dark" in flags and "low_res" not in flags:
        # brightening amplifies sensor noise; mild NL-means keeps edges
        out = cv2.fastNlMeansDenoisingColored(out, None, 3, 3, 5, 15)
    return out


def brighten_for_detection(img: np.ndarray) -> np.ndarray:
    return clahe_lab(adaptive_gamma(img), clip=2.5, grid=8)
