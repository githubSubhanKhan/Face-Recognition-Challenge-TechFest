"""Face alignment: 5-point similarity transform onto the standard ArcFace 112x112 template.

All modern recognisers (ArcFace, AdaFace, Glint360K models) were trained on crops
produced with exactly this template, so matching it is worth several points of
accuracy on its own (see the "no alignment" row in the ablation).
"""
from __future__ import annotations

import cv2
import numpy as np

# Canonical landmark positions in a 112x112 crop:
# left eye, right eye, nose tip, left mouth corner, right mouth corner
ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def estimate_transform(kps: np.ndarray, size: int = 112) -> np.ndarray:
    """Similarity transform (rotation + uniform scale + translation) from landmarks to template."""
    dst = ARCFACE_TEMPLATE * (size / 112.0)
    M, _ = cv2.estimateAffinePartial2D(np.asarray(kps, dtype=np.float32), dst, method=cv2.LMEDS)
    if M is None:  # degenerate landmarks: fall back to least squares
        M, _ = cv2.estimateAffinePartial2D(np.asarray(kps, dtype=np.float32), dst)
    return M


def align_face(img: np.ndarray, kps: np.ndarray, size: int = 112) -> np.ndarray:
    M = estimate_transform(kps, size)
    return cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def crop_no_align(img: np.ndarray, bbox: np.ndarray, size: int = 112, expand: float = 0.15) -> np.ndarray:
    """Plain square bbox crop without landmark alignment (used for the ablation and as a fallback)."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * (1 + expand)
    x1, x2 = int(round(cx - side / 2)), int(round(cx + side / 2))
    y1, y2 = int(round(cy - side / 2)), int(round(cy + side / 2))
    pad = max(0, -x1, -y1, x2 - w, y2 - h)
    if pad:
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        x1, x2, y1, y2 = x1 + pad, x2 + pad, y1 + pad, y2 + pad
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        crop = img
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA if crop.shape[0] > size else cv2.INTER_CUBIC)


def whole_image_fallback(img: np.ndarray, size: int = 112) -> np.ndarray:
    """Used when no detector finds a face: assume the image is already a (tight) face crop."""
    h, w = img.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    crop = img[y0 : y0 + side, x0 : x0 + side]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA if side > size else cv2.INTER_CUBIC)
