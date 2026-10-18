"""Degradation and occlusion augmentation (pure OpenCV/NumPy, no extra deps).

Two uses:
  * training-time augmentation on aligned 112x112 crops (fine-tuning), so the model
    sees CCTV blur, low light, masks, hijab/dupatta, caps, sunglasses;
  * test-time stress tests on full images (robustness tables for the report).

Occlusions are placed using the fixed ArcFace template landmark positions, which is
why they are only applied in crop space.
"""
from __future__ import annotations

import cv2
import numpy as np

# ---------------------------------------------------------------- photometric / sensor


def gaussian_blur(img, rng, lo=0.6, hi=3.0):
    s = rng.uniform(lo, hi)
    return cv2.GaussianBlur(img, (0, 0), s)


def motion_blur(img, rng, lo=5, hi=15):
    k = int(rng.integers(lo, hi + 1)) | 1
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1.0)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    kernel /= kernel.sum() + 1e-8
    return cv2.filter2D(img, -1, kernel)


def downscale(img, rng, lo=14, hi=40):
    """Simulate a low-resolution face: shrink so the long side is lo..hi px, then upsample back."""
    h, w = img.shape[:2]
    target = rng.uniform(lo, hi)
    s = target / max(h, w)
    if s >= 1:
        return img
    small = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    interp = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_NEAREST][int(rng.integers(0, 3))]
    return cv2.resize(small, (w, h), interpolation=interp)


def jpeg(img, rng, lo=8, hi=45):
    q = int(rng.integers(lo, hi + 1))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img


def noise(img, rng, lo=3, hi=18):
    n = rng.normal(0, rng.uniform(lo, hi), img.shape)
    return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)


def low_light(img, rng):
    g = rng.uniform(1.6, 3.2)
    out = ((img.astype(np.float32) / 255.0) ** g) * 255 * rng.uniform(0.5, 0.9)
    out = np.clip(out + rng.normal(0, rng.uniform(2, 8), img.shape), 0, 255)
    return out.astype(np.uint8)


def overexpose(img, rng):
    g = rng.uniform(0.35, 0.7)
    return np.clip(((img.astype(np.float32) / 255.0) ** g) * 255 * rng.uniform(1.0, 1.2), 0, 255).astype(np.uint8)


def color_cast(img, rng):
    gains = rng.uniform(0.75, 1.25, 3)
    return np.clip(img.astype(np.float32) * gains, 0, 255).astype(np.uint8)


def grayscale(img, rng=None):
    return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def cctv(img, rng):
    """Composite CCTV look: optional IR grayscale, low-res, blur, compression and sensor noise."""
    out = img
    if rng.random() < 0.5:
        out = grayscale(out)
    out = downscale(out, rng, 18, 36) if min(out.shape[:2]) <= 160 else downscale(out, rng, 60, 120)
    if rng.random() < 0.5:
        out = gaussian_blur(out, rng, 0.5, 1.5)
    out = noise(out, rng, 3, 12)
    return jpeg(out, rng, 10, 35)


# ---------------------------------------------------------------- occlusions (crop space, 112x112 template)


def _sc(pts, size):
    return (np.array(pts, dtype=np.float32) * size / 112.0).astype(np.int32)


def _fabric_color(rng):
    palette = [(235, 235, 235), (30, 30, 30), (200, 160, 110), (60, 40, 140), (40, 90, 160),
               (120, 60, 30), (90, 30, 120), (20, 110, 60), (180, 180, 60), (150, 150, 200)]
    c = np.array(palette[int(rng.integers(0, len(palette)))], dtype=np.float32)
    return tuple(int(v) for v in np.clip(c + rng.normal(0, 15, 3), 0, 255))


def _textured_fill(img, mask, color, rng):
    layer = np.full_like(img, color)
    if rng.random() < 0.5:  # simple print / fold shading
        h, w = img.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        stripes = (np.sin((xx * rng.uniform(0.2, 0.6) + yy * rng.uniform(0, 0.4))) * rng.uniform(5, 25))
        layer = np.clip(layer.astype(np.float32) + stripes[..., None], 0, 255).astype(np.uint8)
    out = img.copy()
    out[mask > 0] = layer[mask > 0]
    return out


def face_mask(img, rng):
    s = img.shape[0]
    top = rng.uniform(62, 74)
    poly = _sc([[14, top], [56, top - 4], [98, top], [102, 96], [82, 113], [30, 113], [10, 96]], s)
    m = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(m, [poly], 255)
    col = [(200, 170, 120), (235, 235, 235), (30, 30, 30), (150, 150, 200)][int(rng.integers(0, 4))]
    return _textured_fill(img, m, col, rng)


def sunglasses(img, rng):
    s = img.shape[0]
    out = img.copy()
    col = tuple(int(v) for v in rng.integers(5, 45, 3))
    for cx, cy in [(38.3, 51.7), (73.5, 51.5)]:
        c = tuple(int(v) for v in _sc([[cx, cy]], s)[0])
        ax = tuple(int(v) for v in _sc([[rng.uniform(11, 15), rng.uniform(8, 11)]], s)[0])
        cv2.ellipse(out, c, ax, 0, 0, 360, col, -1)
    cv2.line(out, tuple(_sc([[50, 50]], s)[0]), tuple(_sc([[62, 50]], s)[0]), col, max(1, s // 56))
    return out


def head_cover(img, rng):
    """Hijab / dupatta / turban style: fabric over the forehead and down the sides of the face."""
    s = img.shape[0]
    top = rng.uniform(22, 36)       # how low the fabric comes over the forehead
    side = rng.uniform(6, 18)       # how much of the cheeks/temples it hides
    left = [[0, 0], [112, 0], [112, 112], [112 - side, 112], [112 - side * 0.6, 60], [96, top], [56, top - 6],
            [16, top], [side * 0.6, 60], [side, 112], [0, 112]]
    m = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(m, [_sc(left, s)], 255)
    return _textured_fill(img, m, _fabric_color(rng), rng)


def cap(img, rng):
    s = img.shape[0]
    out = img.copy()
    brim = rng.uniform(18, 32)
    m = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(m, [_sc([[0, 0], [112, 0], [112, brim], [56, brim + 4], [0, brim]], s)], 255)
    out = _textured_fill(out, m, _fabric_color(rng), rng)
    # visor shadow over the eyes
    y0, y1 = int(brim * s / 112), int(min(112, brim + rng.uniform(15, 30)) * s / 112)
    out[y0:y1] = (out[y0:y1].astype(np.float32) * rng.uniform(0.45, 0.75)).astype(np.uint8)
    return out


def random_erase(img, rng):
    s = img.shape[0]
    out = img.copy()
    w, h = int(rng.uniform(0.15, 0.4) * s), int(rng.uniform(0.15, 0.4) * s)
    x, y = int(rng.integers(0, s - w)), int(rng.integers(0, s - h))
    out[y : y + h, x : x + w] = rng.integers(0, 255, 3)
    return out


PHOTOMETRIC = [gaussian_blur, motion_blur, downscale, jpeg, noise, low_light, overexpose, color_cast, grayscale, cctv]
OCCLUSIONS = [face_mask, sunglasses, head_cover, cap, random_erase]


def train_augment(crop: np.ndarray, rng: np.random.Generator, p_degrade: float = 0.6, p_occlude: float = 0.3,
                  p_flip: float = 0.5) -> np.ndarray:
    out = crop
    if rng.random() < p_flip:
        out = np.ascontiguousarray(out[:, ::-1])
    if rng.random() < p_occlude:
        out = OCCLUSIONS[int(rng.integers(0, len(OCCLUSIONS)))](out, rng)
    if rng.random() < p_degrade:
        for _ in range(int(rng.integers(1, 3))):
            out = PHOTOMETRIC[int(rng.integers(0, len(PHOTOMETRIC)))](out, rng)
    return out


# ---------------------------------------------------------------- stress tests on full images
def _stress(fn, **kw):
    def f(img, seed=0):
        return fn(img, np.random.default_rng(seed), **kw)
    return f


STRESS_TESTS = {
    "blur_sigma2": lambda img: cv2.GaussianBlur(img, (0, 0), 2.0 * max(img.shape[:2]) / 250),
    "motion_blur": _stress(motion_blur, lo=11, hi=11),
    "lowres_long_side_48": lambda img: downscale(img, np.random.default_rng(0), 48, 48),
    "lowres_long_side_32": lambda img: downscale(img, np.random.default_rng(0), 32, 32),
    "low_light": _stress(low_light),
    "jpeg_q10": lambda img: jpeg(img, np.random.default_rng(0), 10, 10),
    "cctv": _stress(cctv),
    "grayscale": lambda img: grayscale(img),
}
