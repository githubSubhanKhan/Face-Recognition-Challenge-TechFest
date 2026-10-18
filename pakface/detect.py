"""Robust face detection with retry cascade.

Uses the SCRFD detector shipped in InsightFace model packs (a RetinaFace-family
detector with 5-point landmarks). Real benchmark images break detectors in
predictable ways, so we retry instead of giving up:

  1. plain detection
  2. pad the image (tightly-cropped faces have no context -> detector misses them)
  3. upscale tiny images (CCTV / thumbnails)
  4. brighten dark images
  5. last resort: treat the whole image as a face crop (flagged "no_detection")
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .enhance import brighten_for_detection


@dataclass
class Detection:
    bbox: np.ndarray          # x1, y1, x2, y2 in original image coordinates
    kps: np.ndarray | None    # (5, 2) landmarks, None for the whole-image fallback
    score: float
    stage: str                # which retry stage found it

    def to_json(self) -> dict:
        return {
            "bbox": [float(v) for v in self.bbox[:4]],
            "kps": None if self.kps is None else np.asarray(self.kps).tolist(),
            "score": float(self.score),
            "stage": self.stage,
        }

    @classmethod
    def from_json(cls, d: dict) -> "Detection":
        return cls(
            np.array(d["bbox"], dtype=np.float32),
            None if d["kps"] is None else np.array(d["kps"], dtype=np.float32),
            d["score"],
            d["stage"],
        )


class FaceDetector:
    def __init__(self, pack: str = "buffalo_l", det_size: int = 640, det_thresh: float = 0.35,
                 ctx_id: int = 0, pad_retry: bool = True, upscale_retry: bool = True,
                 enhance_retry: bool = True):
        from insightface.app import FaceAnalysis

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if ctx_id >= 0 else ["CPUExecutionProvider"]
        self.app = FaceAnalysis(name=pack, allowed_modules=["detection"], providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_thresh=det_thresh, det_size=(det_size, det_size))
        self.det = self.app.det_model
        self.pad_retry, self.upscale_retry, self.enhance_retry = pad_retry, upscale_retry, enhance_retry

    # ------------------------------------------------------------------
    def _raw(self, img: np.ndarray):
        bboxes, kpss = self.det.detect(img, max_num=0, metric="default")
        if bboxes is None or len(bboxes) == 0:
            return None, None
        return bboxes, kpss

    @staticmethod
    def _pick(bboxes: np.ndarray, kpss: np.ndarray, shape) -> int:
        """Primary face = large, confident and central."""
        h, w = shape[:2]
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        cx = (bboxes[:, 0] + bboxes[:, 2]) / 2 - w / 2
        cy = (bboxes[:, 1] + bboxes[:, 3]) / 2 - h / 2
        center_pen = 1.0 - 0.5 * np.sqrt((cx / w) ** 2 + (cy / h) ** 2)
        return int(np.argmax(areas / areas.max() * bboxes[:, 4] * center_pen))

    def detect(self, img: np.ndarray) -> Detection:
        h, w = img.shape[:2]

        attempts = [("plain", img, 0, 1.0)]
        if self.pad_retry:
            p = int(0.4 * max(h, w))
            padded = cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_CONSTANT, value=(114, 114, 114))
            attempts.append(("padded", padded, p, 1.0))
        if self.upscale_retry and max(h, w) < 400:
            s = 400.0 / max(h, w)
            up = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
            p = int(0.4 * max(up.shape[:2]))
            up = cv2.copyMakeBorder(up, p, p, p, p, cv2.BORDER_CONSTANT, value=(114, 114, 114))
            attempts.append(("upscaled", up, p, s))
        if self.enhance_retry:
            gray_mean = float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())
            if gray_mean < 80:
                br = brighten_for_detection(img)
                p = int(0.4 * max(h, w))
                br = cv2.copyMakeBorder(br, p, p, p, p, cv2.BORDER_CONSTANT, value=(114, 114, 114))
                attempts.append(("brightened", br, p, 1.0))

        for stage, im, pad, scale in attempts:
            bboxes, kpss = self._raw(im)
            if bboxes is None:
                continue
            i = self._pick(bboxes, kpss, im.shape)
            bbox = bboxes[i, :4].astype(np.float32)
            kps = kpss[i].astype(np.float32) if kpss is not None else None
            # map back to original coordinates
            bbox = (bbox - pad) / scale
            if kps is not None:
                kps = (kps - pad) / scale
            return Detection(bbox, kps, float(bboxes[i, 4]), stage)

        return Detection(np.array([0, 0, w, h], dtype=np.float32), None, 0.0, "no_detection")
