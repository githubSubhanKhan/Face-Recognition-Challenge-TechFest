"""Optional presentation-attack (anti-spoofing) check.

Off by default and NEVER applied to benchmark scoring: a benchmark photo of a
person is not a spoof attempt, and rejecting it would only cost accuracy. This
module is for the live demo / deployment story in the presentation.

Expected model: a MiniFASNet-style ONNX export (e.g. from the
minivision-ai Silent-Face-Anti-Spoofing project) that takes an 80x80 crop around
the face (box scaled by ~2.7) and outputs logits for [fake_print, real, fake_replay]
or similar. Verify the class order and input scaling on ~10 live and ~10
printed/screen photos before trusting it; adjust `real_index` / `scale_255`.
"""
from __future__ import annotations

import cv2
import numpy as np


class AntiSpoof:
    def __init__(self, onnx_path: str, input_size: int = 80, box_scale: float = 2.7,
                 real_index: int = 1, scale_255: bool = False):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.size, self.box_scale, self.real_index, self.scale_255 = input_size, box_scale, real_index, scale_255

    def _crop(self, img, bbox):
        h, w = img.shape[:2]
        x1, y1, x2, y2 = bbox[:4]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw, bh = (x2 - x1) * self.box_scale, (y2 - y1) * self.box_scale
        x1, x2 = int(max(0, cx - bw / 2)), int(min(w, cx + bw / 2))
        y1, y2 = int(max(0, cy - bh / 2)), int(min(h, cy + bh / 2))
        return cv2.resize(img[y1:y2, x1:x2], (self.size, self.size))

    def real_probability(self, img_bgr: np.ndarray, bbox) -> float:
        x = self._crop(img_bgr, bbox).astype(np.float32)
        if self.scale_255:
            x /= 255.0
        x = x.transpose(2, 0, 1)[None]
        logits = self.sess.run(None, {self.inp: x})[0][0]
        p = np.exp(logits - logits.max())
        p /= p.sum()
        return float(p[self.real_index])
