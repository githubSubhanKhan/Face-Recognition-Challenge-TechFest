"""End-to-end pipeline: image -> detection -> alignment -> quality -> enhancement -> embeddings.

Embeddings are cached per pipeline variant so the ablation study and threshold
tuning never recompute the expensive parts.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .align import align_face, crop_no_align, whole_image_fallback
from .config import PipelineConfig
from .detect import Detection
from .enhance import enhance_crop, needs_enhancement
from .quality import assess


def load_image(path: str) -> np.ndarray:
    """EXIF-aware loader (phone photos are often stored rotated) that always returns BGR uint8."""
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


@dataclass
class FeatureStore:
    paths: list
    emb: dict                      # model name -> (N, D) float32 normalised
    norm: dict                     # model name -> (N,) raw feature norm
    quality: np.ndarray            # (N,) combined quality score
    flags: list                    # per-image list of quality flags
    det_stage: list                # which detection stage succeeded
    index: dict = field(default_factory=dict)

    def __post_init__(self):
        self.index = {p: i for i, p in enumerate(self.paths)}

    def save(self, path: str | Path) -> None:
        arrays = {f"emb__{k}": v for k, v in self.emb.items()}
        arrays.update({f"norm__{k}": v for k, v in self.norm.items()})
        np.savez_compressed(
            path, quality=self.quality, paths=np.array(self.paths),
            flags=np.array([json.dumps(f) for f in self.flags]), det_stage=np.array(self.det_stage), **arrays,
        )

    @classmethod
    def load(cls, path: str | Path) -> "FeatureStore":
        z = np.load(path, allow_pickle=False)
        emb = {k[5:]: z[k] for k in z.files if k.startswith("emb__")}
        norm = {k[6:]: z[k] for k in z.files if k.startswith("norm__")}
        return cls(list(z["paths"]), emb, norm, z["quality"], [json.loads(f) for f in z["flags"]], list(z["det_stage"]))

    def subset_scores(self, a: list[str], b: list[str], weights: dict | None = None):
        """Cosine score per model + weighted fusion for pairs (a[i], b[i])."""
        ia = np.array([self.index[p] for p in a])
        ib = np.array([self.index[p] for p in b])
        per_model = {m: np.sum(e[ia] * e[ib], axis=1) for m, e in self.emb.items()}
        weights = weights or {m: 1.0 for m in per_model}
        wsum = sum(weights.get(m, 0.0) for m in per_model) or 1.0
        fused = sum(weights.get(m, 0.0) * s for m, s in per_model.items()) / wsum
        pair_q = np.minimum(self.quality[ia], self.quality[ib])
        return fused, per_model, pair_q


class FacePipeline:
    def __init__(self, cfg: PipelineConfig, load_models: bool = True, load_embedders: bool = True,
                 detector=None, embedders=None):
        """detector / embedders can be passed in to share loaded models between ablation variants."""
        self.cfg = cfg
        os.makedirs(cfg.cache_dir, exist_ok=True)
        self.det_cache_path = Path(cfg.cache_dir) / f"detections_{self._det_key()}.json"
        self.det_cache = json.loads(self.det_cache_path.read_text()) if self.det_cache_path.exists() else {}
        self.detector = detector
        self.embedders = embedders or []
        if load_models:
            if self.detector is None:
                from .detect import FaceDetector

                self.detector = FaceDetector(cfg.detector_pack, cfg.det_size, cfg.det_thresh, cfg.ctx_id,
                                             cfg.pad_retry, cfg.upscale_retry, cfg.enhance_for_detection)
            if load_embedders and not self.embedders:
                from .embedders import build_embedders

                self.embedders = build_embedders(cfg)

    def save_det_cache(self) -> None:
        self.det_cache_path.write_text(json.dumps(self.det_cache))

    def _det_key(self) -> str:
        c = self.cfg
        return f"{c.detector_pack}_{c.det_size}_{c.det_thresh}_{int(c.pad_retry)}{int(c.upscale_retry)}{int(c.enhance_for_detection)}"

    # ------------------------------------------------------------------
    def detect(self, path: str, img: np.ndarray) -> Detection:
        key = os.path.abspath(path)
        if key in self.det_cache:
            return Detection.from_json(self.det_cache[key])
        det = self.detector.detect(img)
        self.det_cache[key] = det.to_json()
        return det

    def crop(self, img: np.ndarray, det: Detection) -> np.ndarray:
        if det.kps is not None and self.cfg.use_alignment:
            return align_face(img, det.kps)
        if det.stage != "no_detection":
            return crop_no_align(img, det.bbox)
        return whole_image_fallback(img)

    def preprocess(self, path: str, img: np.ndarray | None = None):
        img = load_image(path) if img is None else img
        det = self.detect(path, img)
        crop = self.crop(img, det)
        face_px = float(min(det.bbox[2] - det.bbox[0], det.bbox[3] - det.bbox[1]))
        q = assess(crop, face_px, det.score if det.stage != "no_detection" else 0.3, det.kps)
        flags = list(q.flags) + (["no_detection"] if det.stage == "no_detection" else [])
        return crop, q.score, flags, det.stage

    # ------------------------------------------------------------------
    def store_path(self) -> Path:
        names = "+".join(f"{m.name}{'-ft' + hashlib.md5(m.finetuned_ckpt.encode()).hexdigest()[:6] if m.finetuned_ckpt else ''}"
                         for m in self.cfg.models)
        return Path(self.cfg.cache_dir) / f"features_{self.cfg.variant_key()}_{names}.npz"

    def build_store(self, paths: list[str], batch: int = 128, degrade_fn=None, tag: str = "") -> FeatureStore:
        """Embed every path once (cached). degrade_fn(img)->img is used for robustness stress tests."""
        sp = self.store_path()
        if tag:
            sp = sp.with_name(sp.stem + f"_{tag}.npz")
        paths = list(dict.fromkeys(paths))
        cached = FeatureStore.load(sp) if sp.exists() else None
        todo = [p for p in paths if cached is None or p not in cached.index]

        new = None
        if todo:
            if not self.embedders:
                raise RuntimeError("pipeline created with load_models=False but uncached images were requested")
            embs = {e.name: [] for e in self.embedders}
            norms = {e.name: [] for e in self.embedders}
            quals, flags_all, stages = [], [], []
            for i in tqdm(range(0, len(todo), batch), desc=f"embedding [{self.cfg.variant_key()}]"):
                chunk = todo[i : i + batch]
                crops, enh_idx, enh_crops = [], [], []
                for j, p in enumerate(chunk):
                    img = load_image(p)
                    if degrade_fn is not None:
                        img = degrade_fn(img)
                    crop, q, flags, stage = self.preprocess(p, img) if degrade_fn is None else self._preprocess_nocache(img)
                    crops.append(crop)
                    quals.append(q)
                    flags_all.append(flags)
                    stages.append(stage)
                    if self.cfg.use_enhancement and needs_enhancement(flags):
                        enh_idx.append(j)
                        enh_crops.append(enhance_crop(crop, flags))
                if self.cfg.use_enhancement and not self.cfg.enhancement_fusion:
                    for j, ec in zip(enh_idx, enh_crops):
                        crops[j] = ec
                for e in self.embedders:
                    f, n = e.embed(crops, flip_tta=self.cfg.use_flip_tta)
                    if self.cfg.use_enhancement and self.cfg.enhancement_fusion and enh_crops:
                        fe, _ = e.embed(enh_crops, flip_tta=self.cfg.use_flip_tta)
                        f[enh_idx] = f[enh_idx] + fe
                        f[enh_idx] /= np.linalg.norm(f[enh_idx], axis=1, keepdims=True)
                    embs[e.name].append(f)
                    norms[e.name].append(n)
            new = FeatureStore(
                todo, {k: np.concatenate(v) for k, v in embs.items()}, {k: np.concatenate(v) for k, v in norms.items()},
                np.array(quals, dtype=np.float32), flags_all, stages,
            )
            if degrade_fn is None:
                self.det_cache_path.write_text(json.dumps(self.det_cache))

        store = _merge(cached, new)
        store.save(sp)
        return store

    def _preprocess_nocache(self, img):
        det = self.detector.detect(img)
        crop = self.crop(img, det)
        face_px = float(min(det.bbox[2] - det.bbox[0], det.bbox[3] - det.bbox[1]))
        q = assess(crop, face_px, det.score if det.stage != "no_detection" else 0.3, det.kps)
        flags = list(q.flags) + (["no_detection"] if det.stage == "no_detection" else [])
        return crop, q.score, flags, det.stage


def _merge(a: FeatureStore | None, b: FeatureStore | None) -> FeatureStore:
    if a is None:
        return b
    if b is None:
        return a
    return FeatureStore(
        a.paths + b.paths,
        {k: np.concatenate([a.emb[k], b.emb[k]]) for k in a.emb},
        {k: np.concatenate([a.norm[k], b.norm[k]]) for k in a.norm},
        np.concatenate([a.quality, b.quality]),
        a.flags + b.flags,
        a.det_stage + b.det_stage,
    )
