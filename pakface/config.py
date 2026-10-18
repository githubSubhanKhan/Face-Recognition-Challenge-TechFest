"""Central configuration for the PakFace pipeline.

Every pipeline stage can be switched on/off from here, which is what makes the
ablation study (scripts/05_ablation.py) a one-liner per variant.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ModelSpec:
    """One recognition backbone in the ensemble."""

    name: str                      # unique key, e.g. "adaface_ir101"
    kind: str                      # "adaface_hf" | "insightface_onnx"
    path: str = ""                 # local dir (adaface_hf) or .onnx file (insightface_onnx)
    repo_id: str = ""              # HF repo for adaface_hf (auto-download)
    channel_order: str = "rgb"     # adaface_hf only: "rgb" or "bgr" (verify with scripts/00_sanity_check.py)
    weight: float = 1.0            # score-fusion weight (tune on val with 05_ablation.py --tune-weights)
    finetuned_ckpt: str = ""       # optional state_dict produced by scripts/03_finetune.py


@dataclass
class PipelineConfig:
    # ---------------- detection ----------------
    detector_pack: str = "buffalo_l"     # insightface model pack providing SCRFD/RetinaFace-style detector
    det_size: int = 640
    det_thresh: float = 0.35             # lower than default 0.5: we prefer a weak face over no face
    pad_retry: bool = True               # retry with border padding (tight crops kill detectors)
    upscale_retry: bool = True           # retry with 2x upscale for tiny images
    enhance_for_detection: bool = True   # retry detection on a brightened image when dark

    # ---------------- alignment ----------------
    use_alignment: bool = True           # 5-point similarity transform to the ArcFace 112x112 template

    # ---------------- quality + enhancement ----------------
    use_enhancement: bool = True         # adaptive low-light / contrast fix on the aligned crop
    enhancement_fusion: bool = True      # embed original AND enhanced crop, average (never loses info)

    # ---------------- embedding ----------------
    use_flip_tta: bool = True
    models: list[ModelSpec] = field(default_factory=list)

    # ---------------- matching ----------------
    fusion: str = "weighted_mean"        # score fusion across models
    quality_binned_threshold: bool = True

    # ---------------- runtime ----------------
    ctx_id: int = 0                      # GPU id, -1 for CPU
    cache_dir: str = "cache"

    # ------------------------------------------------------------------
    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> "PipelineConfig":
        raw = json.loads(Path(path).read_text())
        raw["models"] = [ModelSpec(**m) for m in raw.get("models", [])]
        return cls(**raw)

    def variant_key(self) -> str:
        """Key that changes whenever an option that affects embeddings changes (for caching)."""
        parts = [
            f"al{int(self.use_alignment)}",
            f"en{int(self.use_enhancement)}{int(self.enhancement_fusion)}",
            f"tta{int(self.use_flip_tta)}",
            f"pad{int(self.pad_retry)}up{int(self.upscale_retry)}ed{int(self.enhance_for_detection)}",
            f"dt{self.det_thresh}",
        ]
        return "_".join(parts)


def default_config() -> PipelineConfig:
    """The competition configuration: AdaFace IR-101 (low-quality specialist) + Glint360K R100 ensemble."""
    return PipelineConfig(
        models=[
            ModelSpec(
                name="adaface_ir101",
                kind="adaface_hf",
                repo_id="minchul/cvlface_adaface_ir101_webface12m",
                path="models/adaface_ir101_webface12m",
                channel_order="rgb",
                weight=1.0,
            ),
            ModelSpec(
                name="glint_r100",
                kind="insightface_onnx",
                path="~/.insightface/models/antelopev2/glintr100.onnx",
                weight=1.0,
            ),
        ]
    )
