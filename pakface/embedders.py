"""Recognition backbones.

* AdaFaceHF        - AdaFace IR-101 trained on WebFace12M (CVLFace release on Hugging Face).
                     AdaFace's quality-adaptive margin is designed for low-quality,
                     blurry, low-resolution faces: exactly the hidden-benchmark risk.
* InsightFaceONNX  - any InsightFace ArcFace ONNX model, e.g. glintr100.onnx (antelopev2 pack,
                     R100 on Glint360K) or w600k_r50.onnx (buffalo_l pack).

Both return L2-normalised 512-d embeddings plus the raw feature norm
(for AdaFace the norm is a learned image-quality signal).
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import numpy as np


def _l2n(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n, n[:, 0]


class BaseEmbedder:
    name: str = "base"

    def embed_batch(self, crops: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def embed(self, crops: list[np.ndarray], flip_tta: bool = True, batch_size: int = 64):
        """crops: list of aligned 112x112 BGR uint8 images."""
        feats, norms = [], []
        for i in range(0, len(crops), batch_size):
            chunk = crops[i : i + batch_size]
            f, n = self.embed_batch(chunk)
            if flip_tta:
                ff, nf = self.embed_batch([np.ascontiguousarray(c[:, ::-1]) for c in chunk])
                f, n = f * n[:, None] + ff * nf[:, None], (n + nf) / 2  # norm-weighted fusion
            feats.append(f)
            norms.append(n)
        feats = np.concatenate(feats).astype(np.float32)
        feats, _ = _l2n(feats)
        return feats, np.concatenate(norms).astype(np.float32)


# ---------------------------------------------------------------------------
class InsightFaceONNX(BaseEmbedder):
    def __init__(self, name: str, path: str, ctx_id: int = 0):
        import insightface

        path = os.path.expanduser(path)
        if not os.path.exists(path):
            # antelopev2 sometimes unzips into a nested folder
            hits = glob.glob(os.path.expanduser(f"~/.insightface/models/**/{Path(path).name}"), recursive=True)
            if not hits:
                raise FileNotFoundError(f"{path} not found - run scripts/00_download_models.py first")
            path = hits[0]
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if ctx_id >= 0 else ["CPUExecutionProvider"]
        self.model = insightface.model_zoo.get_model(path, providers=providers)
        self.model.prepare(ctx_id=ctx_id)
        self.name = name

    def embed_batch(self, crops):
        raw = np.asarray(self.model.get_feat(crops), dtype=np.float32)  # handles BGR->RGB + normalisation
        return _l2n(raw)


# ---------------------------------------------------------------------------
def download_hf_repo(repo_id: str, path: str, token: str | None = None) -> None:
    """Mirror of the official CVLFace download helper."""
    from huggingface_hub import hf_hub_download

    os.makedirs(path, exist_ok=True)
    files_txt = os.path.join(path, "files.txt")
    if not os.path.exists(files_txt):
        hf_hub_download(repo_id, "files.txt", token=token, local_dir=path)
    files = [f for f in Path(files_txt).read_text().split("\n") if f]
    for f in files + ["config.json", "wrapper.py", "model.safetensors"]:
        if not os.path.exists(os.path.join(path, f)):
            hf_hub_download(repo_id, f, token=token, local_dir=path)


def load_adaface_hf(path: str, repo_id: str = "", token: str | None = None):
    """Build the CVLFace model the way its wrapper.py does, without AutoModel.

    wrapper.py targets transformers 4.x (it never calls post_init(), which 5.x requires), so we replicate
    it: model.yaml -> get_model -> pretrained_model/model.pt, held under `.model` so state-dict keys (and
    fine-tuned checkpoints) match the original wrapper.
    """
    import torch
    import yaml
    from omegaconf import OmegaConf

    path = os.path.abspath(os.path.expanduser(path))
    if repo_id and not os.path.exists(os.path.join(path, "model.safetensors")):
        download_hf_repo(repo_id, path, token)
    cwd = os.getcwd()
    os.chdir(path)  # CVLFace code resolves yaml_path / pretrained_model/* relative to cwd
    # `from models import get_model` must hit the CVLFace package, not the project root's models/ dir
    # (weights), which Python would otherwise pick up as an empty namespace package.
    sys.path.insert(0, path)
    for name in [m for m in sys.modules if m == "models" or m.startswith("models.")]:
        del sys.modules[name]
    try:
        from models import get_model

        with open("pretrained_model/model.yaml") as f:
            conf = OmegaConf.create(yaml.safe_load(f))
        inner = get_model(conf)
        inner.load_state_dict_from_path("pretrained_model/model.pt")
    finally:
        os.chdir(cwd)
        sys.path.remove(path)

    class CVLFaceRecognitionModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, *args, **kwargs):
            return self.model(*args, **kwargs)

    return CVLFaceRecognitionModel(inner)


def adaface_forward(model, x):
    """CVLFace wrappers may return a tensor, a tuple (feat, norm) or a dict; normalise that."""
    out = model(x)
    if isinstance(out, dict):
        out = out.get("embedding", next(iter(out.values())))
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out


class AdaFaceHF(BaseEmbedder):
    def __init__(self, name: str, path: str, repo_id: str = "", channel_order: str = "rgb",
                 ctx_id: int = 0, finetuned_ckpt: str = ""):
        import torch

        self.torch = torch
        self.device = torch.device(f"cuda:{ctx_id}" if ctx_id >= 0 and torch.cuda.is_available() else "cpu")
        self.model = load_adaface_hf(path, repo_id, os.environ.get("HF_TOKEN"))
        if finetuned_ckpt:
            state = torch.load(finetuned_ckpt, map_location="cpu")
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if missing or unexpected:
                print(f"[{name}] fine-tuned ckpt: {len(missing)} missing / {len(unexpected)} unexpected keys")
        self.model.eval().to(self.device)
        self.channel_order = channel_order
        self.name = name

    def preprocess(self, crops):
        arr = np.stack(crops).astype(np.float32)  # N,H,W,3 BGR
        if self.channel_order == "rgb":
            arr = arr[..., ::-1]
        arr = (arr / 255.0 - 0.5) / 0.5
        return self.torch.from_numpy(np.ascontiguousarray(arr.transpose(0, 3, 1, 2)))

    def embed_batch(self, crops):
        with self.torch.no_grad():
            x = self.preprocess(crops).to(self.device)
            f = adaface_forward(self.model, x).float().cpu().numpy()
        return _l2n(f)


# ---------------------------------------------------------------------------
def build_embedders(cfg) -> list[BaseEmbedder]:
    out = []
    for spec in cfg.models:
        if spec.kind == "adaface_hf":
            out.append(AdaFaceHF(spec.name, spec.path, spec.repo_id, spec.channel_order, cfg.ctx_id, spec.finetuned_ckpt))
        elif spec.kind == "insightface_onnx":
            out.append(InsightFaceONNX(spec.name, spec.path, cfg.ctx_id))
        else:
            raise ValueError(f"unknown model kind {spec.kind}")
    return out
