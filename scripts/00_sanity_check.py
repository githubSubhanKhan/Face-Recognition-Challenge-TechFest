"""Five-minute sanity check before anything else.

1. Confirms the GPU is used by ONNX Runtime and PyTorch.
2. Settles AdaFace's input channel order (RGB vs BGR) empirically: the right one gives a
   clearly larger genuine/impostor gap. Writes the winner into configs/pipeline.json.
3. Prints a quick EER for each backbone on a small sample of your data.

    python scripts/00_sanity_check.py --data data/raw --ids 40
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from pakface.config import PipelineConfig, default_config  # noqa: E402
from pakface.data import make_pairs, scan  # noqa: E402
from pakface.metrics import compute_eer  # noqa: E402


def dprime(s, y):
    g, i = s[y == 1], s[y == 0]
    return (g.mean() - i.mean()) / np.sqrt(0.5 * (g.var() + i.var()) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--ids", type=int, default=40)
    ap.add_argument("--config", default="configs/pipeline.json")
    ap.add_argument("--ctx", type=int, default=0)
    args = ap.parse_args()

    import onnxruntime as ort
    import torch

    print("onnxruntime providers:", ort.get_available_providers())
    print("torch cuda:", torch.cuda.is_available())

    df = scan(args.data)
    keep = df.groupby("identity").size()
    keep = keep[keep >= 3].index[: args.ids]
    df = df[df["identity"].isin(keep)]
    pairs = make_pairs(df, n_genuine_per_id=6, impostor_ratio=1.0, hard_negative_cols=())
    print(f"sample: {len(keep)} identities, {len(pairs)} pairs")

    cfg = PipelineConfig.from_json(args.config) if os.path.exists(args.config) else default_config()
    cfg.ctx_id = args.ctx
    cfg.cache_dir = "cache/sanity"

    from pakface.pipeline import FacePipeline

    results = {}
    for order in ["rgb", "bgr"]:
        for m in cfg.models:
            if m.kind == "adaface_hf":
                m.channel_order = order
        cfg.cache_dir = f"cache/sanity_{order}"
        pipe = FacePipeline(cfg)
        store = pipe.build_store(pairs["path_a"].tolist() + pairs["path_b"].tolist())
        _, per_model, _ = store.subset_scores(pairs["path_a"].tolist(), pairs["path_b"].tolist())
        y = pairs["label"].to_numpy()
        for name, s in per_model.items():
            results[(order, name)] = (dprime(s, y), compute_eer(s, y))
            print(f"[{order}] {name:16s} d'={results[(order, name)][0]:.2f}  EER={results[(order, name)][1]*100:.2f}%")

    ada = [m for m in cfg.models if m.kind == "adaface_hf"]
    if ada:
        best = max(["rgb", "bgr"], key=lambda o: results[(o, ada[0].name)][0])
        for m in ada:
            m.channel_order = best
        print(f"\n==> AdaFace channel order: {best.upper()} (written to {args.config})")
    cfg.cache_dir = "cache"
    os.makedirs(os.path.dirname(args.config), exist_ok=True)
    cfg.to_json(args.config)


if __name__ == "__main__":
    main()
