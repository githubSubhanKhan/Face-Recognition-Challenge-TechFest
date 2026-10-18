"""Careful fine-tuning of AdaFace IR-101 on the Pakistani training identities.

The danger with fine-tuning a model trained on 12M images using a few thousand
images is catastrophic forgetting: it gets better on the training identities and
WORSE on new people (which is all the hidden benchmark contains). Guards used here:

  1. Partial unfreezing      - only the last --unfreeze-frac of the backbone trains.
  2. Frozen BatchNorm        - small batches would corrupt running statistics.
  3. Feature distillation    - loss term keeping embeddings close to the frozen original
                               (Learning-without-Forgetting style), weight --distill.
  4. Warm-started head       - class centres initialised from pretrained embeddings,
                               so early gradients are sensible instead of random.
  5. Tiny backbone LR + cosine decay, few epochs.
  6. Early stopping on VALIDATION IDENTITIES (unseen people) - and the checkpoint is
     only saved if it beats the pretrained model's val EER. If it never does, the
     script tells you to ship the pretrained model; that is a legitimate result to
     report in the ablation.
  7. Heavy degradation + occlusion augmentation (CCTV blur, low light, masks, hijab /
     dupatta, caps, sunglasses) so the adaptation is towards robustness, not memorisation.

    python scripts/03_finetune.py --epochs 6 --batch 64
"""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler  # noqa: E402
from tqdm import tqdm  # noqa: E402

from pakface.augment import train_augment  # noqa: E402
from pakface.config import PipelineConfig, default_config  # noqa: E402
from pakface.embedders import adaface_forward, load_adaface_hf  # noqa: E402
from pakface.heads import AdaFaceHead, ArcFaceHead  # noqa: E402
from pakface.metrics import compute_eer  # noqa: E402


def to_tensor(crop_bgr: np.ndarray, channel_order: str) -> torch.Tensor:
    x = crop_bgr.astype(np.float32)
    if channel_order == "rgb":
        x = x[..., ::-1]
    x = (x / 255.0 - 0.5) / 0.5
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


class CropDataset(Dataset):
    def __init__(self, paths, labels, channel_order, augment=True, seed=0):
        self.paths, self.labels, self.co, self.augment = list(paths), list(labels), channel_order, augment
        self.seed = seed

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.imread(self.paths[i])
        if img.shape[:2] != (112, 112):
            img = cv2.resize(img, (112, 112))
        if self.augment:
            rng = np.random.default_rng([self.seed, i, int(torch.randint(1 << 30, (1,)).item())])
            img = train_augment(img, rng)
        return to_tensor(img, self.co), self.labels[i]


def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)


def set_trainable(model, frac: float) -> int:
    params = list(model.named_parameters())
    for _, p in params:
        p.requires_grad_(False)
    k = max(1, int(len(params) * frac))
    n = 0
    for _, p in params[-k:]:
        p.requires_grad_(True)
        n += p.numel()
    return n


@torch.no_grad()
def embed_paths(model, paths, co, device, bs=128, flip=True):
    model.eval()
    out = []
    for i in range(0, len(paths), bs):
        imgs = [cv2.resize(cv2.imread(p), (112, 112)) for p in paths[i : i + bs]]
        x = torch.stack([to_tensor(im, co) for im in imgs]).to(device)
        f = adaface_forward(model, x).float()
        if flip:
            f = f + adaface_forward(model, torch.flip(x, dims=[3])).float()
        out.append(F.normalize(f, dim=1).cpu())
    return torch.cat(out)


def val_eer(model, pairs, raw2aligned, co, device):
    pa = [raw2aligned[p] for p in pairs["path_a"]]
    pb = [raw2aligned[p] for p in pairs["path_b"]]
    uniq = list(dict.fromkeys(pa + pb))
    E = embed_paths(model, uniq, co, device)
    idx = {p: i for i, p in enumerate(uniq)}
    s = (E[[idx[p] for p in pa]] * E[[idx[p] for p in pb]]).sum(1).numpy()
    return compute_eer(s, pairs["label"].to_numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="data/splits.csv")
    ap.add_argument("--pairs-val", default="data/pairs_val.csv")
    ap.add_argument("--config", default="configs/pipeline.json")
    ap.add_argument("--model-name", default="adaface_ir101")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5, help="backbone LR")
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--unfreeze-frac", type=float, default=0.25)
    ap.add_argument("--distill", type=float, default=1.0, help="weight of the stay-close-to-pretrained loss")
    ap.add_argument("--head", choices=["adaface", "arcface"], default="adaface")
    ap.add_argument("--min-images", type=int, default=3, help="identities with fewer train images are skipped")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = PipelineConfig.from_json(args.config) if os.path.exists(args.config) else default_config()
    spec = next(m for m in cfg.models if m.name == args.model_name)
    co = spec.channel_order
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.splits)
    raw2aligned = dict(zip(df["path"], df["aligned_path"]))
    tr = df[(df["split"] == "train") & (df["train_usable"])]
    counts = tr.groupby("identity").size()
    tr = tr[tr["identity"].isin(counts[counts >= args.min_images].index)]
    classes = sorted(tr["identity"].unique())
    cls_idx = {c: i for i, c in enumerate(classes)}
    labels = tr["identity"].map(cls_idx).to_numpy()
    print(f"train: {len(tr)} images, {len(classes)} identities, device={device}")
    pairs_val = pd.read_csv(args.pairs_val)

    student = load_adaface_hf(spec.path, spec.repo_id, os.environ.get("HF_TOKEN")).to(device)
    teacher = copy.deepcopy(student).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    base_eer = val_eer(student, pairs_val, raw2aligned, co, device)
    print(f"pretrained val EER: {base_eer*100:.3f}%")

    # warm-start class centres from pretrained embeddings
    E = embed_paths(teacher, tr["aligned_path"].tolist(), co, device, flip=False)
    centers = torch.zeros(len(classes), E.shape[1])
    centers.index_add_(0, torch.from_numpy(labels.copy()), E)
    head_cls = AdaFaceHead if args.head == "adaface" else ArcFaceHead
    head = head_cls(E.shape[1], len(classes), init_centers=centers).to(device)

    n_train = set_trainable(student, args.unfreeze_frac)
    print(f"trainable backbone params: {n_train/1e6:.2f}M (last {args.unfreeze_frac*100:.0f}% of tensors)")
    opt = torch.optim.AdamW([
        {"params": [p for p in student.parameters() if p.requires_grad], "lr": args.lr, "weight_decay": 5e-4},
        {"params": head.parameters(), "lr": args.head_lr, "weight_decay": 0.0},
    ])
    # class-balanced sampling: big public identities must not drown the Pakistani ones
    w = 1.0 / np.sqrt(counts.reindex(tr["identity"]).to_numpy())
    sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.double), num_samples=len(tr), replacement=True)
    loader = DataLoader(CropDataset(tr["aligned_path"], labels, co, augment=True, seed=args.seed),
                        batch_size=args.batch, sampler=sampler, num_workers=args.workers, drop_last=True,
                        pin_memory=device.type == "cuda")
    steps = args.epochs * len(loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr, args.head_lr], total_steps=steps,
                                                pct_start=0.1, anneal_strategy="cos")
    try:
        scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    except (AttributeError, TypeError, RuntimeError):
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    os.makedirs(args.out, exist_ok=True)
    best_eer, best_path, history = base_eer, None, [{"epoch": 0, "val_eer": base_eer}]
    for ep in range(1, args.epochs + 1):
        student.train()
        freeze_bn(student)
        pbar = tqdm(loader, desc=f"epoch {ep}/{args.epochs}")
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                raw = adaface_forward(student, x)
                with torch.no_grad():
                    t = F.normalize(adaface_forward(teacher, x).float(), dim=1)
            raw = raw.float()
            norms = raw.norm(dim=1)
            logits = head(raw, norms, y)
            ce = F.cross_entropy(logits, y)
            distill = (1 - (F.normalize(raw, dim=1) * t).sum(1)).mean()
            loss = ce + args.distill * distill
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            pbar.set_postfix(ce=f"{ce.item():.3f}", distill=f"{distill.item():.4f}")
        eer = val_eer(student, pairs_val, raw2aligned, co, device)
        history.append({"epoch": ep, "val_eer": eer})
        print(f"epoch {ep}: val EER {eer*100:.3f}%  (pretrained {base_eer*100:.3f}%, best {best_eer*100:.3f}%)")
        if eer < best_eer:
            best_eer = eer
            best_path = Path(args.out) / f"{args.model_name}_finetuned.pt"
            torch.save(student.state_dict(), best_path)
            print(f"  saved {best_path}")

    Path(args.out, "finetune_history.json").write_text(json.dumps(history, indent=2))
    if best_path is None:
        print("\nFine-tuning never beat the pretrained model on unseen validation identities.\n"
              "Ship the pretrained model, and report this honestly in the ablation (it is a real finding).")
    else:
        print(f"\nBest val EER {best_eer*100:.3f}% vs pretrained {base_eer*100:.3f}%.\n"
              f"Add to configs/pipeline.json -> models[{args.model_name}].finetuned_ckpt = \"{best_path}\"\n"
              "then re-run 04_evaluate.py and 05_ablation.py --finetuned to confirm on the TEST split.")


if __name__ == "__main__":
    main()
