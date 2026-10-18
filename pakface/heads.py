"""Margin-based classification heads for fine-tuning (PyTorch).

AdaFaceHead follows the AdaFace paper (Kim et al., CVPR 2022): the margin adapts to
image quality, using the feature norm as a quality proxy. High-quality hard samples
get pushed harder; low-quality samples are not over-emphasised (they may be unrecognisable).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaFaceHead(nn.Module):
    def __init__(self, emb_dim: int, n_classes: int, m: float = 0.4, h: float = 0.333, s: float = 64.0,
                 t_alpha: float = 0.01, init_centers: torch.Tensor | None = None):
        super().__init__()
        self.kernel = nn.Parameter(torch.empty(emb_dim, n_classes))
        if init_centers is not None:  # warm start: class centres from the pretrained model
            self.kernel.data.copy_(F.normalize(init_centers, dim=1).t())
        else:
            nn.init.normal_(self.kernel, std=0.01)
        self.m, self.h, self.s, self.t_alpha, self.eps = m, h, s, t_alpha, 1e-3
        self.register_buffer("batch_mean", torch.ones(1) * 20)
        self.register_buffer("batch_std", torch.ones(1) * 100)

    def forward(self, emb: torch.Tensor, norms: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        emb = F.normalize(emb, dim=1)
        cos = (emb @ F.normalize(self.kernel, dim=0)).clamp(-1 + self.eps, 1 - self.eps)
        safe = norms.clamp(0.001, 100).detach().view(-1, 1)
        with torch.no_grad():
            mean, std = safe.mean(), safe.std() if safe.numel() > 1 else torch.zeros_like(safe.mean())
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std = std * self.t_alpha + (1 - self.t_alpha) * self.batch_std
        scaler = ((safe - self.batch_mean) / (self.batch_std + self.eps) * self.h).clamp(-1, 1)

        onehot = torch.zeros_like(cos).scatter_(1, labels.view(-1, 1), 1.0)
        # angular margin  g_angle = -m * scaler
        theta = cos.acos()
        theta_m = (theta + onehot * (-self.m * scaler)).clamp(self.eps, math.pi - self.eps)
        cos_m = theta_m.cos()
        # additive margin g_add = m * scaler + m
        cos_m = cos_m - onehot * (self.m + self.m * scaler)
        return cos_m * self.s


class ArcFaceHead(nn.Module):
    def __init__(self, emb_dim: int, n_classes: int, m: float = 0.5, s: float = 64.0,
                 init_centers: torch.Tensor | None = None):
        super().__init__()
        self.kernel = nn.Parameter(torch.empty(emb_dim, n_classes))
        if init_centers is not None:
            self.kernel.data.copy_(F.normalize(init_centers, dim=1).t())
        else:
            nn.init.normal_(self.kernel, std=0.01)
        self.m, self.s = m, s

    def forward(self, emb, norms, labels):
        cos = (F.normalize(emb, dim=1) @ F.normalize(self.kernel, dim=0)).clamp(-1 + 1e-7, 1 - 1e-7)
        onehot = torch.zeros_like(cos).scatter_(1, labels.view(-1, 1), 1.0)
        theta = cos.acos()
        logits = torch.cos((theta + onehot * self.m).clamp(max=math.pi))
        return logits * self.s
