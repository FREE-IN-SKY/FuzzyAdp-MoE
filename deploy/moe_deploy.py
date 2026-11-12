#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
moe_deploy.py: Model definitions for device deployment (self-contained, does not depend on p1/train_moe.py, no scikit-fuzzy needed on the board).

Corresponds item by item to the inference path of p1/train_moe.py (2026-11-09, code 411e887 / 624f1b0):
    Shared features  SharedFeatureExtractor: conv_stem -> bn1 (activation built in for timm>=0.6) -> blocks[:6]
    Router           CosineRouter (R4): GAP -> Linear(c, 16) -> L2 normalization -> dot product with normalized expert vectors -> cosine scores (no noise at validation)
    Probabilities    softmax(scores / tau); for auxfree, sigmoid(scores / tau), route_bias is added when selecting top-k, weights use only the probabilities
    Selection        top-2 (or deployment variant top-1), weights w = selected probabilities / their sum
    Output           Σ_k w_k · expert_k(shared features)  (prediction rule of cls_loss = mix; checkpoints with other cls_loss are refused for export)
Parameter and buffer names are the same as in the training code; state_dict is loaded with strict=True, so a structure mismatch raises an error directly.
Only R4 is implemented; the formal experiments (runs/final) are all R4. The single model is timm efficientnet_b0 + a 27-class classification head.
"""
from typing import Dict, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_EXPERTS = 5
SPLIT_BLOCK = 6
ROUTER_PROJ_DIM = 16
INPUT_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class SharedFeatureExtractor(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.conv_stem = base.conv_stem
        self.bn1 = base.bn1
        self.act1 = getattr(base, "act1", nn.Identity())
        self.blocks = nn.Sequential(*base.blocks[:SPLIT_BLOCK])

    def forward(self, x):
        return self.blocks(self.act1(self.bn1(self.conv_stem(x))))


class CosineRouter(nn.Module):
    """R4. Returns cosine scores (B, N_e). No noise at deployment (the training code also adds none in eval mode)."""

    def __init__(self, in_channels, num_experts):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(in_channels, ROUTER_PROJ_DIM)
        self.expert_emb = nn.Parameter(torch.empty(ROUTER_PROJ_DIM, num_experts))
        nn.init.normal_(self.expert_emb, 0, 0.01)
        self.register_buffer("noise_scale", torch.tensor(0.1))  # only to align with state_dict, unused at inference

    def forward(self, x):
        h = self.gap(x).flatten(1).float()
        q = F.normalize(self.proj(h), dim=-1)
        k = F.normalize(self.expert_emb, dim=0)
        return q @ k


class ExpertNetwork(nn.Module):
    def __init__(self, base, num_classes):
        super().__init__()
        self.blocks = nn.Sequential(*base.blocks[SPLIT_BLOCK:])
        self.conv_head = base.conv_head
        self.bn2 = base.bn2
        self.act2 = getattr(base, "act2", nn.Identity())
        self.global_pool = base.global_pool
        self.classifier = nn.Linear(base.num_features, num_classes)

    def forward(self, x):
        x = self.act2(self.bn2(self.conv_head(self.blocks(x))))
        return self.classifier(self.global_pool(x))


class MoEDeploy(nn.Module):
    """MoE for deployment. forward returns (logits, topk_idx, w): logits (B, C), topk_idx (B, k), w (B, k).
    k = top_k (2 is the inference rule used in training, 1 is the deployment variant "run only the top-1 expert"). To allow torch.jit.script, the expert loop is unrolled over a fixed
    ModuleList, and each expert uses a mask to check whether it is selected in this batch; with batch 1 each selected expert runs forward exactly once."""

    def __init__(self, num_classes: int, use_bias: bool, tau: float, top_k: int = 2):
        super().__init__()
        assert top_k in (1, 2)
        self.num_classes = num_classes
        self.use_bias = use_bias
        self.top_k = top_k
        base = timm.create_model("efficientnet_b0", pretrained=False)
        self.shared_extractor = SharedFeatureExtractor(base)
        with torch.no_grad():
            c = self.shared_extractor.eval()(torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE)).shape[1]
        self.router = CosineRouter(c, NUM_EXPERTS)
        self.experts = nn.ModuleList([
            ExpertNetwork(timm.create_model("efficientnet_b0", pretrained=False), num_classes) for _ in range(NUM_EXPERTS)
        ])
        self.register_buffer("tau", torch.tensor(float(tau)))
        self.register_buffer("route_bias", torch.zeros(NUM_EXPERTS))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = self.shared_extractor(x)
        scores_raw = self.router(feats)
        if self.use_bias:
            probs = torch.sigmoid(scores_raw / self.tau)
            scores = probs + self.route_bias
        else:
            probs = F.softmax(scores_raw / self.tau, dim=-1)
            scores = probs
        topk_idx = torch.topk(scores, self.top_k, dim=-1).indices  # (B, k)
        w = probs.gather(1, topk_idx)
        w = w / w.sum(dim=-1, keepdim=True)
        mask = torch.zeros_like(probs).scatter(1, topk_idx, w)  # (B, N_e), 0 where not selected
        out = torch.zeros(x.size(0), self.num_classes, dtype=torch.float32)
        for e, expert in enumerate(self.experts):
            m = mask[:, e]
            if bool((m > 0).any()):
                out = out + m.unsqueeze(1) * expert(feats).float()
        return out, topk_idx, w


def build_single(num_classes: int) -> nn.Module:
    model = timm.create_model("efficientnet_b0", pretrained=False)
    model.reset_classifier(num_classes)
    return model


def load_checkpoint(path: str) -> Tuple[Dict, Dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt["config"], ckpt["model_state_dict"]


def build_from_config(config: Dict, state_dict: Dict, top_k: int = 2) -> nn.Module:
    """Build the model from the checkpoint's config and load the weights strictly. The single model ignores top_k."""
    num_classes = len(config["class_names"])
    if config["model"] == "single":
        model = build_single(num_classes)
    else:
        if config.get("router", "R0") != "R4":
            raise SystemExit(f"[Error] Only router R4 is supported for deployment; checkpoint has {config.get('router')}")
        if config.get("cls_loss", "mix") != "mix":
            raise SystemExit(f"[Error] Only cls_loss=mix checkpoints are supported; checkpoint has {config.get('cls_loss')}")
        if config.get("num_experts", NUM_EXPERTS) != NUM_EXPERTS or config.get("split_block", SPLIT_BLOCK) != SPLIT_BLOCK:
            raise SystemExit("[Error] Checkpoint num_experts / split_block differ from the deployment definition")
        model = MoEDeploy(num_classes, use_bias=config["controller"] == "auxfree", tau=config["tau0"], top_k=top_k)
    model.load_state_dict(state_dict, strict=True)
    return model.eval()


def eval_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
