#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose_speed.py

Determine whether the training speed bottleneck is data loading or GPU compute. Does not train and writes no model files; only prints throughput and GPU memory.

1. Data loading only (no model), comparing throughput for different num_workers
2. GPU compute only (random tensors instead of images, no data reading), comparing cudnn deterministic mode on and off
3. Record peak GPU memory with torch

Usage:
    python diagnose_speed.py --root "~/workspace/dev/mypro/RSCD dataset-1million" --subset-dir subsets
"""

import argparse
import time
from pathlib import Path

import timm
import torch
import torch.nn.functional as F

from common import CSVImageDataset, build_transforms, load_manifest, make_loader


def loader_throughput(ds, batch_size, workers, n_batches):
    loader = make_loader(ds, batch_size, True, workers, 0)
    it = iter(loader)
    next(it)  # warm-up: start the workers
    t = time.time()
    n = 0
    for i, (x, _, _) in enumerate(it):
        n += x.size(0)
        if i + 1 >= n_batches:
            break
    dt = time.time() - t
    del it, loader
    return n / dt


def gpu_throughput(model, batch_size, n_steps, moe):
    device = torch.device("cuda")
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    x = torch.randn(batch_size, 3, 224, 224, device=device)
    y = torch.randint(0, 27, (batch_size,), device=device)

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda"):
            out = model(x)
            if moe:
                out = out[0]
            loss = F.cross_entropy(out.float(), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    for _ in range(5):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    for _ in range(n_steps):
        step()
    torch.cuda.synchronize()
    dt = time.time() - t
    return batch_size * n_steps / dt, torch.cuda.max_memory_allocated() / 2 ** 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--subset-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", default="4,8,14,20")
    ap.add_argument("--n-batches", type=int, default=150)
    ap.add_argument("--n-steps", type=int, default=40)
    args = ap.parse_args()

    subset = Path(args.subset_dir).expanduser()
    class_names, _ = load_manifest(subset / "manifest.json")
    train_tf, _ = build_transforms("B")
    ds = CSVImageDataset(args.root, subset / "train_s0.csv", class_names, train_tf)
    n_train = len(ds)

    print("== 1. Data loading throughput (augmentation B, data loading only, no model)")
    print("   Each worker count uses the same seeded sample order; later reads may benefit from the system cache")
    for w in [int(s) for s in args.workers.split(",")]:
        ips = loader_throughput(ds, args.batch_size, w, args.n_batches)
        print(f"   num_workers={w:>2d}: {ips:7.0f} images/s  -> estimated data loading time per epoch: {n_train / ips:6.1f} s")

    if not torch.cuda.is_available():
        print("CUDA is unavailable; skipping GPU tests")
        return
    print(f"\n== 2. GPU compute throughput (random tensors, batch={args.batch_size}, {torch.cuda.get_device_name(0)})")
    from train_moe import MoEModel
    for det in (True, False):
        torch.backends.cudnn.deterministic = det
        torch.backends.cudnn.benchmark = not det
        label = "Deterministic mode (current setting)" if det else "Nondeterministic mode + benchmark"
        single = timm.create_model("efficientnet_b0", pretrained=False, num_classes=27)
        ips, mem = gpu_throughput(single, args.batch_size, args.n_steps, moe=False)
        print(f"   Single, {label}: {ips:6.0f} images/s -> estimated time per epoch: {n_train / ips:6.1f} s, peak GPU memory {mem:6.0f} MiB")
        del single
        torch.cuda.empty_cache()
        moe = MoEModel(27, pretrained=False)
        ips, mem = gpu_throughput(moe, args.batch_size, args.n_steps, moe=True)
        print(f"   MoE,  {label}: {ips:6.0f} images/s -> estimated time per epoch: {n_train / ips:6.1f} s, peak GPU memory {mem:6.0f} MiB")
        del moe
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
