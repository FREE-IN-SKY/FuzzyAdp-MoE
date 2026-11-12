#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_models.py: Export best.pth to TorchScript for deployment, and check it against the eager model.

    python export_models.py --checkpoint <run>/best.pth --out <dir>/<name> [--top-k 2|1] [--check-images N] \
        [--root <image root dir> --list <val.csv>]

Outputs <name>.pt (torch.jit.script) and <name>.json (source checkpoint, config summary, export environment, check results).
Check: 8 random inputs + (when --root/--list are given) the first N real images; scripted and eager argmax must all match, max logits difference < 1e-3.
The MoE .pt returns (logits, topk_idx, w); the single model returns logits. Export on the board (recommended, avoids TorchScript cross-version loading issues), or export on the laptop.
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from moe_deploy import build_from_config, eval_transform, load_checkpoint  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True, help="Output prefix, e.g. deploy/models/moe_fixed_s3_top2 (writes .pt and .json)")
    ap.add_argument("--top-k", type=int, default=2, choices=(1, 2), help="MoE only: 2 = training-time rule, 1 = deployment variant")
    ap.add_argument("--check-images", type=int, default=32, help="Real images used for the scripted-vs-eager check (0 = random inputs only)")
    ap.add_argument("--root", default=None, help="Image root for --check-images")
    ap.add_argument("--list", default=None, help="CSV list (path,class_name,class_idx,date) for --check-images")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    config, sd = load_checkpoint(args.checkpoint)
    is_moe = config["model"] == "moe"
    model = build_from_config(config, sd, top_k=args.top_k)
    t0 = time.time()
    scripted = torch.jit.script(model)
    scripted = torch.jit.freeze(scripted) if False else scripted  # no freeze: keep the same ops as eager, for easier checking
    scripted.eval()
    script_s = time.time() - t0

    def run(m, x):
        with torch.no_grad():
            y = m(x)
        return y[0] if isinstance(y, tuple) else y

    # Check 1: random inputs
    torch.manual_seed(0)
    xs = [torch.randn(1, 3, 224, 224) for _ in range(8)]
    max_diff, n_mismatch, n_checked = 0.0, 0, 0
    for x in xs:
        a, b = run(model, x), run(scripted, x)
        max_diff = max(max_diff, float((a - b).abs().max()))
        n_mismatch += int(a.argmax(1).item() != b.argmax(1).item())
        n_checked += 1
    # Check 2: real images
    if args.check_images > 0 and args.root and args.list:
        import csv
        from PIL import Image
        tf = eval_transform()
        with open(args.list, newline="") as f:
            rows = list(csv.DictReader(f))[: args.check_images]
        for r in rows:
            with Image.open(Path(args.root).expanduser() / r["path"]) as img:
                x = tf(img.convert("RGB")).unsqueeze(0)
            a, b = run(model, x), run(scripted, x)
            max_diff = max(max_diff, float((a - b).abs().max()))
            n_mismatch += int(a.argmax(1).item() != b.argmax(1).item())
            n_checked += 1
    ok = n_mismatch == 0 and max_diff < 1e-3

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(out.with_suffix(".pt")))
    meta = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "model": config["model"], "top_k": (args.top_k if is_moe else None),
        "controller": config.get("controller"), "router": config.get("router"), "cls_loss": config.get("cls_loss", "mix"),
        "tau0": config.get("tau0"), "seed": config.get("seed"), "train_list": config.get("train_list"),
        "num_classes": len(config["class_names"]), "class_names": config["class_names"],
        "n_params": int(sum(v.numel() for v in sd.values())),
        "export_env": {"python": platform.python_version(), "torch": torch.__version__, "machine": platform.machine(),
                       "node": platform.node(), "timm": __import__("timm").__version__},
        "script_seconds": round(script_s, 1),
        "check": {"n": n_checked, "argmax_mismatch": n_mismatch, "max_abs_logit_diff": max_diff, "ok": ok},
        "pt_bytes": os.path.getsize(out.with_suffix(".pt")),
    }
    with open(out.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"{out.with_suffix('.pt')}: {config['model']}" + (f" top-{args.top_k}" if is_moe else "") +
          f", {meta['n_params']} params, {meta['pt_bytes'] / 1e6:.1f} MB, scripted in {script_s:.1f}s; "
          f"check n={n_checked} mismatch={n_mismatch} max|diff|={max_diff:.2e} -> {'OK' if ok else 'FAILED'}")
    if not ok:
        raise SystemExit("[Error] Scripted model does not match the eager model; do not deploy this file")


if __name__ == "__main__":
    main()
