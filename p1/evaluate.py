#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate.py

Evaluate a trained checkpoint (single or MoE) on a given list, writing:
    metrics.json      overall accuracy, A_pi, balanced accuracy, Macro-F1, per-class metrics, confusion matrix, per-date metrics,
                      for MoE also the per-class expert assignment matrix
    predictions.csv   true class and predicted class of each image (for MoE also the top-k experts and router probabilities)
    dense_experts.npz only with --dense-experts (MoE, validation set only): logits of the 5 experts for each image, router probabilities, top-k, labels,
                      used by analyze_experts.py to compute the expert function analysis offline (analysis/TODO.md "Expert function analysis after the formal experiments")

The test set is used only for the final report. Evaluating test.csv requires an explicit --final, to prevent accidental use in the pilot stage.

Example usage:
    python evaluate.py --checkpoint runs/final/moe_fuzzy_s1/best.pth \
        --root "~/workspace/dev/mypro/RSCD dataset-1million" --subset-dir subsets \
        --list test.csv --final --out-dir runs/final/moe_fuzzy_s1/test_eval
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import timm
import torch

from common import (ENV_LOCK_DEFAULT, CSVImageDataset, build_transforms, check_env, classification_metrics,
                    default_num_workers, load_manifest, make_loader, mixture, prepare_out_dir, save_json)


def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate a checkpoint")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--subset-dir", required=True)
    ap.add_argument("--list", required=True, help="List filename to evaluate, e.g. val.csv or test.csv")
    ap.add_argument("--final", action="store_true", help="Required when evaluating test.csv to confirm this is the final evaluation")
    ap.add_argument("--dense-experts", action="store_true",
                    help="MoE only, validation lists only: also run all experts on every image and save dense_experts.npz")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=default_num_workers())
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--env-lock", default=str(ENV_LOCK_DEFAULT),
                    help="Environment lock file checked before running; 'none' skips the check (smoke tests only)")
    return ap.parse_args()


def build_model(config, num_classes):
    if config["model"] == "single":
        model = timm.create_model("efficientnet_b0", pretrained=False)
        model.reset_classifier(num_classes)
        return model
    from train_moe import MoEModel
    return MoEModel(num_classes, pretrained=False,
                    use_bias=config["controller"] == "auxfree", tau=config["tau0"],
                    router=config.get("router", "R0"))  # configs before 2026-10-27 have no router field; all are R0


def main():
    args = parse_args()
    if Path(args.list).name.startswith("test") and not args.final:
        raise SystemExit("[Error] Evaluating the test set requires --final. Use only the validation set during pilot experiments.")
    if args.dense_experts and Path(args.list).name.startswith("test"):
        raise SystemExit("[Error] --dense-experts is for the validation set only (Pilot_Study.md 11.5 D3)")
    env = check_env(args.env_lock)  # before creating the output directory
    out_dir = prepare_out_dir(args.out_dir, args.overwrite)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    subset_dir = Path(args.subset_dir).expanduser()
    class_names, pi = load_manifest(subset_dir / "manifest.json")
    if class_names != config["class_names"]:
        raise SystemExit("[Error] The checkpoint class order does not match the current manifest")
    num_classes = len(class_names)

    model = build_model(config, num_classes)
    model.load_state_dict(ckpt["model_state_dict"])  # the MoE tau, noise_scale and route_bias are restored as well
    model.to(device).eval()
    is_moe = config["model"] == "moe"
    cls_loss = config.get("cls_loss", "mix")  # configs before 2026-11-08 have no such field; all are mix
    if is_moe:
        from train_moe import mix_weights, prediction_logits
    if args.dense_experts and not is_moe:
        raise SystemExit("[Error] --dense-experts requires an MoE checkpoint")

    _, eval_tf = build_transforms(config["aug"])
    ds = CSVImageDataset(args.root, subset_dir / args.list, class_names, eval_tf)
    loader = make_loader(ds, args.batch_size, False, args.num_workers, 0)

    ys, ps, idxs, topks, probs_all = [], [], [], [], []
    dense, outs = [], []  # --dense-experts: per-expert logits (B, N_e, C) and the outputs of the normal forward pass (for self-check)
    with torch.no_grad():
        for x, y, i in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                if is_moe:
                    out, probs, topk_idx, expert_logits = model(x)
                    out = prediction_logits(cls_loss, out, expert_logits, mix_weights(probs, topk_idx))  # prediction rule consistent with the training loss
                    topks.append(topk_idx.cpu().numpy())
                    probs_all.append(probs.float().cpu().numpy())
                    if args.dense_experts:  # run the shared features once more and let every expert process the whole batch; the normal forward results are unaffected
                        feats = model.shared_extractor(x)
                        dense.append(torch.stack([ex(feats).float() for ex in model.experts], 1).cpu().numpy())
                        outs.append(out.float().cpu().numpy())
                else:
                    out = model(x)
            ys.append(y.numpy())
            ps.append(out.argmax(1).cpu().numpy())
            idxs.append(i.numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    order = np.concatenate(idxs)

    metrics = classification_metrics(y, p, num_classes, pi)
    metrics.update({"env": env, "checkpoint": str(args.checkpoint), "checkpoint_epoch": ckpt.get("epoch"),
                    "list": args.list, "n": int(len(y)), "class_names": class_names})

    # per-date metrics
    dates = np.array([ds.samples[k][2] for k in order])
    by_date = {}
    for d in sorted(set(dates)):
        sel = dates == d
        dm = classification_metrics(y[sel], p[sel], num_classes, pi)
        by_date[d] = {"n": int(sel.sum()), "acc": dm["acc"], "balanced_acc": dm["balanced_acc"],
                      "macro_f1": dm["macro_f1"], "missing_classes": dm["missing_classes"]}
    metrics["by_date"] = by_date

    if is_moe:
        topk = np.concatenate(topks)
        assign = np.zeros((num_classes, len(model.experts)), dtype=np.int64)
        for j in range(topk.shape[1]):
            np.add.at(assign, (y, topk[:, j]), 1)
        metrics["per_class_expert_assignment"] = assign.tolist()
        metrics["tau"] = float(model.tau)
        metrics["route_bias"] = model.route_bias.cpu().numpy().tolist()
    save_json(metrics, out_dir / "metrics.json")

    if args.dense_experts:
        logits, probs_cat, out_cat = np.concatenate(dense), np.concatenate(probs_all), np.concatenate(outs)
        w = np.take_along_axis(probs_cat, topk, 1)
        w = w / w.sum(1, keepdims=True)
        mix = mixture(logits, topk, w, cls_loss)  # recombine by the normal routing; mixing rule consistent with training
        agree, max_diff = float(np.mean(mix.argmax(1) == p)), float(np.abs(mix - out_cat).max())
        np.savez_compressed(out_dir / "dense_experts.npz", logits=logits.astype(np.float32), probs=probs_cat.astype(np.float32),
                            topk=topk.astype(np.int64), label=y.astype(np.int64), order=order.astype(np.int64),
                            route_bias=model.route_bias.cpu().numpy().astype(np.float32), auxfree=np.array(config["controller"] == "auxfree"),
                            cls_loss=np.array(cls_loss))
        print(f"dense experts: recombined prediction agreement {agree:.4f}, max |logit diff| {max_diff:.4f}")
        if agree < 0.999:
            raise SystemExit(f"[Error] Recombined predictions agree on only {agree:.4f} of images; check the dense forward")

    with open(out_dir / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        header = ["path", "date", "label", "pred"]
        if is_moe:
            header += ["expert_1", "expert_2"] + [f"router_p{e}" for e in range(probs_all[0].shape[1])]
        w.writerow(header)
        probs_cat = np.concatenate(probs_all) if is_moe else None
        for r, k in enumerate(order):
            row = [ds.samples[k][0], ds.samples[k][2], int(y[r]), int(p[r])]
            if is_moe:
                row += [int(v) for v in topk[r]] + [round(float(v), 5) for v in probs_cat[r]]
            w.writerow(row)

    print(f"{args.list}: n={len(y)} acc={metrics['acc']:.4f} A_pi={metrics['acc_pi']:.4f} "
          f"BA={metrics['balanced_acc']:.4f} Macro-F1={metrics['macro_f1']:.4f}")
    for d, v in by_date.items():
        print(f"  {d}: n={v['n']} acc={v['acc']:.4f} BA={v['balanced_acc']:.4f} F1={v['macro_f1']:.4f}")


if __name__ == "__main__":
    main()