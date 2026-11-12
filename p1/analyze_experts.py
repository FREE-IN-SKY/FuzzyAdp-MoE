#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_experts.py

Expert function analysis (analysis/TODO.md "Expert function analysis after the formal experiments", Pilot_Study.md 11.3). Only reads the dense_experts.npz
saved by evaluate.py --dense-experts on the validation set; no new forward pass. Results are descriptive only: no thresholds, not used in any verdict.

For each run compute:
    (a) Forced-routing drop: mask the top-1 expert of each image, re-take top-2 among the remaining experts by routing score and renormalize the weights, Δ1 = A_pi(normal) - A_pi(masked);
        masking both top-2 experts gives Δ2. Reference: 2 random experts per image with equal weights (random seed 0, mean over 20 repeats).
        auxfree selects experts by "affinity + bias" and computes weights by affinity, consistent with MoEModel.forward.
    (b) Per-class expert accuracy: each expert alone processes all images, giving a 27-class × 5-expert recall matrix R; the dominant expert e*(c) of each class is the expert with the most top-1 assignments,
        gain(c) = R[c, e*(c)] - mean of R[c, e] over the other experts; report the π-weighted mean gain and the number of classes with gain > 0.
    (c) Inter-expert similarity: pairwise linear CKA of the logits of the 5 experts (mean, minimum); mean pairwise prediction disagreement rate (fraction of images with different argmax).
Then summarize the mean and range over the 3 seeds per control method; the seed-paired differences against fixed are descriptive only.

Usage (under p1/):
    python analyze_experts.py --runs-dir ../runs/final --names <run directory names...> --out ../runs/final/experts.json
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from analyze_calib import load_pi
from common import classification_metrics, mixture

KEYS = ("acc_pi_normal", "delta1", "delta2", "acc_pi_random", "normal_minus_random", "gain_pi_mean", "n_gain_pos",
        "cka_mean", "cka_min", "disagree_mean")


def acc_pi(y, pred, pi):
    return classification_metrics(y, pred, len(pi), pi)["acc_pi"]


def route(probs, scores, k=2):
    """Take top-k by scores (masked experts have scores -inf); the weights are probs renormalized over the selected experts."""
    sel = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    w = np.take_along_axis(probs, sel, 1)
    return sel, w / w.sum(1, keepdims=True)


def linear_cka(x, y):
    x = x - x.mean(0)
    y = y - y.mean(0)
    return float(np.linalg.norm(y.T @ x) ** 2 / (np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)))


def analyze(npz_path, pi, reps=20):
    d = np.load(npz_path)
    logits, probs, topk, y = d["logits"].astype(np.float64), d["probs"].astype(np.float64), d["topk"], d["label"]
    n, n_e, n_c = logits.shape
    mode = str(d["cls_loss"]) if "cls_loss" in d.files else "mix"  # mixing rule (--cls-loss); old npz files lack this field and are all mix
    scores = probs + d["route_bias"][None, :] if bool(d["auxfree"]) else probs.copy()
    rows = np.arange(n)
    w = np.take_along_axis(probs, topk, 1)
    a_norm = acc_pi(y, mixture(logits, topk, w / w.sum(1, keepdims=True), mode).argmax(1), pi)
    out = {"n": int(n), "acc_pi_normal": a_norm}
    # (a) re-route after masking top-1 / top-2
    for k, key in ((1, "delta1"), (2, "delta2")):
        s = scores.copy()
        for j in range(k):
            s[rows, topk[:, j]] = -np.inf
        sel, ww = route(probs, s)
        out[key] = a_norm - acc_pi(y, mixture(logits, sel, ww, mode).argmax(1), pi)
    rng = np.random.default_rng(0)
    rand = []
    for _ in range(reps):
        sel = np.argsort(rng.random((n, n_e)), axis=1)[:, :2]
        rand.append(acc_pi(y, mixture(logits, sel, np.full((n, 2), 0.5), mode).argmax(1), pi))
    out["acc_pi_random"] = float(np.mean(rand))
    out["normal_minus_random"] = a_norm - out["acc_pi_random"]
    # (b) per-class expert accuracy and its consistency with routing
    single = logits.argmax(2)  # (N, E)
    recall = np.full((n_c, n_e), np.nan)
    for c in range(n_c):
        m = y == c
        if m.any():
            recall[c] = (single[m] == c).mean(0)
    top1 = np.zeros((n_c, n_e), dtype=np.int64)
    np.add.at(top1, (y, topk[:, 0]), 1)
    present = top1.sum(1) > 0
    gain = np.full(n_c, np.nan)
    for c in np.where(present)[0]:
        e = int(top1[c].argmax())
        gain[c] = recall[c, e] - np.mean(np.delete(recall[c], e))
    out["gain_pi_mean"] = float(np.nansum(pi * gain) / pi[present].sum())
    out["n_gain_pos"] = int(np.sum(gain[present] > 0))
    out["n_classes"] = int(present.sum())
    out["recall_matrix"] = np.round(recall, 4).tolist()
    out["dominant_expert"] = [int(top1[c].argmax()) if present[c] else None for c in range(n_c)]
    # (c) inter-expert similarity
    cka = [linear_cka(logits[:, i], logits[:, j]) for i, j in combinations(range(n_e), 2)]
    out["cka_mean"], out["cka_min"] = float(np.mean(cka)), float(np.min(cka))
    out["disagree_mean"] = float(np.mean([np.mean(single[:, i] != single[:, j]) for i, j in combinations(range(n_e), 2)]))
    return out


def main():
    ap = argparse.ArgumentParser(description="Expert functional analysis from dense_experts.npz (descriptive only)")
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--names", nargs="+", required=True, help="Run directory names under --runs-dir (MoE runs)")
    ap.add_argument("--eval-subdir", default="val_dense_best", help="Subdirectory holding dense_experts.npz")
    ap.add_argument("--subset-dir", default="../datasets/subsets")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    runs, groups = {}, {}
    for name in args.names:
        rd = Path(args.runs_dir) / name
        cfg = json.loads((rd / "config.json").read_text(encoding="utf-8"))
        if cfg["model"] != "moe":
            raise SystemExit(f"[Error] {name} is not an MoE run")
        r = analyze(rd / args.eval_subdir / "dense_experts.npz", load_pi(cfg, args.subset_dir))
        r.update(controller=cfg["controller"], seed=int(cfg["seed"]))
        runs[name] = r
        groups.setdefault(cfg["controller"], {})[int(cfg["seed"])] = r
        print(f"{name}: A_pi {r['acc_pi_normal']:.4f}; drop when masking top-1 {r['delta1'] * 100:+.2f} pp, top-2 {r['delta2'] * 100:+.2f} pp; "
              f"random 2 {r['acc_pi_random']:.4f}; gain {r['gain_pi_mean'] * 100:+.2f} pp ({r['n_gain_pos']}/{r['n_classes']} classes > 0); "
              f"CKA mean {r['cka_mean']:.3f}, min {r['cka_min']:.3f}; disagreement {r['disagree_mean']:.3f}")
    summary = {}
    for ctrl, by_seed in groups.items():
        seeds = sorted(by_seed)
        summary[ctrl] = {"seeds": seeds, **{k: {"mean": float(np.mean([by_seed[s][k] for s in seeds])),
                                                "min": float(np.min([by_seed[s][k] for s in seeds])),
                                                "max": float(np.max([by_seed[s][k] for s in seeds]))} for k in KEYS}}
        if ctrl != "fixed" and "fixed" in groups:  # seed-paired difference against fixed, descriptive only
            common_seeds = [s for s in seeds if s in groups["fixed"]]
            summary[ctrl]["paired_vs_fixed"] = {k: {s: by_seed[s][k] - groups["fixed"][s][k] for s in common_seeds} for k in KEYS}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "runs": runs, "summary": summary}, f, ensure_ascii=False, indent=2)
    print(f"Results written to {args.out} (descriptive only; no thresholds)")


if __name__ == "__main__":
    main()
