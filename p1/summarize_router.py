#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
summarize_router.py

Summarize pilot stage 5 (non-inferiority check of R4 vs R0, analysis/Pilot_Plan.md) and give conclusions by pre-set rules. Reads run directories only;
writes results only when --out is given.

Run identification reuses summarize_pilot.load_runs (by config.json fields; an old config without the router field is treated as R0); the setting is fixed to
augmentation B, lr 1e-3, wd 1e-4, 70 epoch. R0 seed 0 is taken from --repro-dir (reproduction run), the rest from --router-dir.

Rules (fixed before the stage 5 runs):
- Compute seed-paired d_k = A_R4,k − A_R0,k (k = 0, 1, 2), where A is the validation A_pi at the best epoch
- mean d_k >= −X: confirm adopting R4; mean d_k < −X: do not fall back to R0, stop and discuss (check the routing diagnostics first)
- X is read from the stage 0 threshold.json; it is a decision rule, not a significance bound

Report (not used for selection):
- BA, Macro-F1, mean A_pi over the last 10 epochs
- minimum weighted usage within the last 10 epochs, normalized mutual information (assignment matrix of last.pth on val.csv), cos gap p50, w1 (last epoch)
- Second-expert check: minimum of each expert's top-1 share (π-weighted, computed from val_routing.npz) within the last 10 epochs and minimum of the w share
- router parameter count (router_params in the config)

Usage (in p1/):
    python summarize_router.py --router-dir ../runs/router --repro-dir ../runs/repro \
        --threshold ../runs/pilot/threshold.json --lambda0 1.5 --sigma0 0.2 [--out ../runs/router/stage5.json]
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from analyze_calib import load_pi, normalized_mi
from summarize_pilot import LAST_K, find, load_runs, load_threshold

SETTING = dict(aug="B", lr=1e-3, wd=1e-4, epochs=70)  # selection of stages 1 to 4
SEEDS = (0, 1, 2)


def routing_report(run_dir, subset_dir):
    """Routing report of one MoE run: minimum weighted usage, second-expert check, normalized mutual information, cos gap, w1, router parameter count."""
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    with open(run_dir / "history.csv", newline="") as f:
        hist = list(csv.DictReader(f))
    tail = hist[-LAST_K:]
    pi = load_pi(cfg, subset_dir)
    rep = {"router_params": cfg.get("router_params"),
           "min_usage_tail": min(min(json.loads(r["val_usage_weighted"])) for r in tail),
           "cos_gap_p50": float(hist[-1]["val_cos_gap_p50"]), "w1": float(hist[-1]["val_w1_mean"]),
           "w_share_min_tail": None, "top1_share_min_tail": None, "nmi": None}
    if tail[0].get("diag_val_weight_share_min") not in (None, ""):
        rep["w_share_min_tail"] = min(float(r["diag_val_weight_share_min"]) for r in tail)
    npz = run_dir / "val_routing.npz"
    if npz.exists():
        with open(subset_dir / cfg["val_list"], newline="") as f:
            y = np.array([int(r["class_idx"]) for r in csv.DictReader(f)])
        n_c = np.bincount(y, minlength=len(pi))
        top1 = np.load(npz)["topk"][-LAST_K:, :, 0].astype(int)  # (epoch, sample)
        shares = [[np.sum(pi * np.bincount(y[t == e], minlength=len(pi)) / n_c) * 100 for e in range(cfg["num_experts"])]
                  for t in top1]
        rep["top1_share_min_tail"] = float(np.min(shares))
    metrics = run_dir / "val_eval_last" / "metrics.json"
    if metrics.exists():
        assign = json.loads(metrics.read_text(encoding="utf-8"))["per_class_expert_assignment"]
        rep["nmi"] = float(normalized_mi(assign, pi)[0])
    return rep


def main():
    ap = argparse.ArgumentParser(description="Summarize stage 5 (R4 vs R0 non-inferiority check)")
    ap.add_argument("--router-dir", default="../runs/router")
    ap.add_argument("--repro-dir", default="../runs/repro")
    ap.add_argument("--threshold", default="../runs/pilot/threshold.json")
    ap.add_argument("--subset-dir", default="../datasets/subsets")
    ap.add_argument("--tau0", type=float, default=0.5)
    ap.add_argument("--lambda0", type=float, required=True)
    ap.add_argument("--sigma0", type=float, required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    subset_dir = Path(args.subset_dir)

    # X was measured on R0 in stage 0; load_threshold checks τ0, λ0, σ0 and router
    x = load_threshold(Path(args.threshold), argparse.Namespace(**vars(args), router="R0"))
    print(f"[Stage 5] Tie threshold X = {x * 100:.2f} pp (from {args.threshold}); setting {SETTING}")

    def runs_in(d, router):
        return load_runs(d, args.tau0, args.lambda0, args.sigma0, router)

    sources = {"R0": {0: args.repro_dir, 1: args.router_dir, 2: args.router_dir},
               "R4": {s: args.router_dir for s in SEEDS}}
    table = {}
    for router, by_seed in sources.items():
        loaded = {d: runs_in(d, router) for d in set(by_seed.values())}
        for s, d in by_seed.items():
            r = find(loaded[d], "moe", seed=s, **SETTING)
            if r is not None:
                r = {**r, "dir": str(Path(d) / r["name"]), **routing_report(Path(d) / r["name"], subset_dir)}
            table[(router, s)] = r

    print(f"\n{'run':10s} {'A_pi':>7s} {'ep':>3s} {'BA':>7s} {'F1':>7s} {'tail':>7s} {'minU%':>6s} {'top1min%':>8s} "
          f"{'wmin%':>6s} {'nmi':>6s} {'gap50':>6s} {'w1':>6s} {'params':>8s}")
    f = lambda v, p=4: "-" if v is None else f"{v:.{p}f}"
    for (router, s), r in table.items():
        if r is None:
            print(f"{router} s{s:<6d} missing")
            continue
        print(f"{router} s{s:<6d} {r['acc_pi']:7.4f} {r['best_epoch']:3d} {r['ba']:7.4f} {r['f1']:7.4f} {r['tail_pi']:7.4f} "
              f"{f(r['min_usage_tail'], 1):>6s} {f(r['top1_share_min_tail'], 1):>8s} {f(r['w_share_min_tail'], 1):>6s} "
              f"{f(r['nmi'], 3):>6s} {f(r['cos_gap_p50'], 3):>6s} {f(r['w1'], 3):>6s} {r['router_params'] or '-':>8}")

    pairs = {s: table[("R4", s)]["acc_pi"] - table[("R0", s)]["acc_pi"]
             for s in SEEDS if table[("R0", s)] and table[("R4", s)]}
    result = {"stage": "router", "args": vars(args), "X": x, "setting": SETTING,
              "runs": {f"{k[0]}_s{k[1]}": v for k, v in table.items()}, "d": pairs, "d_mean": None, "decision": None}
    if len(pairs) < len(SEEDS):
        print(f"\n[Info] Paired seeds available: {sorted(pairs)}; all of {list(SEEDS)} are required. No decision yet")
    else:
        d_mean = sum(pairs.values()) / len(pairs)
        result["d_mean"] = d_mean
        print("\nd_k = A_R4 - A_R0: " + ", ".join(f"s{s} {v * 100:+.2f} pp" for s, v in pairs.items())
              + f" | mean {d_mean * 100:+.2f} pp, range {min(pairs.values()) * 100:+.2f} ~ {max(pairs.values()) * 100:+.2f} pp")
        if d_mean >= -x:
            result["decision"] = "adopt R4"
            print(f"-> Adopt R4: mean d_k is not below -X ({-x * 100:.2f} pp)")
        else:
            result["decision"] = "stop and discuss"
            print(f"[STOP: discuss before continuing] mean d_k is below -X ({-x * 100:.2f} pp); do not revert to R0 "
                  "automatically. Check routing diagnostics first (see Pilot_Plan.md stage 5)")
    print("\nNote: X is a predefined tolerance, not a significance bound. Accuracy-only rule; routing columns are reported, not used.")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fo:
            json.dump(result, fo, ensure_ascii=False, indent=2, default=str)
        print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
