#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
summarize_stage6.py

Summarize pilot stage 6 (analysis/Pilot_Plan.md) and give conclusions by pre-set rules. Reads run directories only; writes results only when --out is given.

--part A  fixed baseline τ0, σ0, λ0 calibration (R4, augmentation B, lr 1e-3, wd 1e-4, 70 epoch). The center point (λ 1.5, σ 0.2, τ 0.5) is taken from --router-dir
          (stage 5 R4 runs), candidates from --stage6-dir. Rules:
          1. Exclusion: minimum weighted usage < 5% within the last 10 epochs of any seed
          2. Screening: candidates that are not excluded and whose seed 0 A_pi is higher than center seed 0 by X or more get seeds 1, 2 run; the rest are treated as not better than the center point
          3. For candidates with extra runs, compute seed-paired d_k = A_cand,k - A_center,k; adopt the parameter value if the mean >= X (if both values of the same parameter qualify, take the one with the larger mean)
          4. If two or more parameters are adopted, run the combined configuration with 3 seeds paired with the center point and adopt the combination if the mean >= X; otherwise adopt only the single item with the largest mean d_k
          5. On a tie keep the center point
          Candidates listed in --extend (e.g. lambda=4.5) skip the screening threshold of item 2, get seeds 1, 2 run directly, and are then judged by item 3
          (rule exception of 2026-10-28, see Pilot_Plan.md stage 6 A "λ 4.5 extension"); exclusion item 1 still applies.
          Outputs the runs still needed (to_run, read and run by run_pilot.sh fixed6), the final selection (when to_run is empty), the acceptable set of each parameter
          (not excluded and seed 0 A_pi within X of the best of that parameter's 3 values), the routing report and the controller input distribution.
--part B  auxfree confirmation (R4, seed 0). If minimum weighted usage < 5% within the last 10 epochs, suggest stopping to discuss; everything else is only reported.
--part C  aux loss warmup check (stage 6 C): the no-warmup center point in --stage6-dir (R4, λ 1.5, σ 0.2, τ 0.5, seeds 0/1/2)
          is paired by seed with the warmup center point in --ref-dir. Adopt "no warmup" if mean d_k >= -X and no seed is excluded; also report early
          (epochs 4 ~ 23) routing health and normalized mutual information, and state whether the lr / epoch stages need rerunning per the level-2 trigger conditions (a)(b)(c).
--aux-warmup-frac warmup fraction used to select runs (default 1/3 = old setting; the no-warmup runs from stage 6 C on use 0).

X is read from the stage 0 threshold.json; it is a decision rule, not a significance bound.

Usage (in p1/):
    python summarize_stage6.py --part A --lambda0 1.5 --sigma0 0.2 [--out ../runs/stage6/stage6a.json]
    python summarize_stage6.py --part B --lambda0 1.5 --sigma0 0.2 [--out ../runs/stage6/stage6b.json]
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_calib import load_pi
from summarize_pilot import LAST_K, find, load_runs, load_threshold
from summarize_router import SEEDS, SETTING, routing_report

PARAMS = ("lambda", "sigma", "tau")
CENTER = {"lambda": "1.5", "sigma": "0.2", "tau": "0.5"}
CANDS = {"lambda": ("0.5", "4.5"), "sigma": ("0.1", "0.4"), "tau": ("0.25", "1.0")}  # strings match the directory naming in run_pilot.sh
MIN_USAGE = 5.0
INPUTS = ("bs_raw", "t_fast_raw", "t_slow_raw")


def key(cfg):
    return tuple(cfg[p] for p in PARAMS)


def part_a(args, x):
    cache = {}

    def get(cfg, seed):
        d = args.router_dir if cfg == CENTER else args.stage6_dir
        k = (d, key(cfg))
        if k not in cache:
            cache[k] = load_runs(d, float(cfg["tau"]), float(cfg["lambda"]), float(cfg["sigma"]), "R4", args.aux_warmup_frac) if Path(d).is_dir() else []
        r = find(cache[k], "moe", seed=seed, **SETTING)
        if r is not None and "min_usage_tail" not in r:
            r.update(dir=str(Path(d) / r["name"]), **routing_report(Path(d) / r["name"], Path(args.subset_dir)))
        return r

    def excluded(runs):
        return any(r["min_usage_tail"] < MIN_USAGE for r in runs if r)

    def paired(cfg):
        c, b = [get(cfg, s) for s in SEEDS], [get(CENTER, s) for s in SEEDS]
        if not all(c) or not all(b):
            return None
        return sum(ci["acc_pi"] - bi["acc_pi"] for ci, bi in zip(c, b)) / len(SEEDS)

    to_run, table, center0 = [], {}, get(CENTER, 0)
    if center0 is None:
        raise SystemExit(f"[Error] Center run (R4, seed 0) not found in {args.router_dir}")
    adopted = {}  # parameter -> (value, mean d)
    for p in PARAMS:
        for v in CANDS[p]:
            cfg = {**CENTER, p: v}
            r0 = get(cfg, 0)
            row = table[f"{p}={v}"] = {"cfg": cfg, "seed0": r0, "status": None, "d_mean": None}
            if r0 is None:
                to_run.append({**cfg, "seed": 0}); row["status"] = "pending (seed 0)"
                continue
            if excluded([r0]):
                row["status"] = "excluded"; continue
            extended = f"{p}={v}" in args.extend
            if r0["acc_pi"] - center0["acc_pi"] < x and not extended:
                row["status"] = "not better than center (screen)"; continue
            missing = [s for s in SEEDS[1:] if get(cfg, s) is None]
            if missing:
                to_run += [{**cfg, "seed": s} for s in missing]
                row["status"] = ("extended" if extended else "screen passed") + "; seeds 1, 2 pending"
                continue
            if excluded([get(cfg, s) for s in SEEDS]):
                row["status"] = "excluded (seed 1 or 2)"; continue
            d = paired(cfg)
            row["d_mean"] = d
            row["status"] = "adopted" if d >= x else "tie after pairing"
            if d >= x and (p not in adopted or d > adopted[p][1]):
                adopted[p] = (v, d)

    final, combo = None, None
    if not to_run:
        if len(adopted) >= 2:
            combo = {**CENTER, **{p: v for p, (v, _) in adopted.items()}}
            missing = [s for s in SEEDS if get(combo, s) is None]
            if missing:
                to_run += [{**combo, "seed": s} for s in missing]
            else:
                d = paired(combo)
                ok = d >= x and not excluded([get(combo, s) for s in SEEDS])
                table["combination"] = {"cfg": combo, "d_mean": d, "status": "adopted" if ok else "rejected"}
                if ok:
                    final = combo
                else:
                    p, (v, _) = max(adopted.items(), key=lambda kv: kv[1][1])
                    final = {**CENTER, p: v}
        elif adopted:
            (p, (v, _)), = adopted.items()
            final = {**CENTER, p: v}
        else:
            final = dict(CENTER)

    # Acceptable set: among each parameter's 3 values, those not excluded whose seed 0 A_pi is within X of the best
    acceptable = {}
    for p in PARAMS:
        vals = [(v, center0 if v == CENTER[p] else get({**CENTER, p: v}, 0)) for v in sorted((CENTER[p], *CANDS[p]), key=float)]
        if any(r is None for _, r in vals):
            acceptable[p] = None  # seed 0 not finished
            continue
        ok = [(v, r) for v, r in vals if not excluded([r])]
        best = max(r["acc_pi"] for _, r in ok) if ok else None
        keep = [float(v) for v, r in ok if best - r["acc_pi"] < x]
        acceptable[p] = {"values": keep, "range": [min(keep), max(keep)] if keep else None}

    print(f"\n{'config':12s} {'A_pi s0':>8s} {'vs ctr':>8s} {'minU%':>6s} {'top1min%':>8s} {'wmin%':>6s} {'nmi':>6s} {'gap50':>6s} {'w1':>6s}  status")
    f = lambda v, p=1: "-" if v is None else f"{v:.{p}f}"
    r = center0
    print(f"{'center':12s} {r['acc_pi']:8.4f} {'':>8s} {f(r['min_usage_tail']):>6s} {f(r['top1_share_min_tail']):>8s} "
          f"{f(r['w_share_min_tail']):>6s} {f(r['nmi'], 3):>6s} {f(r['cos_gap_p50'], 3):>6s} {f(r['w1'], 3):>6s}")
    for name, row in table.items():
        r = row.get("seed0")
        if r is None:
            print(f"{name:12s} {'-':>8s} {'-':>8s} {'':>6s} {'':>8s} {'':>6s} {'':>6s} {'':>6s} {'':>6s}  {row['status']}"
                  + (f" (d mean {row['d_mean'] * 100:+.2f} pp)" if row.get("d_mean") is not None else ""))
            continue
        extra = f" (d mean {row['d_mean'] * 100:+.2f} pp)" if row["d_mean"] is not None else ""
        print(f"{name:12s} {r['acc_pi']:8.4f} {(r['acc_pi'] - center0['acc_pi']) * 100:+7.2f}p {f(r['min_usage_tail']):>6s} "
              f"{f(r['top1_share_min_tail']):>8s} {f(r['w_share_min_tail']):>6s} {f(r['nmi'], 3):>6s} {f(r['cos_gap_p50'], 3):>6s} "
              f"{f(r['w1'], 3):>6s}  {row['status']}{extra}")

    # Controller input distribution (from epoch 4 on, i.e. after the learning-rate warmup; seed 0 of the center point and all candidates), for designing the membership functions in stage 7
    inputs = {k: [] for k in INPUTS}
    for r in [center0] + [row["seed0"] for row in table.values() if row.get("seed0")]:
        with open(Path(r["dir"]) / "history.csv", newline="") as fh:
            for h in list(csv.DictReader(fh))[3:]:
                for k in INPUTS:
                    if h.get(k) not in (None, ""):
                        inputs[k].append(float(h[k]))
    dist = {k: dict(zip(("p5", "p25", "p50", "p75", "p95"), np.percentile(v, [5, 25, 50, 75, 95]).round(5).tolist()))
            for k, v in inputs.items() if v}
    print("\nController inputs (epochs >= 4, seed-0 runs): " + "; ".join(f"{k} {v}" for k, v in dist.items()))
    print("Acceptable sets (for stage 7): " + ", ".join(f"{p} {acceptable[p]['values'] if acceptable[p] else 'pending'}" for p in PARAMS))

    if to_run:
        print("\n[To run] " + "; ".join(f"lambda {r['lambda']}, sigma {r['sigma']}, tau {r['tau']}, seed {r['seed']}" for r in to_run))
        print("No final selection yet")
    else:
        print(f"\nFinal selection: lambda0 {final['lambda']}, sigma0 {final['sigma']}, tau0 {final['tau']}"
              + (" (center kept)" if final == CENTER else ""))
    return {"to_run": to_run, "final": final, "adopted": {p: {"value": v, "d_mean": d} for p, (v, d) in adopted.items()},
            "combination": combo, "acceptable": acceptable, "controller_inputs": dist, "center_seed0": center0,
            "candidates": table}


def part_b(args):
    found = []
    for d in sorted(Path(args.stage6_dir).iterdir()) if Path(args.stage6_dir).is_dir() else []:
        cfg_p = d / "config.json"
        if d.is_dir() and cfg_p.exists():
            cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
            if cfg.get("controller") == "auxfree" and cfg.get("router") == "R4" and cfg["seed"] == 0 and (d / "summary.json").exists():
                found.append(d)
    if len(found) != 1:
        raise SystemExit(f"[Error] Expected exactly one completed auxfree R4 seed-0 run in {args.stage6_dir}, found {[d.name for d in found]}")
    d = found[0]
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    summ = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    with open(d / "history.csv", newline="") as fh:
        hist = list(csv.DictReader(fh))
    tail = hist[-LAST_K:]
    rep = routing_report(d, Path(args.subset_dir))
    center = find(load_runs(args.router_dir, 0.5, 1.5, 0.2, "R4", args.aux_warmup_frac), "moe", seed=0, **SETTING)
    res = {"run": d.name, "tau0": cfg["tau0"], "sigma0": cfg["sigma0"], "auxfree_lambda": cfg["auxfree_lambda"],
           "auxfree_gamma": cfg["auxfree_gamma"], "best_epoch": summ["best_epoch"], "acc_pi": summ["best_val_acc_pi"],
           "center_seed0_acc_pi": center["acc_pi"] if center else None, **rep,
           "nobias_min_usage_tail": min(min(json.loads(r["val_usage_nobias"])) for r in tail),
           "bias_changed_top2_tail_mean": float(np.mean([float(r["val_bias_changed_top2"]) for r in tail])),
           "route_bias_last": json.loads(hist[-1]["route_bias"])}
    print(f"\n[auxfree] {d.name}: tau0 {cfg['tau0']}, sigma0 {cfg['sigma0']}, lambda {cfg['auxfree_lambda']}, gamma {cfg['auxfree_gamma']}")
    print(f"  best A_pi {res['acc_pi']:.4f} (epoch {res['best_epoch']})"
          + (f"; fixed center seed 0 {center['acc_pi']:.4f} ({(res['acc_pi'] - center['acc_pi']) * 100:+.2f} pp, reported only)" if center else ""))
    print(f"  last {LAST_K} epochs: min weighted usage {rep['min_usage_tail']:.1f}%, top-1 share min {rep['top1_share_min_tail']:.1f}%, "
          f"w share min {rep['w_share_min_tail']:.1f}%; without bias: min usage {res['nobias_min_usage_tail']:.1f}%; "
          f"top-2 changed by bias {res['bias_changed_top2_tail_mean'] * 100:.1f}% of validation samples")
    print(f"  route_bias (last epoch) {res['route_bias_last']}")
    res["stop"] = rep["min_usage_tail"] < MIN_USAGE
    if res["stop"]:
        print(f"[STOP: discuss before continuing] min weighted usage in the last {LAST_K} epochs is below {MIN_USAGE:g}%")
    else:
        print("-> No routing collapse by the stage 6 B criterion; bias-related columns are reported only")
    return res


def early_report(run_dir, subset_dir):
    """Early (epochs 4 ~ 23) routing health: median top-1 share of the weakest expert (π-weighted), median minimum weighted usage, p5/p50 of B."""
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    with open(run_dir / "history.csv", newline="") as f:
        hist = list(csv.DictReader(f))[3:23]
    pi = load_pi(cfg, subset_dir)
    with open(subset_dir / cfg["val_list"], newline="") as f:
        y = np.array([int(r["class_idx"]) for r in csv.DictReader(f)])
    n_c = np.bincount(y, minlength=len(pi))
    top1 = np.load(run_dir / "val_routing.npz")["topk"][3:23, :, 0].astype(int)
    t1min = [min(np.sum(pi * np.bincount(y[t == e], minlength=len(pi)) / n_c) * 100 for e in range(cfg["num_experts"])) for t in top1]
    b = [float(r["bs_raw"]) for r in hist if r["bs_raw"] not in ("", None)]
    return {"early_top1_share_min_median": float(np.median(t1min)),
            "early_min_usage_median": float(np.median([min(json.loads(r["val_usage_weighted"])) for r in hist])),
            "early_B_p5": float(np.percentile(b, 5)), "early_B_p50": float(np.percentile(b, 50))}


def part_c(args, x):
    def load(d, frac):
        runs = {s: find(load_runs(d, 0.5, 1.5, 0.2, "R4", frac), "moe", seed=s, **SETTING) for s in SEEDS}
        for s, r in runs.items():
            if r is not None:
                r.update(dir=str(Path(d) / r["name"]), **routing_report(Path(d) / r["name"], Path(args.subset_dir)),
                         **early_report(Path(d) / r["name"], Path(args.subset_dir)))
        return runs
    new, ref = load(args.stage6_dir, args.aux_warmup_frac), load(args.ref_dir, args.ref_aux_warmup_frac)
    print(f"\n{'run':12s} {'A_pi':>7s} {'ep':>3s} {'BA':>7s} {'tail':>7s} {'minU%':>6s} {'t1min%':>7s} {'nmi':>6s} | early: {'t1min%':>7s} {'minU%':>6s} {'B p5/p50':>10s}")
    f = lambda v, p=1: "-" if v is None else f"{v:.{p}f}"
    for label, runs in (("warmup", ref), ("no-warmup", new)):
        for s in SEEDS:
            r = runs[s]
            if r is None:
                print(f"{label[:9]:9s} s{s}  missing"); continue
            print(f"{label[:9]:9s} s{s} {r['acc_pi']:7.4f} {r['best_epoch']:3d} {r['ba']:7.4f} {r['tail_pi']:7.4f} {f(r['min_usage_tail']):>6s} "
                  f"{f(r['top1_share_min_tail']):>7s} {f(r['nmi'], 3):>6s} |        {f(r['early_top1_share_min_median']):>7s} "
                  f"{f(r['early_min_usage_median']):>6s} {f(r['early_B_p5']):>4s}/{f(r['early_B_p50']):>5s}")
    res = {"new": {f"s{s}": new[s] for s in SEEDS}, "ref": {f"s{s}": ref[s] for s in SEEDS}, "d": {}, "d_mean": None,
           "decision": None, "tier2": None, "new_range": None}
    if not all(new.values()):
        print("\n[Info] No-warmup center runs incomplete; no decision yet"); return res
    if not all(ref.values()):
        raise SystemExit(f"[Error] Reference (warmup) center runs missing in {args.ref_dir}")
    d = {s: new[s]["acc_pi"] - ref[s]["acc_pi"] for s in SEEDS}; dm = sum(d.values()) / len(d)
    acc = [new[s]["acc_pi"] for s in SEEDS]; res.update(d={f"s{s}": v for s, v in d.items()}, d_mean=dm, new_range=max(acc) - min(acc))
    excluded = [s for s in SEEDS if new[s]["min_usage_tail"] < MIN_USAGE]
    print("\nd_k = A_no-warmup - A_warmup: " + ", ".join(f"s{s} {v * 100:+.2f} pp" for s, v in d.items())
          + f" | mean {dm * 100:+.2f} pp | no-warmup seed range {res['new_range'] * 100:.2f} pp (X = {x * 100:.2f} pp)")
    if excluded:
        res["decision"] = "keep warmup (excluded)"; print(f"-> Keep the aux warmup: seeds {excluded} have min weighted usage below {MIN_USAGE:g}% in the last {LAST_K} epochs")
    elif dm >= -x:
        res["decision"] = "adopt no-warmup"; print("-> Adopt the no-warmup setting for all later stages and the main experiments (mean d_k is not below -X)")
    else:
        res["decision"] = "keep warmup"; print("-> Keep the aux warmup: mean d_k is below -X; design stage 7 with the x0.1 factor documented")
    # Level-2 trigger conditions (Pilot_Plan.md stage 6 C)
    m = lambda runs, k: float(np.mean([runs[s][k] for s in SEEDS]))
    e_new, e_ref = m(new, "early_top1_share_min_median"), m(ref, "early_top1_share_min_median")
    n_new, n_ref = m(new, "nmi"), m(ref, "nmi")
    ratio = e_new / e_ref if e_ref > 0 else float("inf")
    tier2 = {"a_dmean_gt_X": dm > x, "b_early_top1_ratio": ratio, "b_triggered": ratio >= 2 or ratio <= 0.5,
             "c_nmi_new": n_new, "c_nmi_ref": n_ref, "c_triggered": n_new < 0.5 * n_ref}
    tier2["triggered"] = bool(tier2["a_dmean_gt_X"] or tier2["b_triggered"] or tier2["c_triggered"])
    res["tier2"] = tier2
    print(f"Tier 2 check: (a) mean d > X: {tier2['a_dmean_gt_X']}; (b) early top-1 share min median {e_new:.1f}% vs {e_ref:.1f}% "
          f"(ratio {ratio:.2f}): {tier2['b_triggered']}; (c) nmi {n_new:.3f} vs {n_ref:.3f}: {tier2['c_triggered']}")
    print("-> Tier 2 reruns (lr 1e-4/3e-4/3e-3, epochs 50/100 for the MoE) ARE required" if tier2["triggered"]
          else "-> Tier 2 reruns not required")
    return res


def main():
    ap = argparse.ArgumentParser(description="Summarize stage 6 (A: fixed baseline calibration on R4; B: auxfree check)")
    ap.add_argument("--part", choices=["A", "B", "C"], required=True)
    ap.add_argument("--stage6-dir", default="../runs/stage6")
    ap.add_argument("--router-dir", default="../runs/router", help="Directory holding the center (R4, lambda 1.5, sigma 0.2, tau 0.5) runs for parts A and B")
    ap.add_argument("--ref-dir", default="../runs/router", help="Part C: directory holding the reference (with warmup) center runs")
    ap.add_argument("--aux-warmup-frac", type=float, default=1 / 3, help="aux loss warmup fraction of the runs to summarize (0 = no warmup)")
    ap.add_argument("--ref-aux-warmup-frac", type=float, default=1 / 3, help="Part C: warmup fraction of the reference runs")
    ap.add_argument("--threshold", default="../runs/pilot/threshold.json")
    ap.add_argument("--subset-dir", default="../datasets/subsets")
    ap.add_argument("--tau0", type=float, default=0.5)
    ap.add_argument("--lambda0", type=float, required=True, help="Stage -1 lambda0, used only to check threshold.json")
    ap.add_argument("--sigma0", type=float, required=True, help="Stage -1 sigma0, used only to check threshold.json")
    ap.add_argument("--extend", nargs="*", default=[], choices=[f"{p}={v}" for p in PARAMS for v in CANDS[p]],
                    help="Candidates that skip the screening threshold (rule exception; part A only)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    x = load_threshold(Path(args.threshold), argparse.Namespace(**vars(args), router="R0"))
    print(f"[Stage 6 {args.part}] Tie threshold X = {x * 100:.2f} pp (from {args.threshold}); setting {SETTING}, router R4")
    body = {"A": lambda: part_a(args, x), "B": lambda: part_b(args), "C": lambda: part_c(args, x)}[args.part]()
    result = {"part": args.part, "args": vars(args), "X": x, **body}
    print("\nNote: X is a predefined tolerance, not a significance bound. Routing columns are reported, not used, except the 5% usage exclusion.")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fo:
            json.dump(result, fo, ensure_ascii=False, indent=2, default=str)
        print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
