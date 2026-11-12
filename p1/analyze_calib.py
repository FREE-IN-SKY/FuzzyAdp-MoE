#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_calib.py

Analysis script for stage −1 (calibration of routing defaults), see analysis/Pilot_Plan.md. Only reads files, does not modify anything
(writes a summary JSON when --out is given).

Each run directory needs:
    config.json, history.csv       output of train_moe.py
    <eval-subdir>/metrics.json     output of evaluate.py on val.csv for last.pth (class × expert assignment matrix, τ)
    <eval-subdir>/predictions.csv  same as above (routing probabilities of the 5 experts for each image)

For each run compute:
    A_pi           validation A_π of the last epoch
    min_usage      minimum weighted expert usage (%) of the last epoch
    nmi            normalized mutual information I(C;E)/H(E). The joint distribution is π-weighted: P(c,e) = π_c · M_ce / (top_k · n_c),
                   consistent with how weighted_expert_usage combines them. Close to 0 means the expert assignment is independent of the class
    flip@σ         fraction of validation samples whose top-2 (as a set) changes after adding N(0, σ²) score noise, Monte Carlo estimate.
                   The score difference is recovered from s_i − s_j = τ·log(p_i / p_j) (no noise at validation, the scores differ only by a common constant)
    cos_gap_p10/p50/p90, w1_mean   values of the last epoch, from history.csv

Runs are grouped into configs by (λ0, σ0) in config; a config can contain several seeds (the seed is read from config, not from the directory name):
    A_pi takes the mean, min_usage takes the minimum over seeds, nmi takes the mean and reports the range, the other quantities take the mean

Selection rules (fixed before the experiment, per config):
    1. Exclude configs with min_usage < --min-usage (default 5%) or with mean A_pi lower than the highest mean over all configs by > --acc-drop (default 1 pp)
    2. Among the remaining configs, select the one with the highest mean nmi
    3. When the mean nmi of all configs is < --nmi-floor, only warn and make no selection (stop for discussion)
    4. Additionally report whether the gap in mean nmi between first and second place exceeds the seed range of both; does not change the selection

Usage:
    python analyze_calib.py ../runs/calib/lam0.1_sig0.1 ../runs/calib/lam0.1_sig0.1_s{1,2} ... [--out x.json]
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

TOP_K = 2


def load_pi(config, subset_dir):
    """Read π from the manifest in the class order of config."""
    with open(Path(subset_dir).expanduser() / "manifest.json", encoding="utf-8") as f:
        m = json.load(f)
    pi_values = m["target_distribution_pi"]["values"]
    return np.array([pi_values[name] for name in config["class_names"]], dtype=np.float64)


def normalized_mi(assign, pi):
    """assign: (C, N_e) top-k count matrix. Returns (I(C;E)/H(E), H(E)), using only classes that have samples in the validation set."""
    m = np.asarray(assign, dtype=np.float64)
    n_c = m.sum(1) / TOP_K
    present = n_c > 0
    w = pi * present
    w = w / w.sum()
    cond = np.zeros_like(m)
    cond[present] = m[present] / (TOP_K * n_c[present, None])  # P(e | c)
    joint = w[:, None] * cond
    p_e = joint.sum(0)
    h_e = -np.sum(p_e[p_e > 0] * np.log(p_e[p_e > 0]))
    nz = joint > 0
    mi = np.sum(joint[nz] * np.log(joint[nz] / (w[:, None] * p_e[None, :])[nz]))
    return (mi / h_e if h_e > 0 else float("nan")), h_e


def flip_rate(probs, tau, sigma, n_draws, rng):
    """Fraction of validation samples whose top-2 set changes after adding N(0, σ²) score noise."""
    if sigma <= 0:
        return 0.0
    s = tau * np.log(np.clip(probs, 1e-12, None))  # differs from the true scores only by a common constant per row, does not affect the ranking
    base = np.sort(np.argsort(-s, axis=1)[:, :TOP_K], axis=1)
    changed = 0
    for _ in range(n_draws):
        noisy = s + rng.normal(0.0, sigma, size=s.shape)
        top = np.sort(np.argsort(-noisy, axis=1)[:, :TOP_K], axis=1)
        changed += np.any(top != base, axis=1).sum()
    return changed / (n_draws * len(s))


def analyze_run(run_dir, eval_subdir, subset_dir, sigmas, n_draws, seed):
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("model") != "moe":
        sys.exit(f"[Error] {run_dir} is not a MoE run")
    if not (run_dir / "summary.json").exists():
        sys.exit(f"[Error] {run_dir} has no summary.json; the run may be incomplete")
    with open(run_dir / "history.csv", newline="") as f:
        hist = list(csv.DictReader(f))
    if len(hist) != int(config["epochs"]):
        print(f"[Warning] {run_dir.name}: history.csv has {len(hist)} epochs, but {config['epochs']} were configured")
    last = hist[-1]

    ev = run_dir / eval_subdir
    metrics = json.loads((ev / "metrics.json").read_text(encoding="utf-8"))
    if metrics.get("checkpoint_epoch") != int(config["epochs"]):
        print(f"[Warning] {run_dir.name}: the evaluated checkpoint is from epoch {metrics.get('checkpoint_epoch')}; last.pth is expected")
    with open(ev / "predictions.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    cols = sorted((c for c in rows[0] if c.startswith("router_p")), key=lambda c: int(c[len("router_p"):]))
    probs = np.array([[float(r[c]) for c in cols] for r in rows], dtype=np.float64)

    pi = load_pi(config, subset_dir)
    nmi, h_e = normalized_mi(metrics["per_class_expert_assignment"], pi)
    rng = np.random.default_rng(seed)
    flips = {s: flip_rate(probs, metrics["tau"], s, n_draws, rng) for s in sorted({*sigmas, float(config["sigma0"])})}
    return {
        "name": run_dir.name, "seed": int(config["seed"]), "train_list": config.get("train_list"),
        "lambda0": float(config["lambda0"]), "sigma0": float(config["sigma0"]), "tau": float(metrics["tau"]),
        "epochs": int(config["epochs"]), "acc_pi": float(last["val_acc_pi"]),
        "min_usage": float(min(json.loads(last["val_usage_weighted"]))),
        "usage": json.loads(last["val_usage_weighted"]),
        "nmi": float(nmi), "h_e": float(h_e), "flips": flips,
        "cos_gap": [float(last[f"val_cos_gap_p{q}"]) for q in (10, 50, 90)],
        "w1_mean": float(last["val_w1_mean"]),
    }


def group_runs(res):
    """Group by (λ0, σ0), check that settings within a group are consistent and seeds are not duplicated."""
    groups = defaultdict(list)
    for r in res:
        groups[(r["lambda0"], r["sigma0"])].append(r)
    out = []
    for (lam, sig), runs in sorted(groups.items()):
        seeds = [r["seed"] for r in runs]
        if len(set(seeds)) != len(seeds):
            sys.exit(f"[Error] Duplicate seeds in configuration lambda0={lam:g}, sigma0={sig:g}: {[r['name'] for r in runs]}")
        if len({(r["tau"], r["epochs"]) for r in runs}) > 1:
            sys.exit(f"[Error] Runs in configuration lambda0={lam:g}, sigma0={sig:g} differ in tau or number of epochs")
        for r in runs:
            if r["train_list"] != f"train_s{r['seed']}.csv":
                sys.exit(f"[Error] {r['name']}: training list {r['train_list']} does not match seed {r['seed']} (scheme A)")
        runs = sorted(runs, key=lambda r: r["seed"])
        nmis = [r["nmi"] for r in runs]
        sig_keys = sorted({s for r in runs for s in r["flips"]})
        out.append({
            "lambda0": lam, "sigma0": sig, "seeds": [r["seed"] for r in runs], "runs": [r["name"] for r in runs],
            "acc_pi": float(np.mean([r["acc_pi"] for r in runs])),
            "acc_pi_range": [min(r["acc_pi"] for r in runs), max(r["acc_pi"] for r in runs)],
            "min_usage": min(r["min_usage"] for r in runs),
            "nmi": float(np.mean(nmis)), "nmi_range": [min(nmis), max(nmis)],
            "flips": {s: float(np.mean([r["flips"][s] for r in runs if s in r["flips"]])) for s in sig_keys},
            "cos_gap": np.mean([r["cos_gap"] for r in runs], axis=0).tolist(),
            "w1_mean": float(np.mean([r["w1_mean"] for r in runs])),
        })
    return out


def main():
    ap = argparse.ArgumentParser(description="Analyze stage -1 routing-default calibration runs and select by the predefined rules")
    ap.add_argument("runs", nargs="+", help="Run directories to compare; grouped automatically by (lambda0, sigma0) from config")
    ap.add_argument("--eval-subdir", default="val_eval_last")
    ap.add_argument("--subset-dir", default="../datasets/subsets")
    ap.add_argument("--sigmas", default="0.05,0.1,0.2", help="Comma-separated sigma values at which to also report the top-2 flip rate")
    ap.add_argument("--n-draws", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0, help="Monte Carlo random seed")
    ap.add_argument("--min-usage", type=float, default=5.0, help="Exclusion threshold: minimum weighted expert usage (%%)")
    ap.add_argument("--acc-drop", type=float, default=0.01, help="Exclusion threshold: drop below the highest mean A_pi among all configurations")
    ap.add_argument("--nmi-floor", type=float, default=0.05,
                    help="If every configuration has mean nmi below this value, report and do not select (triggers discussion only)")
    ap.add_argument("--out", default=None, help="Optional JSON output file")
    args = ap.parse_args()

    sigmas = [float(s) for s in args.sigmas.split(",") if s]
    res = [analyze_run(r, args.eval_subdir, args.subset_dir, sigmas, args.n_draws, args.seed) for r in args.runs]
    all_sigmas = sorted({s for r in res for s in r["flips"]})
    flip_head = " ".join(f"{'flip@' + format(s, 'g'):>9s}" for s in all_sigmas)

    def flip_cols(flips):
        return " ".join(f"{flips[s]:9.3f}" if s in flips else f"{'-':>9s}" for s in all_sigmas)

    print("[Runs]")
    print(f"{'run':30s} {'lambda0':>7s} {'sigma0':>6s} {'seed':>4s} {'A_pi':>7s} {'min_usage':>9s} {'nmi':>6s} {flip_head}")
    for r in sorted(res, key=lambda r: (r["lambda0"], r["sigma0"], r["seed"])):
        print(f"{r['name']:30s} {r['lambda0']:7g} {r['sigma0']:6g} {r['seed']:>4d} {r['acc_pi']:7.4f} "
              f"{r['min_usage']:9.2f} {r['nmi']:6.3f} {flip_cols(r['flips'])}")
        print(f"{'':30s} usage = {np.round(r['usage'], 1).tolist()}")

    groups = group_runs(res)
    n_seeds = {len(g["seeds"]) for g in groups}
    if len(n_seeds) > 1:
        print(f"\n[Warning] Configurations have different numbers of seeds: {[(format(g['lambda0'], 'g'), format(g['sigma0'], 'g'), g['seeds']) for g in groups]}")
    print("\n[Configurations: A_pi = mean, min_usage = minimum over seeds, nmi = mean (range)]")
    print(f"{'lambda0':>7s} {'sigma0':>6s} {'seeds':>9s} {'A_pi':>7s} {'A_pi_range':>15s} {'min_usage':>9s} "
          f"{'nmi':>6s} {'nmi_range':>13s} {flip_head} {'gap p10/p50/p90':>20s} {'w1':>6s}")
    for g in groups:
        gap = "/".join(f"{x:.3f}" for x in g["cos_gap"])
        print(f"{g['lambda0']:7g} {g['sigma0']:6g} {','.join(map(str, g['seeds'])):>9s} {g['acc_pi']:7.4f} "
              f"{g['acc_pi_range'][0]:.4f}~{g['acc_pi_range'][1]:.4f} {g['min_usage']:9.2f} {g['nmi']:6.3f} "
              f"{g['nmi_range'][0]:.3f}~{g['nmi_range'][1]:.3f} {flip_cols(g['flips'])} {gap:>20s} {g['w1_mean']:6.3f}")

    best_acc = max(g["acc_pi"] for g in groups)
    for g in groups:
        reasons = []
        if g["min_usage"] < args.min_usage:
            reasons.append(f"min usage {g['min_usage']:.2f}% < {args.min_usage:g}%")
        if best_acc - g["acc_pi"] > args.acc_drop:
            reasons.append(f"mean A_pi is {(best_acc - g['acc_pi']) * 100:.2f} pp below the best > {args.acc_drop * 100:g} pp")
        g["excluded"] = reasons
    print()
    for g in groups:
        if g["excluded"]:
            print(f"[Excluded] lambda0={g['lambda0']:g}, sigma0={g['sigma0']:g}: {'; '.join(g['excluded'])}")

    kept = sorted([g for g in groups if not g["excluded"]], key=lambda g: -g["nmi"])
    pick = None
    if all(g["nmi"] < args.nmi_floor for g in groups):
        print(f"[STOP: discuss before continuing] Every configuration has mean nmi below {args.nmi_floor:g}; expert assignment is nearly independent of class, so the problem may lie in the router itself")
    elif not kept:
        print("[STOP: discuss before continuing] All configurations were excluded")
    else:
        pick = kept[0]
        print(f"-> Select lambda0={pick['lambda0']:g}, sigma0={pick['sigma0']:g}: highest mean nmi among non-excluded configurations ({pick['nmi']:.3f})")
        if len(kept) > 1:
            second = kept[1]
            gap = pick["nmi"] - second["nmi"]
            head = f"  vs. runner-up (lambda0={second['lambda0']:g}, sigma0={second['sigma0']:g}, nmi {second['nmi']:.3f}): gap {gap:.3f}"
            if min(len(pick["seeds"]), len(second["seeds"])) < 2:
                print(head + "; single seed, cannot tell whether the gap is within seed variation")
            else:
                spread = max(pick["nmi_range"][1] - pick["nmi_range"][0], second["nmi_range"][1] - second["nmi_range"][0])
                verdict = "larger than" if gap > spread else "not larger than"
                print(head + f", {verdict} the larger seed range of the two ({spread:.3f})"
                      + ("" if gap > spread else "; the gap is within seed variation, so the selection only follows the rule"))
    print(f"Note: {'/'.join(map(str, sorted(n_seeds)))} seed(s) per configuration, "
          f"{'/'.join(sorted({str(r['epochs']) for r in res}))} epochs; results are provisional defaults for stages 0 to 5 only.")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"runs": res, "groups": groups,
                       "pick": {"lambda0": pick["lambda0"], "sigma0": pick["sigma0"]} if pick else None,
                       "rule": {"min_usage": args.min_usage, "acc_drop": args.acc_drop, "nmi_floor": args.nmi_floor}},
                      f, ensure_ascii=False, indent=2, default=float)


if __name__ == "__main__":
    main()
