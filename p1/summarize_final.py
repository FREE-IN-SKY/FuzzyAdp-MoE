#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
summarize_final.py

Summarize the formal experiment (seeds 3, 4, 5) per analysis/Pilot_Study.md 11.3, 11.4. Reads run directories only; output is written to --out-dir
(default ../analysis/final/: final_summary.json holds the full data, final_tables.md the numbers for each table in the paper).

- Main comparison (A, test set, best.pth): 3-seed mean and range of A_pi, accuracy, BA, Macro-F1 for each method; seed-paired d_k against fixed,
  mean >= +X is better, within ±X a tie, <= -X worse (X is the same as in the pilot study, recomputed by summarize_study.compute_x).
  Supplementary (not used in the verdict): the same numbers for last.pth; paired differences on the validation set by best value and by last-10-epoch mean (whether the rankings agree); exclusion condition.
- Robustness (B, test set, best.pth): paired difference and verdict of fuzzy minus fixed under each bad start; balance results (number of excluded seeds, first
  epoch with usage < 5%, whether it recovered) are reported separately; the paired difference of fuzzy bad start minus baseline start is descriptive only.
- Routing table (the 4 MoE methods of A): minimum weighted usage over the last 10 validation epochs, minimum top-1 share (routing_report), class-expert normalized
  mutual information and maximum assignment similarity between two experts (assignment matrix of val_eval_last); the same metrics for test-set best.pth as a supplement.
  Assignment similarity: for each expert, the fraction of samples in each class for which it is selected into the top-2 forms a 27-dim vector; take the maximum pairwise Pearson correlation (defined 2026-11-06).
- Timing (C): total_time_min and steady-state per-epoch time from epoch 2 on (training + validation).
- Expert function analysis: merged in when experts.json exists.
If runs or test evaluations for A, B are missing, only the missing items are listed, no verdict is given and no output is written; if C or the expert analysis is missing, this is only marked in the corresponding section.

Usage (in p1/):
    python summarize_final.py [--final-dir ../runs/final] [--timing-dir ../runs/final_timing] [--out-dir ../analysis/final]
"""

import argparse
import csv
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

from analyze_calib import TOP_K, load_pi, normalized_mi
from summarize_router import routing_report
from summarize_study import MIN_USAGE, Store, compute_x, paired, usage_flags

SEEDS = (3, 4, 5)
METHODS = ("single", "fixed", "schedule", "auxfree", "fuzzy")
MOE = METHODS[1:]
NAMES = {"single": "Monolithic EfficientNet-B0", "fixed": "Fixed MoE", "schedule": "Schedule MoE", "auxfree": "Aux-free MoE",
         "fuzzy": "FuzzyAdp-MoE"}
BASE = (1.5, 0.4, 0.5)
STARTS = {"lambda_0": (0.0, 0.4, 0.5), "combo_collapse": (0.0, 0.0, 0.125), "combo_flat": (40.5, 3.2, 2.0)}  # the 3 bad starts of 9b
METRICS = ("acc_pi", "acc", "balanced_acc", "macro_f1")
FUZZY_CONFIG = Path(__file__).resolve().parent / "fuzzy_v2_r3.json"


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def history(d):
    with open(Path(d) / "history.csv", newline="") as f:
        return list(csv.DictReader(f))


def classify(cfg, fz_cfg):
    """Returns (part, method, start key); runs not belonging to the formal experiment return None."""
    if cfg["model"] != "moe":
        return ("A", "single", None) if abs(float(cfg["weight_decay"]) - 0.01) < 1e-12 else None
    ctrl, start = cfg["controller"], (float(cfg["lambda0"]), float(cfg["sigma0"]), float(cfg["tau0"]))
    if ctrl == "fuzzy" and cfg.get("fuzzy_config") != fz_cfg:
        raise SystemExit(f"[Error] {cfg['out_dir']}: fuzzy config differs from {FUZZY_CONFIG.name}")
    if ctrl == "auxfree":
        return ("A", "auxfree", None) if start == (0.0, 0.0, 0.5) else None
    if start == BASE:
        return ("A", ctrl, None)
    for key, st in STARTS.items():
        if start == st and ctrl in ("fixed", "fuzzy"):
            return ("B", ctrl, key)
    return None


def collect(d, fz_cfg):
    """Formal runs in d: {(part, method, start key, seed): run directory}."""
    found = {}
    for run in sorted(Path(d).iterdir()) if Path(d).is_dir() else []:
        if not (run.is_dir() and (run / "config.json").exists() and (run / "summary.json").exists()):
            continue
        cfg = read_json(run / "config.json")
        c = classify(cfg, fz_cfg)
        if c is None or int(cfg["seed"]) not in SEEDS:
            continue
        key = (*c, int(cfg["seed"]))
        if key in found:
            raise SystemExit(f"[Error] Two runs match {key}: {found[key].name}, {run.name}")
        found[key] = run
    return found


def expected():
    keys = [("A", m, None, s) for m in METHODS for s in SEEDS]
    keys += [("B", m, k, s) for k in STARTS for m in ("fixed", "fuzzy") for s in SEEDS]
    return keys


def shares(assign, pi):
    """Assignment matrix (C, E) -> per-class top-k fraction of each expert P(e | c) (C, E), mask of classes that have samples, π-weighted usage (%)."""
    m = np.asarray(assign, dtype=np.float64)
    n_c = m.sum(1) / TOP_K
    present = n_c > 0
    p = np.zeros_like(m)
    p[present] = m[present] / (TOP_K * n_c[present, None])
    w = pi * present
    return p, present, (w / w.sum()) @ p * 100


def assignment_similarity(assign, pi):
    """Maximum pairwise Pearson correlation; when an expert's fraction is the same across all classes (e.g. never selected after collapse) the correlation is undefined and that pair is skipped; nan if all are undefined."""
    p, present, _ = shares(assign, pi)
    v = p[present]
    r = [np.corrcoef(v[:, i], v[:, j])[0, 1] for i, j in combinations(range(v.shape[1]), 2) if v[:, i].std() > 0 and v[:, j].std() > 0]
    return float(max(r)) if r else float("nan")


def test_routing(eval_dir, pi):
    """Routing metrics of test-set best.pth (supplementary): minimum weighted usage, minimum top-1 share, normalized mutual information, assignment similarity."""
    m = read_json(eval_dir / "metrics.json")
    assign = np.asarray(m["per_class_expert_assignment"])
    _, present, usage = shares(assign, pi)
    with open(eval_dir / "predictions.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    y = np.array([int(r["label"]) for r in rows])
    e1 = np.array([int(r["expert_1"]) for r in rows])
    n_c = np.bincount(y, minlength=len(pi))
    w = pi * (n_c > 0)
    w = w / w.sum()
    top1 = [float(np.sum(w * np.bincount(y[e1 == e], minlength=len(pi)) / np.maximum(n_c, 1)) * 100) for e in range(assign.shape[1])]
    return {"min_usage": float(usage.min()), "min_top1_share": float(min(top1)), "nmi": float(normalized_mi(assign, pi)[0]),
            "assign_similarity": assignment_similarity(assign, pi)}


def stats(v):
    return {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v)), "values": [float(x) for x in v]}


def verdict(mean, x):
    return "better" if mean >= x else "worse" if mean <= -x else "tie"


def pdiff(a, b):
    d, m = paired([{"acc_pi": v} for v in a], [{"acc_pi": v} for v in b])
    return {"d": d, "mean": m}


def fmt(s, scale=100, nd=2):
    return f"{s['mean'] * scale:.{nd}f} ({s['min'] * scale:.{nd}f} ~ {s['max'] * scale:.{nd}f})"


def main():
    ap = argparse.ArgumentParser(description="Summarize the formal experiment (Pilot_Study.md 11.3, 11.4)")
    ap.add_argument("--final-dir", default="../runs/final")
    ap.add_argument("--timing-dir", default="../runs/final_timing")
    ap.add_argument("--nw-dir", default="../runs/stage6c", help="Pilot center runs used to recompute X")
    ap.add_argument("--subset-dir", default="../datasets/subsets")
    ap.add_argument("--out-dir", default="../analysis/final")
    args = ap.parse_args()
    fz_cfg = read_json(FUZZY_CONFIG)
    x, _ = compute_x(Store(argparse.Namespace(subset_dir=args.subset_dir)), argparse.Namespace(nw_dir=args.nw_dir))
    print(f"X = {x * 100:.2f} pp (recomputed from the pilot center runs)")

    runs = collect(args.final_dir, fz_cfg)
    missing = [k for k in expected() if k not in runs]
    missing += [f"{runs[k].name}/test_eval_{ck}" for k in expected() if k in runs for ck in ("best", "last")
                if not (runs[k] / f"test_eval_{ck}" / "metrics.json").exists()]
    if missing:
        print("[Missing] " + "\n[Missing] ".join(map(str, missing)))
        sys.exit("[Error] Batches A and B or their test evaluations are incomplete; no verdicts were computed")

    pi = load_pi(read_json(runs[("A", "fixed", None, 3)] / "config.json"), args.subset_dir)
    rec = {}
    for k, d in runs.items():
        h = history(d)
        acc = [float(r["val_acc_pi"]) for r in h]
        r = {"name": d.name, "best_epoch": read_json(d / "summary.json")["best_epoch"], "val_best": max(acc),
             "val_tail10": float(np.mean(acc[-10:])),
             **{f"test_{ck}": {m: read_json(d / f"test_eval_{ck}" / "metrics.json")[m] for m in METRICS} for ck in ("best", "last")}}
        if k[1] != "single":
            rep = routing_report(d, Path(args.subset_dir))
            r["routing_val"] = {"min_usage": rep["min_usage_tail"], "min_top1_share": rep["top1_share_min_tail"], "nmi": rep["nmi"],
                                "assign_similarity": assignment_similarity(read_json(d / "val_eval_last" / "metrics.json")["per_class_expert_assignment"], pi)}
            r["routing_test"] = test_routing(d / "test_eval_best", pi)
            r["excluded"] = rep["min_usage_tail"] < MIN_USAGE
            r["usage"] = usage_flags({"dir": str(d), "min_usage_tail": rep["min_usage_tail"]}, args.subset_dir)
        rec[k] = r
    A = lambda m: [rec[("A", m, None, s)] for s in SEEDS]
    B = lambda m, st: [rec[("B", m, st, s)] for s in SEEDS]

    # Main comparison
    main_res = {}
    for m in METHODS:
        rr = A(m)
        e = {"runs": [r["name"] for r in rr],
             **{ck: {mt: stats([r[f"test_{ck}"][mt] for r in rr]) for mt in METRICS} for ck in ("best", "last")}}
        if m != "fixed":
            fx = A("fixed")
            for ck in ("best", "last"):
                e[ck]["vs_fixed"] = pdiff([r[f"test_{ck}"]["acc_pi"] for r in rr], [r[f"test_{ck}"]["acc_pi"] for r in fx])
            e["verdict"] = verdict(e["best"]["vs_fixed"]["mean"], x)
            e["val_vs_fixed"] = {"best": pdiff([r["val_best"] for r in rr], [r["val_best"] for r in fx]),
                                 "tail10": pdiff([r["val_tail10"] for r in rr], [r["val_tail10"] for r in fx])}
        if m != "single":
            e["excluded_seeds"] = [s for s, r in zip(SEEDS, rr) if r["excluded"]]
        main_res[m] = e
    # Robustness
    robust = {}
    for st in STARTS:
        fx, fz = B("fixed", st), B("fuzzy", st)
        d = pdiff([r["test_best"]["acc_pi"] for r in fz], [r["test_best"]["acc_pi"] for r in fx])
        robust[st] = {"start": STARTS[st], "fixed": [r["test_best"]["acc_pi"] for r in fx], "fuzzy": [r["test_best"]["acc_pi"] for r in fz],
                      "fuzzy_minus_fixed": d, "verdict": verdict(d["mean"], x),
                      "excluded_seeds": {"fixed": [s for s, r in zip(SEEDS, fx) if r["excluded"]], "fuzzy": [s for s, r in zip(SEEDS, fz) if r["excluded"]]},
                      "usage": {"fixed": [r["usage"] for r in fx], "fuzzy": [r["usage"] for r in fz]},
                      "fuzzy_minus_fuzzy_base": pdiff([r["test_best"]["acc_pi"] for r in fz], [r["test_best"]["acc_pi"] for r in A("fuzzy")])}
    # Routing table
    routing = {m: {src: {k: stats([r[f"routing_{src}"][k] for r in A(m)]) for k in ("min_usage", "min_top1_share", "nmi", "assign_similarity")}
                   for src in ("val", "test")} for m in MOE}
    # Timing
    timing, t_missing = {}, []
    t_runs = collect(args.timing_dir, fz_cfg)
    for m in METHODS:
        tt = [t_runs.get(("A", m, None, s)) for s in SEEDS]
        t_missing += [f"timing {m} seed {s}" for s, t in zip(SEEDS, tt) if t is None]
        if all(tt):
            tot = [read_json(t / "summary.json")["total_time_min"] for t in tt]
            ep = [float(np.mean([float(r["train_time_s"]) + float(r["val_time_s"]) for r in history(t) if int(r["epoch"]) >= 2])) for t in tt]
            timing[m] = {"total_time_min": stats(tot), "epoch_time_s": stats(ep)}
    # Expert function analysis
    ex_path = Path(args.final_dir) / "experts.json"
    experts = read_json(ex_path)["summary"] if ex_path.exists() else None

    out = {"X": x, "seeds": SEEDS, "main": main_res, "robustness": robust, "routing": routing, "timing": timing,
           "timing_missing": t_missing, "experts": experts,
           "runs": {"/".join(str(v) for v in k if v is not None): r for k, r in rec.items()}}
    od = Path(args.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    with open(od / "final_summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)

    L = [f"# Formal experiment summary (seeds 3, 4, 5; X = {x * 100:.2f} pp)", "", "Generated by `p1/summarize_final.py`. Numbers are the mean of 3 seeds (min ~ max), in % or pp.", "",
         "## Table II: test set, best.pth", "", "| Method | A_π | Accuracy | BA | Macro-F1 | ΔA_π vs fixed (pp) | Verdict |", "|---|---|---|---|---|---|---|"]
    for m in METHODS:
        e = main_res[m]
        dv = "—" if m == "fixed" else f"{e['best']['vs_fixed']['mean'] * 100:+.2f} ({', '.join(f'{v * 100:+.2f}' for v in e['best']['vs_fixed']['d'])})"
        L.append(f"| {NAMES[m]} | {fmt(e['best']['acc_pi'])} | {fmt(e['best']['acc'])} | {fmt(e['best']['balanced_acc'])} | "
                 f"{fmt(e['best']['macro_f1'], 1, 4)} | {dv} | {e.get('verdict', '—')} |")
    L += ["", "## Supplement: paired differences on last.pth (test set) and the validation set (not used for the verdict)", "",
          "| Method | last.pth A_π | last.pth ΔA_π (pp) | Validation best ΔA_π (pp) | Validation last 10 epochs ΔA_π (pp) | Excluded seeds |", "|---|---|---|---|---|---|"]
    for m in METHODS:
        e = main_res[m]
        f = lambda dd: "—" if dd is None else f"{dd['mean'] * 100:+.2f}"
        L.append(f"| {NAMES[m]} | {fmt(e['last']['acc_pi'])} | {f(e['last'].get('vs_fixed'))} | {f(e.get('val_vs_fixed', {}).get('best'))} | "
                 f"{f(e.get('val_vs_fixed', {}).get('tail10'))} | {'—' if m == 'single' else (e['excluded_seeds'] or 'none')} |")
    L += ["", "## Robustness: fuzzy minus fixed from poor starting points (test set, best.pth)", "",
          "| Start (λ0, σ0, τ0) | fixed A_π | fuzzy A_π | Paired difference (pp) | Verdict | Excluded seeds fixed / fuzzy | Descriptive only: fuzzy minus fuzzy from the baseline start (pp) |", "|---|---|---|---|---|---|---|"]
    for st, r in robust.items():
        L.append(f"| {st} {r['start']} | {' / '.join(f'{v * 100:.2f}' for v in r['fixed'])} | {' / '.join(f'{v * 100:.2f}' for v in r['fuzzy'])} | "
                 f"{r['fuzzy_minus_fixed']['mean'] * 100:+.2f} ({', '.join(f'{v * 100:+.2f}' for v in r['fuzzy_minus_fixed']['d'])}) | {r['verdict']} | "
                 f"{len(r['excluded_seeds']['fixed'])} / {len(r['excluded_seeds']['fuzzy'])} | {r['fuzzy_minus_fuzzy_base']['mean'] * 100:+.2f} |")
    L += ["", "## Routing table: validation set, last 10 epochs (paper); the number outside the parentheses is the mean", "", "| Metric | " + " | ".join(NAMES[m] for m in MOE) + " |", "|---|" + "---|" * len(MOE)]
    for key, label, sc, nd in (("min_usage", "Minimum weighted usage (%)", 1, 1), ("min_top1_share", "Minimum top-1 share (%)", 1, 1),
                               ("nmi", "Class-expert normalized mutual information", 1, 3), ("assign_similarity", "Maximum assignment similarity between two experts", 1, 3)):
        L.append(f"| {label} | " + " | ".join(fmt(routing[m]['val'][key], sc, nd) for m in MOE) + " |")
    L += ["", "Supplement, test set best.pth: " + "; ".join(f"{NAMES[m]} minimum usage {routing[m]['test']['min_usage']['mean']:.1f}%, NMI {routing[m]['test']['nmi']['mean']:.3f}" for m in MOE)]
    L += ["", "## Timing (C, diagnostics off)", ""]
    if t_missing:
        L.append("Missing: " + ", ".join(t_missing))
    if timing:
        L += ["| Method | Total training time (min) | Time per epoch, from epoch 2 (s) |", "|---|---|---|"]
        L += [f"| {NAMES[m]} | {fmt(timing[m]['total_time_min'], 1, 1)} | {fmt(timing[m]['epoch_time_s'], 1, 1)} |" for m in METHODS if m in timing]
    L += ["", "## Expert function analysis (validation set, descriptive only)", ""]
    if experts is None:
        L.append(f"Missing: {ex_path} does not exist")
    else:
        L += ["| Method | A_π drop with top-1 masked (pp) | Top-2 masked (pp) | Normal minus 2 random experts (pp) | Gain, π-weighted (pp) | Mean CKA |", "|---|---|---|---|---|---|"]
        L += [f"| {NAMES[m]} | {fmt(experts[m]['delta1'])} | {fmt(experts[m]['delta2'])} | {fmt(experts[m]['normal_minus_random'])} | "
              f"{fmt(experts[m]['gain_pi_mean'])} | {fmt(experts[m]['cka_mean'], 1, 3)} |" for m in MOE if m in experts]
    (od / "final_tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    for m in (m for m in METHODS if m != "fixed"):
        print(f"{NAMES[m]:28s} ΔA_pi vs fixed (test, best.pth) {main_res[m]['best']['vs_fixed']['mean'] * 100:+.2f} pp -> {main_res[m]['verdict']}")
    for st, r in robust.items():
        print(f"robustness {st:15s} fuzzy - fixed {r['fuzzy_minus_fixed']['mean'] * 100:+.2f} pp -> {r['verdict']}")
    print(f"Written {od / 'final_summary.json'} and {od / 'final_tables.md'}")


if __name__ == "__main__":
    main()
