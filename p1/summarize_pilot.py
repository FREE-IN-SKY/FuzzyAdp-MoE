#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
summarize_pilot.py

Summarize the pilot study (analysis/Pilot_Plan.md stages 0 to 4) and make selections by pre-set rules. Reads run directories only;
only --stage base writes threshold.json; the other stages write results only when --out is given.

Runs are identified by config.json fields; directory names are not parsed. For MoE only runs with controller=fixed and with τ0, λ0, σ0, router
matching the arguments are counted (an old config without the router field is treated as R0). Multiple runs matching the same configuration raise an error.

Rules (fixed before the pilot study started):
- Comparison metric: validation A_pi, taken at the epoch of each run's best checkpoint
- Tie threshold X: computed in the base stage from the range of the best A_pi of Fixed MoE seeds 0/1/2, X = max(0.003, range);
  not changed after being written to threshold.json; the other stages read it from that file. Single seed 0 is only reported and not used for X
- Candidates within X of the highest value are treated as tied. On a tie: augmentation picks B, lr picks 1e-3, epochs pick the shortest,
  wd picks 1e-4, and the lr recheck keeps the lr selected in stage 2
- lr candidates are {1e-4, 3e-4, 1e-3, 3e-3} (3e-3 is a one-step extension added 2026-10-25; no further extension)
- lr recheck (when the selected wd is not 1e-4): the selected lr is compared with the next smaller candidate, 1e-3 vs 3e-4, 3e-3 vs 1e-3
- When single and Fixed MoE selections differ, Fixed MoE decides
- Epoch extension rule: when 100 is the unique highest (all other candidates lower by X or more, i.e. it falls on the upper bound of the candidates),
  also run 140, then select among {100, 140} by the same rule; only this one step is extended

Supplementary report (not used for selection):
- Mean and range of A_pi over the last 10 epochs, mean BA, Macro-F1
- Steady-state per-epoch training / validation time (mean from epoch 2 on) and total_time_min
- When the candidate ranking by best A_pi differs from the ranking by tail mean, this is flagged as is; selection still follows the main rule
- X is a decision rule, not a statistical significance bound

Usage (in p1/):
    python summarize_pilot.py --stage base       --lambda0 L --sigma0 S
    python summarize_pilot.py --stage aug        --lambda0 L --sigma0 S
    python summarize_pilot.py --stage lr         --lambda0 L --sigma0 S --aug B
    python summarize_pilot.py --stage epochs     --lambda0 L --sigma0 S --aug B --lr 1e-3 [--wd 1e-4]
    python summarize_pilot.py --stage wd         --lambda0 L --sigma0 S --aug B --lr 1e-3 --epochs 70
    python summarize_pilot.py --stage lr-recheck --lambda0 L --sigma0 S --aug B --lr 1e-3 --epochs 70 --wd 0.05
"""

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

MODELS = {"single": "Single", "moe": "Fixed MoE"}
LAST_K = 10            # number of tail epochs used in the supplementary report
TIE_FLOOR = 0.003      # lower bound of X (0.3 pp)
BASE = {"aug": "B", "lr": 1e-3, "wd": 1e-4, "epochs": 70}  # starting defaults
BASE_SEEDS = (0, 1, 2)
EPOCH_CANDS = (50, 70, 100)
EPOCH_EXT = 140
LR_CANDS = (1e-4, 3e-4, 1e-3, 3e-3)
LR_RECHECK_LOWER = {1e-3: 3e-4, 3e-3: 1e-3}  # conditional recheck: selected lr -> next smaller candidate


def feq(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-6)


def same(a, b):
    return a == b if isinstance(a, str) else feq(a, b)


def fmt(x):
    """1e-3 -> '1e-3', 0.05 -> '0.05', 70 -> '70', 'B' -> 'B'."""
    if isinstance(x, str) or isinstance(x, int):
        return str(x)
    return f"{x:.0e}".replace("e-0", "e-") if x < 0.01 else f"{x:g}"


def mean_from(hist, col, start_epoch=2):
    vals = [float(r[col]) for r in hist if int(r["epoch"]) >= start_epoch and r.get(col) not in (None, "")]
    return sum(vals) / len(vals) if vals else None


def load_runs(pilot_dir, tau0, lam0, sig0, router, aux_warmup_frac=None):
    """aux_warmup_frac: when not None, keep only MoE runs with a matching aux loss warmup fraction (runs with and without warmup are distinguished from stage 6 C on; old configs are all 1/3)."""
    pilot_dir = Path(pilot_dir)
    if not pilot_dir.is_dir():
        sys.exit(f"[Error] Pilot directory does not exist: {pilot_dir}")
    runs = []
    for d in sorted(pilot_dir.iterdir()):
        cfg_p, sum_p, hist_p = d / "config.json", d / "summary.json", d / "history.csv"
        if not (d.is_dir() and cfg_p.exists()):
            continue
        cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
        if cfg["model"] == "moe":
            if cfg.get("controller") != "fixed":
                continue
            if not (feq(cfg["tau0"], tau0) and feq(cfg["lambda0"], lam0) and feq(cfg["sigma0"], sig0)
                    and cfg.get("router", "R0") == router):
                continue
            if aux_warmup_frac is not None and not feq(cfg.get("aux_warmup_frac", 1 / 3), aux_warmup_frac):
                continue
        if cfg.get("train_list") != f"train_s{cfg['seed']}.csv":
            sys.exit(f"[Error] {d.name}: training list {cfg.get('train_list')} does not match seed {cfg['seed']} (scheme A)")
        if not sum_p.exists():
            print(f"[Info] {d.name} has no summary.json and may be incomplete; skipped")
            continue
        summ = json.loads(sum_p.read_text(encoding="utf-8"))
        with open(hist_p, newline="") as f:
            hist = list(csv.DictReader(f))
        best = next(r for r in hist if int(r["epoch"]) == summ["best_epoch"])
        tail = hist[-LAST_K:]
        tail_pi = [float(r["val_acc_pi"]) for r in tail]
        runs.append({
            "name": d.name, "model": cfg["model"], "aug": cfg["aug"], "lr": float(cfg["lr"]),
            "aux_warmup_frac": float(cfg.get("aux_warmup_frac", 1 / 3)),
            "wd": float(cfg["weight_decay"]), "epochs": int(cfg["epochs"]), "seed": int(cfg["seed"]),
            "best_epoch": summ["best_epoch"],
            "acc_pi": float(best["val_acc_pi"]), "acc": float(best["val_acc"]),
            "ba": float(best["val_balanced_acc"]), "f1": float(best["val_macro_f1"]),
            "train_acc": float(best["train_acc"]), "time_min": summ.get("total_time_min"),
            "n_epochs_logged": len(hist),
            "tail_pi": sum(tail_pi) / len(tail_pi), "tail_min": min(tail_pi), "tail_max": max(tail_pi),
            "tail_ba": sum(float(r["val_balanced_acc"]) for r in tail) / len(tail),
            "tail_f1": sum(float(r["val_macro_f1"]) for r in tail) / len(tail),
            "train_s": mean_from(hist, "train_time_s"), "val_s": mean_from(hist, "val_time_s"),
        })
    return runs


def find(runs, model, aug, lr, wd, epochs, seed=0):
    m = [r for r in runs if r["model"] == model and r["aug"] == aug and feq(r["lr"], lr)
         and feq(r["wd"], wd) and r["epochs"] == epochs and r["seed"] == seed]
    if len(m) > 1:
        sys.exit(f"[Error] Multiple runs match the same configuration: {[r['name'] for r in m]}")
    return m[0] if m else None


def choose(cands, tie, prefer=None, shortest=False):
    """cands: [(value, run)]. Returns (selected value, list of tied values, explanation)."""
    best_val = max(r["acc_pi"] for _, r in cands)
    tied = [v for v, r in cands if best_val - r["acc_pi"] < tie]
    top = max(cands, key=lambda c: c[1]["acc_pi"])[0]
    if shortest:
        pick = min(tied)
        why = "tied; choose the shortest" if len(tied) > 1 else "highest A_pi; all others are at least X lower"
    elif prefer is not None and any(same(v, prefer) for v in tied):
        pick = next(v for v in tied if same(v, prefer))
        why = "tied; choose the default by rule" if len(tied) > 1 else "highest A_pi; all others are at least X lower"
    else:
        pick = top
        why = "highest A_pi; all others are at least X lower" if len(tied) == 1 else "tied without the default; choose the highest A_pi"
    return pick, tied, why


def ftime(x, digits=1):
    return "-" if x is None else f"{x:.{digits}f}"


def print_table(rows, label):
    print("  [Primary metric: values at the best epoch]")
    print(f"  {label:>8s} | {'A_pi':>7s} {'acc':>7s} {'BA':>7s} {'F1':>7s} {'train_acc':>9s} {'best_ep':>7s}/{'ep':<4s}")
    for v, r in rows:
        print(f"  {fmt(v):>8s} | {r['acc_pi']:.4f} {r['acc']:.4f} {r['ba']:.4f} {r['f1']:.4f} "
              f"{r['train_acc']:>9.4f} {r['best_epoch']:>7d}/{r['epochs']:<4d}")
    print(f"  [Supplementary, not used for selection: last {LAST_K} epochs; steady-state time = mean from epoch 2]")
    print(f"  {label:>8s} | {'A_pi_mean':>9s} {'A_pi_range':>15s} {'BA_mean':>7s} {'F1_mean':>7s} "
          f"{'train_s/ep':>10s} {'val_s/ep':>8s} {'total_min':>9s}")
    for v, r in rows:
        if r["n_epochs_logged"] != r["epochs"]:
            print(f"  [Warning] {fmt(v)}: history.csv has {r['n_epochs_logged']} epochs, but {r['epochs']} were configured")
        print(f"  {fmt(v):>8s} | {r['tail_pi']:9.4f} {r['tail_min']:.4f}~{r['tail_max']:.4f} "
              f"{r['tail_ba']:7.4f} {r['tail_f1']:7.4f} {ftime(r['train_s']):>10s} {ftime(r['val_s']):>8s} "
              f"{ftime(r['time_min']):>9s}")


def check_order(cands):
    """Compare the candidate rankings by best A_pi and by tail mean; flag if they differ."""
    by_best = [fmt(v) for v, r in sorted(cands, key=lambda c: -c[1]["acc_pi"])]
    by_tail = [fmt(v) for v, r in sorted(cands, key=lambda c: -c[1]["tail_pi"])]
    if by_best == by_tail:
        print(f"  Consistent ranking: {' > '.join(by_best)} (by best value and by last-epoch mean)")
        return True
    print(f"  [Inconsistent] By best A_pi: {' > '.join(by_best)}; by mean of last {LAST_K} epochs: {' > '.join(by_tail)}")
    print("           Selection still follows the primary rule; report this honestly")
    return False


def late_note(cands):
    late = [v for v, r in cands if r["best_epoch"] >= r["epochs"] - 1]
    if late:
        print(f"  [Info] Best epoch of {[fmt(v) for v in late]} is within the last 2 epochs; may not have converged (heuristic, not used for decisions)")


def compare(cands, label, tie, prefer=None, shortest=False):
    """Print and select. Returns None if runs are missing."""
    missing = [fmt(v) for v, r in cands if r is None]
    if missing:
        print(f"  Missing runs: {label}={missing}; cannot select")
        return None
    print_table(cands, label)
    pick, tied, why = choose(cands, tie, prefer=prefer, shortest=shortest)
    print(f"  -> Select {fmt(pick)} ({why}; tied candidates: {[fmt(t) for t in tied]})")
    check_order(cands)
    late_note(cands)
    return {"pick": pick, "tied": tied, "why": why,
            "runs": {fmt(v): {k: r[k] for k in ("name", "acc_pi", "best_epoch", "tail_pi")} for v, r in cands}}


def final_pick(picks):
    """When both models have a selection, Fixed MoE decides."""
    if "moe" not in picks or picks["moe"] is None:
        print("\nFinal selection: none (Fixed MoE runs are missing)")
        return None
    m = picks["moe"]["pick"]
    s = picks.get("single")
    print()
    if s is None:
        print(f"Final selection: {fmt(m)} (single runs are missing; based on Fixed MoE only)")
    elif same(s["pick"], m):
        print(f"Final selection: {fmt(m)} (both models agree)")
    else:
        print(f"Final selection: {fmt(m)} (models disagree: single selects {fmt(s['pick'])}; Fixed MoE takes precedence by rule; disclose in the paper)")
    return m


def stage_base(runs, args, thr_path):
    print(f"[Stage 0] Configuration: augmentation {BASE['aug']}, lr {fmt(BASE['lr'])}, wd {fmt(BASE['wd'])}, {BASE['epochs']} epochs")
    moe = [(s, find(runs, "moe", BASE["aug"], BASE["lr"], BASE["wd"], BASE["epochs"], s)) for s in BASE_SEEDS]
    single = find(runs, "single", BASE["aug"], BASE["lr"], BASE["wd"], BASE["epochs"], 0)
    missing = [f"MoE seed {s}" for s, r in moe if r is None] + (["single seed 0"] if single is None else [])
    if missing:
        print(f"  Missing runs: {missing}; cannot compute X")
        return None
    print("\n[Fixed MoE, by seed]")
    print_table(moe, "seed")
    vals = [r["acc_pi"] for _, r in moe]
    rng = max(vals) - min(vals)
    x = max(TIE_FLOOR, rng)
    print(f"  Mean best A_pi {sum(vals) / len(vals):.4f}, range {rng * 100:.2f} pp -> X = max(0.30, {rng * 100:.2f}) = {x * 100:.2f} pp")
    print("\n[Single, seed 0, reported only, not used for X]")
    print_table([(0, single)], "seed")
    print(f"  Single - Fixed MoE seed 0: {(single['acc_pi'] - moe[0][1]['acc_pi']) * 100:+.2f} pp (single seed; preliminary comparison only)")

    record = {"X": x, "range": rng, "floor": TIE_FLOOR,
              "moe_runs": [{"seed": s, "name": r["name"], "acc_pi": r["acc_pi"]} for s, r in moe],
              "single_run": {"name": single["name"], "acc_pi": single["acc_pi"]},
              "routing": {"tau0": args.tau0, "lambda0": args.lambda0, "sigma0": args.sigma0, "router": args.router},
              "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if thr_path.exists():
        old = json.loads(thr_path.read_text(encoding="utf-8"))
        print(f"\n[Info] {thr_path} already exists (X = {old['X'] * 100:.2f} pp); not overwritten")
        if not feq(old["X"], x):
            print(f"[Warning] The recomputed X = {x * 100:.2f} pp differs from the recorded value; the recorded value is used. Check whether runs were modified")
    else:
        thr_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote {thr_path}; later stages use X = {x * 100:.2f} pp and it will not be changed")
    return record


def load_threshold(thr_path, args):
    if not thr_path.exists():
        sys.exit(f"[Error] {thr_path} not found; run --stage base first")
    t = json.loads(thr_path.read_text(encoding="utf-8"))
    r = t["routing"]
    if not (feq(r["tau0"], args.tau0) and feq(r["lambda0"], args.lambda0) and feq(r["sigma0"], args.sigma0)
            and r["router"] == args.router):
        sys.exit(f"[Error] Routing settings in threshold.json {r} do not match the current arguments")
    return t["X"]


def need(args, *names):
    miss = [n for n in names if getattr(args, n.replace("-", "_")) is None]
    if miss:
        sys.exit(f"[Error] --stage {args.stage} requires {' '.join('--' + n for n in miss)}")


def main():
    ap = argparse.ArgumentParser(description="Summarize pilot runs and select settings by the predefined rules")
    ap.add_argument("--stage", choices=["base", "aug", "lr", "epochs", "wd", "lr-recheck"], required=True)
    ap.add_argument("--pilot-dir", default="../runs/pilot")
    ap.add_argument("--tau0", type=float, default=0.5)
    ap.add_argument("--lambda0", type=float, required=True, help="lambda0 selected in stage -1")
    ap.add_argument("--sigma0", type=float, required=True, help="sigma0 selected in stage -1")
    ap.add_argument("--router", default="R0")
    ap.add_argument("--aug", choices=["A", "B"])
    ap.add_argument("--lr", type=float)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--wd", type=float)
    ap.add_argument("--out", default=None, help="Optional JSON file for this stage's results")
    args = ap.parse_args()

    runs = load_runs(args.pilot_dir, args.tau0, args.lambda0, args.sigma0, args.router)
    thr_path = Path(args.pilot_dir) / "threshold.json"
    result = {"stage": args.stage, "args": {k: v for k, v in vars(args).items() if k != "out"}}

    if args.stage == "base":
        result["base"] = stage_base(runs, args, thr_path)
    else:
        x = load_threshold(thr_path, args)
        result["X"] = x
        print(f"[{args.stage}] Tie threshold X = {x * 100:.2f} pp (from {thr_path})")
        picks = {}
        if args.stage in ("aug", "lr", "wd", "lr-recheck"):
            if args.stage == "aug":
                values, prefer, label = ["A", "B"], "B", "aug"
                cfg = lambda v: dict(aug=v, lr=BASE["lr"], wd=BASE["wd"], epochs=BASE["epochs"])
            elif args.stage == "lr":
                need(args, "aug")
                values, prefer, label = list(LR_CANDS), 1e-3, "lr"
                cfg = lambda v: dict(aug=args.aug, lr=v, wd=BASE["wd"], epochs=BASE["epochs"])
            elif args.stage == "wd":
                need(args, "aug", "lr", "epochs")
                values, prefer, label = [1e-4, 0.01, 0.05, 0.1], 1e-4, "wd"
                cfg = lambda v: dict(aug=args.aug, lr=args.lr, wd=v, epochs=args.epochs)
            else:
                need(args, "aug", "lr", "epochs", "wd")
                lower = next((v for k, v in LR_RECHECK_LOWER.items() if feq(k, args.lr)), None)
                if lower is None:
                    sys.exit(f"[Error] The lr recheck is defined only for a selected lr in {[fmt(k) for k in LR_RECHECK_LOWER]}; got {fmt(args.lr)}")
                values, prefer, label = [lower, args.lr], args.lr, "lr"
                cfg = lambda v: dict(aug=args.aug, lr=v, wd=args.wd, epochs=args.epochs)
            for model in ("single", "moe"):
                print(f"\n[{MODELS[model]}]")
                cands = [(v, find(runs, model, **cfg(v))) for v in values]
                picks[model] = compare(cands, label, x, prefer=prefer)
            result["models"] = picks
            result["final"] = final_pick(picks)
            if args.stage == "wd" and result["final"] is not None and not feq(result["final"], 1e-4):
                lower = next((v for k, v in LR_RECHECK_LOWER.items() if feq(k, args.lr)), None)
                if lower is not None:
                    print(f"[Info] The selected wd is not 1e-4: per the recheck rule, run lr={fmt(lower)} at this wd (single + MoE), then run --stage lr-recheck")
                else:
                    print(f"[Info] The selected wd is not 1e-4, but no lr recheck is defined for lr={fmt(args.lr)} (see Pilot_Plan.md)")
        else:  # epochs
            need(args, "aug", "lr")
            if args.wd is None:  # default 1e-4; another value is passed only when going back to re-select epochs after the stage 4 conditional recheck selected the smaller lr
                args.wd = BASE["wd"]
                result["args"]["wd"] = args.wd
            print(f"\n[Fixed MoE] wd = {fmt(args.wd)}")
            cands = [(v, find(runs, "moe", args.aug, args.lr, args.wd, v)) for v in EPOCH_CANDS]
            res = compare(cands, "epochs", x, shortest=True)
            result["models"] = {"moe": res}
            result["final"] = res["pick"] if res else None
            r_ext = find(runs, "moe", args.aug, args.lr, args.wd, EPOCH_EXT)
            if res is not None:
                r100 = dict(cands)[EPOCH_CANDS[-1]]
                triggered = res["tied"] == [EPOCH_CANDS[-1]]
                result["extension_triggered"] = triggered
                if triggered and r_ext is None:
                    print(f"\n[Extension triggered] {EPOCH_CANDS[-1]} beats the other candidates by at least X (upper boundary of the grid); "
                          f"run {EPOCH_EXT} epochs, then rerun this command. No final selection yet")
                    result["final"] = None
                elif triggered:
                    print(f"\n[Extension] Select from {{{EPOCH_CANDS[-1]}, {EPOCH_EXT}}} by the same rule:")
                    ext = compare([(EPOCH_CANDS[-1], r100), (EPOCH_EXT, r_ext)], "epochs", x, shortest=True)
                    result["extension"] = ext
                    result["final"] = ext["pick"]
                elif r_ext is not None:
                    print(f"\n[Warning] A {EPOCH_EXT}-epoch run exists, but the extension rule was not triggered; it is not used")
            if result["final"] is not None:
                print(f"\nFinal selection: epochs={result['final']}")

    print("\nNote: single-seed comparison. X is a predefined decision threshold, not a significance bound; a gap below X "
          "does not mean no difference, and a gap above X does not mean a significant one. The rules prevent post-hoc "
          "selection; they do not guarantee the optimum.")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
