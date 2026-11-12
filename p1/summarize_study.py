#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
summarize_study.py

Summarize the pilot-study reruns (stages 3b, 4, 6, 7) under the rules in analysis/Pilot_Study.md and give the conclusions. Reads run directories only; writes results only when --out is given.
The study3b / study4 / study6 / study7 subcommands of run_pilot.sh call this script and read to_run (runs still to be done) and decision from its output.

Shared conventions:
- Fixed MoE runs are taken from --nw-dir (R4 or R0, no aux warmup, i.e. aux_warmup_frac = 0 in the config); single from --pilot-dir;
  auxfree from --auxfree-dir. Starting setting: augmentation B, lr 1e-3, wd 1e-4, 70 epoch, λ0 1.5, σ0 0.2, τ0 0.5.
- Stage 4.3 changed epochs from 70 to 50 (2026-10-31). The augmentation, lr and epoch steps and stage 3b are still conditioned on the starting setting; the wd step, stage 6 and stage 7 run at 50 epoch.
- X = max(0.3 pp, range of the best A_pi of R4 Fixed MoE seeds 0/1/2 under the starting setting) (stage 2), recomputed from the runs each time; threshold.json is not read.
- Exclusion: minimum weighted usage < 5% within the last 10 epochs of any seed.
- Multi-seed comparisons are paired by seed: better = mean d_k >= +X, non-inferior = mean d_k >= -X.

--part x   only compute and print X.
--part 3b  R4 vs R0 (starting setting, seeds 0/1/2): d_k = A_R4 - A_R0; confirm R4 if the mean >= -X, otherwise stop.
--part 4 --step aug|lr|epochs|wd
           Single (seed 0) and MoE (seed 0) each select by the single-seed rule (a tie keeps the default; for epochs the shortest is chosen); MoE decides.
           If MoE's single-seed choice differs from the starting value, confirm first: run seeds 1, 2 of that candidate and pair them with the 3 seeds of the starting setting;
           a deviation because the candidate is better (higher by X) requires mean d_k >= +X; a deviation because a tie picks a shorter epoch count requires mean d_k >= -X.
           If it still deviates from the starting value after confirmation, decision = changed (stop and discuss); otherwise keep.
--part 4s  single wd confirmation (50 epoch; added 2026-10-31 after seeing the wd results): if the wd selected by the single-seed rule for single seed 0 is not 1e-4, pair it with 1e-4 over seeds 0/1/2;
           mean d_k >= +X gives decision = changed (single uses its own wd, stop and discuss); otherwise keep (single uses MoE's wd). Does not affect MoE.
--part 6   fixed baseline calibration: λ {0.5, 4.5}, σ {0.1, 0.4}, τ {0.25, 1.0}, 3 seeds each, paired with the center point. Final values: candidates with mean d_k >= +X
           are adopted (for the same parameter, the one with the larger mean); if two or more parameters are adopted, the combined configuration is run; acceptable range: not excluded and mean d_k >= -X.
--part 6b  recheck after σ0 changed to 0.4 (added 2026-11-01 after seeing the stage 6 results, fixed before the recheck runs), done in order; on failure stop and discuss, settings are not changed automatically:
           (1) σ 0.8 vs 0.4, 3 seeds, stop if the mean >= +X; if the mean relative to the center point σ 0.2 is >= -X and not excluded, the σ range upper bound is extended to 0.8;
           (2) 50 vs 70 epoch, 3 seeds, stop if the mean < -X; (3) wd 0.05 vs 1e-4, 3 seeds, stop if the mean >= +X;
           (4) lr 3e-4, 3e-3 vs 1e-3, seed 0; if higher by X or more, run seeds 1, 2 and stop if the mean >= +X.
--part 8   stage 8 trial runs (--round N; directory --stage8-dir): fuzzy and schedule seed 0 are judged by control behavior (validation accuracy is not used). Pass criteria (fixed before the runs):
           no inference failure; exclusion condition not triggered; λ, σ, τ each reach at least two levels (binned by distance to each level's centroid); all 10 rules have an epoch with activation > 0 (R3, R9 may be 0);
           trajectory direction: early (epochs 4 ~ 19) mean σ > late (epochs 36 ~ 50), early mean λ > late, late mean τ < 0.5.
           Round 3 (fixed 2026-11-06, tests only the σ hypothesis; behavior criteria are only reported): paired mean of fuzzy minus round-2 fuzzy >= +X and of fuzzy minus fixed σ0 0.4
           >= -X gives sigma_cause; the first holds but the second does not gives sigma_partial; the first does not hold gives sigma_not_cause.
--part 7   auxfree γ {1e-4, 1e-3, 1e-2}, seed 0 collapse check and single-seed selection (a tie keeps 1e-3); if it deviates from 1e-3, run seeds 1, 2 of the candidate γ and of 1e-3
           for paired confirmation (>= +X).
--part 9a  bad-start probe (Pilot_Study.md 10.4; directory --stage9-dir): 12 candidates for fixed seed 0 (batch 1: λ0 0.17, 13.5; σ0 0, 1.6; τ0 0.125, 2.0;
           extension: λ0 0, 40.5; σ0 3.2; combinations (λ0, σ0, τ0) = (0, 0, 0.125), (40.5, 3.2, 2.0), (0, 3.2, 2.0)). When runs are missing, only the runs to do are listed.
           "Bad" = (1) minimum weighted usage < 5% in the last 10 epochs, or (2) 5 or more epochs after epoch 10 with usage < 5%, or
           (3) best A_pi lower than fixed σ0 0.4 seed 0 by X or more. 9b takes at most 3: those meeting (1)(2) first; within the same tier, single parameters before same-direction combinations,
           same-direction combinations before mixed-direction combinations, then by deviation multiple ascending (λ0 0 and the combinations have no finite multiple), then by the number of < 5% epochs descending in tier (1)(2) / by A_pi ascending in tier (3).
--part 9b  fuzzy vs fixed under bad starts (Pilot_Study.md 10.4; starts taken from selected_9b of --study9a). Step 1 confirms "bad": paired mean of fixed seeds 0/1/2 against the fixed baseline
           (σ0 0.4) <= -X, or criterion (1)(2) triggered in at least 2 seeds. Step 2, only for confirmed starts: paired mean of fuzzy (--fuzzy-config) minus fixed
           >= +X is better, within ±X is a tie, <= -X is worse. Runs are not dropped from accuracy because of exclusion; the number of excluded seeds is reported separately (A.1 item 19). When runs are missing, only the runs to do are listed.

Usage (in p1/):
    python summarize_study.py --part 4 --step lr [--out ../runs/stage6c/study4_lr.json]
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

from summarize_pilot import LAST_K, feq, find, fmt, load_runs
from analyze_calib import load_pi
from summarize_router import routing_report

SEEDS = (0, 1, 2)
START = {"aug": "B", "lr": 1e-3, "wd": 1e-4, "epochs": 70}
CHOSEN = {**START, "epochs": 50}  # setting after stage 4.3
CENTER ={"lambda": "1.5", "sigma": "0.2", "tau": "0.5"}
CANDS6 = {"lambda": ("0.5", "4.5"), "sigma": ("0.1", "0.4"), "tau": ("0.25", "1.0")}  # matches the directory naming in run_pilot.sh
STEPS = {  # stage 4: candidates, starting value
    "aug": (["A", "B"], "B"),
    "lr": ([1e-4, 3e-4, 1e-3, 3e-3], 1e-3),
    "epochs": ([50, 70, 100], 70),
    "wd": ([1e-4, 0.01, 0.05, 0.1], 1e-4),
}
GAMMAS = (1e-4, 1e-3, 1e-2)
SIGMA6B = "0.4"  # σ0 selected in stage 6; baseline of stage 6b
# Stage 9a candidates: (key, type, multiple of the deviation from the acceptable range, λ0, σ0, τ0); naming matches the directory names in run_pilot.sh (10.4).
# Type 0 single parameter, 1 same-direction combination, 2 mixed-direction combination; 9b ordering within the same criterion tier is by type first, then by multiple (None = no finite multiple, ranked last among single parameters).
# Batch 1 (study9a): one level outside the grid, σ0 0 counted as one level (2x). Extension (study9a_ext, added 2026-11-04 after seeing the batch 1 results, A.1 item 18): two levels outside, λ0 0 and three combinations
CANDS9A = [("lambda_0.17", 0, 3, "0.17", "0.4", "0.5"), ("lambda_13.5", 0, 3, "13.5", "0.4", "0.5"),
           ("sigma_0", 0, 2, "1.5", "0", "0.5"), ("sigma_1.6", 0, 2, "1.5", "1.6", "0.5"),
           ("tau_0.125", 0, 2, "1.5", "0.4", "0.125"), ("tau_2.0", 0, 2, "1.5", "0.4", "2.0"),
           ("lambda_0", 0, None, "0", "0.4", "0.5"), ("lambda_40.5", 0, 9, "40.5", "0.4", "0.5"),
           ("sigma_3.2", 0, 4, "1.5", "3.2", "0.5"),
           ("combo_collapse", 1, None, "0", "0", "0.125"), ("combo_flat", 1, None, "40.5", "3.2", "2.0"),
           ("combo_mixed", 2, None, "0", "3.2", "2.0")]
MAX_9B = 3
FUZZY_RANGE = {"lambda": (0.5, 4.5), "sigma": (0.1, 0.8)}  # output universe of fuzzy (9b only reports the epoch of returning into the universe)
MIN_USAGE = 5.0


class Store:
    """Cache load_runs results by (directory, routing parameters, router, warmup fraction), and attach the routing report to the MoE runs retrieved."""

    def __init__(self, args):
        self.args, self.cache = args, {}

    def get(self, d, model, seed, lam="1.5", sig="0.2", tau="0.5", router="R4", frac=0.0, **setting):
        cfg = {**START, **setting}
        k = (d, lam, sig, tau, router, frac)
        if k not in self.cache:
            self.cache[k] = load_runs(d, float(tau), float(lam), float(sig), router, frac) if Path(d).is_dir() else []
        r = find(self.cache[k], model, seed=seed, **cfg)
        if r is not None and model == "moe" and "min_usage_tail" not in r:
            r.update(dir=str(Path(d) / r["name"]), **routing_report(Path(d) / r["name"], Path(self.args.subset_dir)))
        return r


def compute_x(store, args):
    runs = [store.get(args.nw_dir, "moe", s) for s in SEEDS]
    if not all(runs):
        sys.exit(f"[Error] Center runs (R4, no aux warmup, start setting, seeds 0/1/2) incomplete in {args.nw_dir}; X undefined")
    acc = [r["acc_pi"] for r in runs]
    return max(0.003, max(acc) - min(acc)), runs


def chosen_center(store, args):
    """Center point (50 epoch) seeds 0/1/2 after stage 4.3. Its range is only reported; X is unchanged."""
    runs = [store.get(args.nw_dir, "moe", s, epochs=CHOSEN["epochs"]) for s in SEEDS]
    if not all(runs):
        sys.exit(f"[Error] {CHOSEN['epochs']}-epoch center runs (seeds 0/1/2) incomplete in {args.nw_dir}")
    acc = [r["acc_pi"] for r in runs]
    print(f"[Reference] {CHOSEN['epochs']}-epoch center seeds {', '.join(f'{a:.4f}' for a in acc)}; "
          f"range {(max(acc) - min(acc)) * 100:.2f} pp (reported only; X is unchanged)")
    return runs


def paired(a, b):
    """a, b: run lists of 3 seeds (candidate, reference). Returns the d_k list and the mean; returns (None, None) if any is missing."""
    if not (all(a) and all(b)):
        return None, None
    d = [x["acc_pi"] - y["acc_pi"] for x, y in zip(a, b)]
    return d, sum(d) / len(d)


def excluded(runs):
    return any(r["min_usage_tail"] < MIN_USAGE for r in runs if r and "min_usage_tail" in r)


def route_row(r):
    """Routing report of one run (report only, not used in the verdict)."""
    return {k: r.get(k) for k in ("min_usage_tail", "top1_share_min_tail", "w_share_min_tail", "nmi")}


def pp(v):
    return "-" if v is None else f"{v * 100:+.2f} pp"


def part_3b(store, args, x, center):
    r0 = [store.get(args.nw_dir, "moe", s, router="R0") for s in SEEDS]
    to_run = [{"router": "R0", "seed": s} for s, r in zip(SEEDS, r0) if r is None]
    res = {"to_run": to_run, "decision": "pending", "d": None, "d_mean": None}
    for s, a, b in zip(SEEDS, center, r0):
        print(f"  s{s}: R4 {a['acc_pi']:.4f}  R0 " + ("-" if b is None else f"{b['acc_pi']:.4f}"))
    if to_run:
        print(f"[To run] R0 seeds {[t['seed'] for t in to_run]}"); return res
    d, m = paired(center, r0)
    res.update(d=d, d_mean=m, r0_excluded=excluded(r0), r4_excluded=excluded(center))
    print("d_k = A_R4 - A_R0: " + ", ".join(pp(v) for v in d) + f" | mean {pp(m)}")
    if m >= -x and not excluded(center):
        res["decision"] = "confirm R4"; print("-> Confirm R4 (mean d_k is not below -X)")
    else:
        res["decision"] = "stop"; print("[STOP: discuss before continuing] R4 is not confirmed; later stages assume R4")
    return res


def part_4(store, args, x, center):
    values, start = STEPS[args.step]
    base = CHOSEN if args.step == "wd" else START  # wd comes after epoch, compared at 50 epoch
    if args.step == "wd":
        center = chosen_center(store, args)
    cfg = lambda v: {"epochs": base["epochs"], args.step: v}
    res ={"step": args.step, "to_run": [], "decision": "pending", "models": {}}

    def same(v, w):
        return v == w if isinstance(v, str) else feq(v, w)

    def pick(cands):  # single-seed rule: on a tie (difference from the highest < X) keep the starting value; the epoch step picks the shortest among the tied
        best = max(r["acc_pi"] for _, r in cands)
        tied = [v for v, r in cands if best - r["acc_pi"] < x]
        if args.step == "epochs":
            return min(tied), tied
        if any(same(v, start) for v in tied):
            return start, tied
        return max(cands, key=lambda c: c[1]["acc_pi"])[0], tied

    def run_spec(v, seed):  # to_run entry; lr / wd use the fixed notation accepted by run_pilot.sh
        c = {**base, args.step: v}
        return {"aug": c["aug"], "lr": fmt(c["lr"]), "wd": fmt(c["wd"]), "epochs": c["epochs"], "seed": seed}

    # Single (the epoch step does not run single)
    if args.step != "epochs":
        cands = [(v, store.get(args.pilot_dir, "single", 0, **cfg(v))) for v in values]
        if all(r for _, r in cands):
            p, tied = pick(cands)
            res["models"]["single"] = {"pick": p, "tied": tied, "acc": {fmt(v): r["acc_pi"] for v, r in cands}}
            print("[Single] " + ", ".join(f"{fmt(v)} {r['acc_pi']:.4f}" for v, r in cands) + f" -> {fmt(p)}")
        else:
            print(f"[Single] missing runs in {args.pilot_dir}: {[fmt(v) for v, r in cands if r is None]}")
    # MoE seed 0
    moe0 = [(v, center[0] if same(v, start) else store.get(args.nw_dir, "moe", 0, **cfg(v))) for v in values]
    missing = [v for v, r in moe0 if r is None]
    if missing:
        res["to_run"] = [run_spec(v, 0) for v in missing]
        print(f"[To run] MoE seed 0: {[fmt(v) for v in missing]}"); return res
    for v, r in moe0:
        print(f"[MoE s0] {fmt(v)}: {r['acc_pi']:.4f}" + ("  (excluded)" if excluded([r]) else ""))
    usable = [(v, r) for v, r in moe0 if not excluded([r])]
    p, tied = pick(usable)
    is_start = same(p, start)
    entry = {"pick_seed0": p, "tied": tied, "acc_seed0": {fmt(v): r["acc_pi"] for v, r in moe0}, "confirmed": None}
    res["models"]["moe"] = entry
    if is_start:
        res["decision"] = "keep"; print(f"-> MoE keeps the start value {fmt(start)}")
        return res
    # Deviation from the starting value: run seeds 1, 2 to confirm
    alt = [store.get(args.nw_dir, "moe", s, **cfg(p)) for s in SEEDS]
    need = [s for s, r in zip(SEEDS, alt) if r is None]
    if need:
        res["to_run"] = [run_spec(p, s) for s in need]
        print(f"[To run] confirm {fmt(p)}: MoE seeds {need}"); return res
    d, m = paired(alt, center)
    by_tie = args.step == "epochs" and p < start  # tie picks the shorter
    ok = (m >= -x if by_tie else m >= x) and not excluded(alt)
    entry.update(confirmed=ok, d=d, d_mean=m, criterion="non-inferiority (>= -X)" if by_tie else "superiority (>= +X)")
    print(f"Confirm {fmt(p)} vs start {fmt(start)}: d_k " + ", ".join(pp(v) for v in d) + f" | mean {pp(m)} ({entry['criterion']}): {ok}")
    if ok:
        res["decision"] = "changed"
        print(f"[STOP: discuss before continuing] the {args.step} choice changes from {fmt(start)} to {fmt(p)}"
              + (" (100 confirmed better: the 140-epoch extension rule applies after discussion)" if args.step == "epochs" and p == 100 else ""))
    else:
        res["decision"] = "keep"; print(f"-> Not confirmed; keep the start value {fmt(start)}")
    return res


def part_4s(store, args, x):
    values, start = STEPS["wd"]
    ep = CHOSEN["epochs"]
    get = lambda wd, s: store.get(args.pilot_dir, "single", s, epochs=ep, wd=wd)
    s0 = [(v, get(v, 0)) for v in values]
    if not all(r for _, r in s0):
        sys.exit(f"[Error] Single-model wd runs (seed 0) incomplete in {args.pilot_dir}")
    best = max(r["acc_pi"] for _, r in s0)
    tied = [v for v, r in s0 if best - r["acc_pi"] < x]
    p = start if start in tied else max(s0, key=lambda c: c[1]["acc_pi"])[0]
    res = {"pick_seed0": p, "tied": tied, "acc_seed0": {fmt(v): r["acc_pi"] for v, r in s0}, "decision": "keep", "d": None, "d_mean": None}
    print("[Single s0] " + ", ".join(f"{fmt(v)} {r['acc_pi']:.4f}" for v, r in s0) + f" -> {fmt(p)}")
    if p == start:
        print(f"-> The single model keeps wd {fmt(start)}"); return res
    alt, ref = [get(p, s) for s in SEEDS], [get(start, s) for s in SEEDS]
    if not (all(alt) and all(ref)):
        sys.exit(f"[Error] Single-model confirmation runs (wd {fmt(p)} and {fmt(start)}, seeds 0/1/2) incomplete in {args.pilot_dir}")
    d, m = paired(alt, ref)
    res.update(d=d, d_mean=m, acc_pick=[r["acc_pi"] for r in alt], acc_start=[r["acc_pi"] for r in ref], decision="changed" if m >= x else "keep")
    print(f"Confirm single wd {fmt(p)} vs {fmt(start)}: d_k " + ", ".join(pp(v) for v in d) + f" | mean {pp(m)} (superiority (>= +X)): {m >= x}")
    print(f"[STOP: discuss before continuing] the single model would use wd {fmt(p)}; the MoE settings are unaffected" if m >= x
          else f"-> Not confirmed; the single model uses wd {fmt(start)}")
    return res


def part_6(store, args, x):
    center = chosen_center(store, args)
    ep = CHOSEN["epochs"]
    res = {"to_run": [], "candidates": {}, "final": None, "acceptable": {}, "combination": None}
    adopted = {}
    for p, vals in CANDS6.items():
        for v in vals:
            c = {**CENTER, p: v}
            runs = [store.get(args.nw_dir, "moe", s, lam=c["lambda"], sig=c["sigma"], tau=c["tau"], epochs=ep) for s in SEEDS]
            need = [s for s, r in zip(SEEDS, runs) if r is None]
            res["to_run"] += [{**c, "seed": s} for s in need]
            d, m = paired(runs, center)
            ex = excluded(runs)
            row = res["candidates"][f"{p}={v}"] = {"d": d, "d_mean": m, "excluded": ex,
                                                   "acc": [None if r is None else r["acc_pi"] for r in runs]}
            print(f"{p}={v:5s} " + " ".join("   -   " if r is None else f"{r['acc_pi']:.4f}" for r in runs)
                  + (f" | d mean {pp(m)}" if m is not None else " | pending") + ("  (excluded)" if ex else ""))
            if m is not None and not ex and m >= x and (p not in adopted or m > adopted[p][1]):
                adopted[p] = (v, m)
    for p, vals in CANDS6.items():
        rows = {v: res["candidates"][f"{p}={v}"] for v in vals}
        if any(r["d_mean"] is None for r in rows.values()):
            res["acceptable"][p] = None; continue
        ok = [float(CENTER[p])] + [float(v) for v, r in rows.items() if not r["excluded"] and r["d_mean"] >= -x]
        res["acceptable"][p] = {"values": sorted(ok), "range": [min(ok), max(ok)]}
    if res["to_run"]:
        print(f"[To run] {len(res['to_run'])} candidate runs"); return res
    if len(adopted) >= 2:
        combo = {**CENTER, **{p: v for p, (v, _) in adopted.items()}}
        runs = [store.get(args.nw_dir, "moe", s, lam=combo["lambda"], sig=combo["sigma"], tau=combo["tau"], epochs=ep) for s in SEEDS]
        need = [s for s, r in zip(SEEDS, runs) if r is None]
        if need:
            res["to_run"] = [{**combo, "seed": s} for s in need]; print(f"[To run] combination {combo}"); return res
        d, m = paired(runs, center)
        ok = m >= x and not excluded(runs)
        res["combination"] = {"cfg": combo, "d": d, "d_mean": m, "adopted": ok}
        res["final"] = combo if ok else {**CENTER, max(adopted.items(), key=lambda kv: kv[1][1])[0]: max(adopted.items(), key=lambda kv: kv[1][1])[1][0]}
    elif adopted:
        (p, (v, _)), = adopted.items(); res["final"] = {**CENTER, p: v}
    else:
        res["final"] = dict(CENTER)
    res["changed"] = res["final"] != CENTER
    print("Acceptable ranges: " + ", ".join(f"{p} {a['values']}" for p, a in res["acceptable"].items()))
    print(f"Final: {res['final']}" + (" (center kept)" if not res["changed"] else
          "  [STOP: differs from the stage-1 provisional values; re-check stages 2 to 5]"))
    return res


def part_6b(store, args, x):
    """Stage 6b: recheck after σ0 changed to 0.4. Done in order; on failure decision is not keep, settings are not changed automatically, and later items are not run."""
    center = chosen_center(store, args)
    ep = CHOSEN["epochs"]
    get = lambda s, sig=SIGMA6B, lr=1e-3, wd=1e-4, epochs=ep: store.get(args.nw_dir, "moe", s, sig=sig, lr=lr, wd=wd, epochs=epochs)
    base = [get(s) for s in SEEDS]
    if not all(base):
        sys.exit(f"[Error] Base runs (sigma {SIGMA6B}, {ep} epochs, seeds 0/1/2) incomplete in {args.nw_dir}")
    res = {"to_run": [], "decision": "pending", "base_acc": [r["acc_pi"] for r in base], "base_range": max(r["acc_pi"] for r in base) - min(r["acc_pi"] for r in base),
           "base_routing": [route_row(r) for r in base], "checks": {}, "sigma_range": None}
    print(f"[Base] sigma {SIGMA6B}: " + ", ".join(f"{a:.4f}" for a in res["base_acc"]) + f"; range {res['base_range'] * 100:.2f} pp (reported only; X is unchanged)")

    def collect(label, runs, spec):
        """If runs are missing, write them to to_run and return True. spec: (σ, lr, wd, epochs); lr, wd match the directory naming in run_pilot.sh."""
        need = [s for s, r in zip(SEEDS[:len(runs)], runs) if r is None]
        res["to_run"] += [dict(zip(("sigma", "lr", "wd", "epochs"), spec), seed=s) for s in need]
        if need:
            print(f"[To run] {label}: seeds {need}")
        return bool(need)

    def pair(label, runs, ref, keep_if, stop):
        """Record one paired comparison. keep_if: mean d_k -> whether to keep the current setting. If not kept, decision = stop and return True."""
        d, m = paired(runs, ref)
        ex = excluded(runs)
        keep = ex or keep_if(m)
        res["checks"][label] = {"acc": [r["acc_pi"] for r in runs], "d": d, "d_mean": m, "excluded": ex, "keep": keep, "routing": [route_row(r) for r in runs]}
        print(f"{label}: d_k " + ", ".join(pp(v) for v in d) + f" | mean {pp(m)}" + ("  (excluded)" if ex else "") + f" -> {'keep' if keep else 'STOP'}")
        if not keep:
            res["decision"] = stop
        return not keep

    # 1. σ 0.8: relative to σ 0.4 decides whether the baseline changes; relative to the center point σ 0.2 decides the acceptable range (same as stage 6 rule 4)
    s08 = [get(s, sig="0.8") for s in SEEDS]
    if collect("sigma 0.8", s08, ("0.8", "1e-3", "1e-4", ep)):
        return res
    stop = pair("sigma 0.8 vs 0.4", s08, base, lambda m: m < x, "sigma_0.8")
    _, m_center = paired(s08, center)
    ok = not excluded(s08) and m_center >= -x
    res["sigma_range"] = {"d_mean_vs_center": m_center, "upper_bound": 0.8 if ok else 0.4}
    print(f"sigma 0.8 vs center: mean {pp(m_center)} -> sigma range upper bound {res['sigma_range']['upper_bound']}")
    if stop:
        print("[STOP: discuss before continuing] sigma 0.8 beats 0.4 by at least X"); return res

    # 2. 50 vs 70 epoch: keep 50 if non-inferior (d_k = A_50 - A_70). Unlike 4.3, only the two existing candidates are compared here
    e70 = [get(s, epochs=START["epochs"]) for s in SEEDS]
    if collect("70 epochs", e70, ("0.4", "1e-3", "1e-4", START["epochs"])):
        return res
    d, m = paired(base, e70)
    keep = m >= -x
    res["checks"]["50 vs 70 epochs"] = {"acc_70": [r["acc_pi"] for r in e70], "d": d, "d_mean": m, "keep": keep, "routing_70": [route_row(r) for r in e70]}
    print("50 vs 70 epochs: d_k " + ", ".join(pp(v) for v in d) + f" | mean {pp(m)} -> {'keep' if keep else 'STOP'}")
    if not keep:
        res["decision"] = "epochs"; print("[STOP: discuss before continuing] 50 epochs is below 70 by more than X"); return res

    # 3. wd 0.05
    w05 = [get(s, wd=0.05) for s in SEEDS]
    if collect("wd 0.05", w05, ("0.4", "1e-3", "0.05", ep)):
        return res
    if pair("wd 0.05 vs 1e-4", w05, base, lambda m: m < x, "wd_0.05"):
        print("[STOP: discuss before continuing] wd 0.05 beats 1e-4 by at least X"); return res

    # 4. both sides of lr, seed 0; if higher by X or more, run seeds 1, 2 to confirm
    cands = (("3e-4", 3e-4), ("3e-3", 3e-3))
    first = [get(0, lr=v) for _, v in cands]
    for (name, _), r in zip(cands, first):
        collect(f"lr {name} seed 0", [r], ("0.4", name, "1e-4", ep))
    if res["to_run"]:
        return res
    res["checks"]["lr seed 0"] = {name: {"acc": r["acc_pi"], "d": r["acc_pi"] - base[0]["acc_pi"], "excluded": excluded([r]), "routing": route_row(r)} for (name, _), r in zip(cands, first)}
    print("lr seed 0: " + ", ".join(f"{name} {r['acc_pi']:.4f} ({pp(r['acc_pi'] - base[0]['acc_pi'])})" for (name, _), r in zip(cands, first)))
    trig = [(name, v) for (name, v), r in zip(cands, first) if r["acc_pi"] - base[0]["acc_pi"] >= x and not excluded([r])]
    for name, v in trig:
        runs = [get(s, lr=v) for s in SEEDS]
        collect(f"lr {name} confirmation", runs, ("0.4", name, "1e-4", ep))
    if res["to_run"]:
        return res
    for name, v in trig:
        if pair(f"lr {name} vs 1e-3", [get(s, lr=v) for s in SEEDS], base, lambda m: m < x, f"lr_{name}"):
            print(f"[STOP: discuss before continuing] lr {name} beats 1e-3 by at least X"); return res
    res["decision"] = "keep"
    print(f"-> Recheck passed: sigma {SIGMA6B}, 50 epochs, wd 1e-4, lr 1e-3 are kept")
    return res


def part_7(store, args, x):
    found = {}
    d = Path(args.auxfree_dir)
    for run in (sorted(d.iterdir()) if d.is_dir() else []):
        cp = run / "config.json"
        if not (run.is_dir() and cp.exists() and (run / "summary.json").exists()):
            continue
        c = json.loads(cp.read_text(encoding="utf-8"))
        if c.get("controller") == "auxfree" and c.get("router") == "R4" and feq(c["tau0"], 0.5) and c["aug"] == CHOSEN["aug"] \
                and feq(c["lr"], CHOSEN["lr"]) and feq(c["weight_decay"], CHOSEN["wd"]) and c["epochs"] == CHOSEN["epochs"]:
            g = next((g for g in GAMMAS if feq(c["auxfree_gamma"], g)), None)
            if g is not None:
                k = (g, int(c["seed"]))
                if k in found:
                    sys.exit(f"[Error] Multiple auxfree runs for gamma {fmt(g)}, seed {k[1]}")
                summ = json.loads((run / "summary.json").read_text(encoding="utf-8"))
                found[k] = {"name": run.name, "acc_pi": summ["best_val_acc_pi"], "best_epoch": summ["best_epoch"],
                            **routing_report(run, Path(args.subset_dir))}
    res = {"to_run": [], "decision": "pending", "runs": {f"{fmt(g)}_s{s}": r for (g, s), r in found.items()}}
    need0 = [g for g in GAMMAS if (g, 0) not in found]
    if need0:
        res["to_run"] = [{"gamma": fmt(g), "seed": 0} for g in need0]; print(f"[To run] gamma {[fmt(g) for g in need0]} seed 0"); return res
    for g in GAMMAS:
        r = found[(g, 0)]
        print(f"gamma {fmt(g):5s} s0: A_pi {r['acc_pi']:.4f}, min usage {r['min_usage_tail']:.1f}%, top-1 share min {r['top1_share_min_tail']:.1f}%"
              + ("  (collapse: excluded)" if r["min_usage_tail"] < MIN_USAGE else ""))
    if found[(1e-3, 0)]["min_usage_tail"] < MIN_USAGE:
        res["decision"] = "stop"; print("[STOP: discuss before continuing] the default gamma 1e-3 collapses"); return res
    usable = [(g, found[(g, 0)]) for g in GAMMAS if found[(g, 0)]["min_usage_tail"] >= MIN_USAGE]
    best = max(r["acc_pi"] for _, r in usable)
    tied = [g for g, r in usable if best - r["acc_pi"] < x]
    p = 1e-3 if 1e-3 in tied else max(usable, key=lambda c: c[1]["acc_pi"])[0]
    res["pick_seed0"] = p
    if p == 1e-3:
        res["decision"] = "keep"; print("-> Keep gamma 1e-3"); return res
    need = [{"gamma": fmt(g), "seed": s} for g in (p, 1e-3) for s in SEEDS[1:] if (g, s) not in found]
    if need:
        res["to_run"] = need; print(f"[To run] confirm gamma {fmt(p)}: {need}"); return res
    dd = [found[(p, s)]["acc_pi"] - found[(1e-3, s)]["acc_pi"] for s in SEEDS]; m = sum(dd) / 3
    ok = m >= x and not any(found[(p, s)]["min_usage_tail"] < MIN_USAGE for s in SEEDS)
    res.update(d=dd, d_mean=m, decision="changed" if ok else "keep")
    print(f"Confirm gamma {fmt(p)} vs 1e-3: d_k " + ", ".join(pp(v) for v in dd) + f" | mean {pp(m)}: {ok}")
    print("[STOP: discuss before continuing] gamma changes" if ok else "-> Not confirmed; keep gamma 1e-3")
    return res


def part_8(store, args, x):
    import skfuzzy as fuzz
    d = Path(args.stage8_dir)
    runs = {}
    for run in (sorted(d.iterdir()) if d.is_dir() else []):
        cp = run / "config.json"
        if run.is_dir() and cp.exists() and (run / "summary.json").exists() and run.name.endswith(f"_r{args.round}"):
            c = json.loads(cp.read_text(encoding="utf-8"))
            if c.get("controller") in ("fuzzy", "schedule") and c["epochs"] == CHOSEN["epochs"]:
                runs[(c["controller"], int(c["seed"]))] = (run, c)
    # Round 1: fuzzy, schedule seed 0; from round 2 on (10.3): fuzzy seeds 0/1/2, plus a 3-seed pairing with fixed σ0 0.4
    need = [("fuzzy", 0), ("schedule", 0)] if args.round == 1 else [("fuzzy", s) for s in SEEDS]
    runs = {k: runs[k] for k in need if k in runs}
    res = {"round": args.round, "to_run": [{"controller": k, "seed": s} for k, s in need if (k, s) not in runs], "runs": {}, "decision": "pending"}
    if res["to_run"]:
        print(f"[To run] round {args.round}: {res['to_run']}"); return res
    passed = True
    for (ctrl, seed), (run, c) in runs.items():
        with open(run / "history.csv", newline="") as f:
            hist = list(csv.DictReader(f))
        ep = lambda r: int(r["epoch"])
        act = [r for r in hist if r.get("controller_active") == "True"]
        rep = routing_report(run, Path(args.subset_dir))
        summ = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        out = {"name": run.name, "acc_pi": summ["best_val_acc_pi"], "best_epoch": summ["best_epoch"], "min_usage_tail": rep["min_usage_tail"],
               "top1_share_min_tail": rep["top1_share_min_tail"], "w_share_min_tail": rep["w_share_min_tail"], "nmi": rep["nmi"],
               "excluded": rep["min_usage_tail"] < MIN_USAGE, "checks": {}}
        phase = lambda lo, hi, k: float(np.mean([float(r[k]) for r in hist if lo <= ep(r) <= hi]))
        out["trajectory"] = {k: {"ep4_19": phase(4, 19, k), "ep20_35": phase(20, 35, k), "ep36_50": phase(36, 50, k),
                                 "min": min(float(r[k]) for r in act), "max": max(float(r[k]) for r in act)} for k in ("lambda", "sigma", "tau")}
        tr = out["trajectory"]
        out["checks"]["direction"] = (tr["sigma"]["ep4_19"] > tr["sigma"]["ep36_50"] and tr["lambda"]["ep4_19"] > tr["lambda"]["ep36_50"]
                                      and tr["tau"]["ep36_50"] < 0.5)
        out["checks"]["not_excluded"] = not out["excluded"]
        if ctrl == "fuzzy":
            cfg = c["fuzzy_config"]
            out["fuzzy_failures"] = sum(r.get("fuzzy_ok") == "False" for r in act)
            out["fuzzy_modes"] = {m: sum(r.get("fuzzy_mode") == m for r in act) for m in ("early-idle", "early", "full")}
            out["checks"]["no_failure"] = out["fuzzy_failures"] == 0
            # Bin by distance to each level's centroid; λ, σ use output values, τ uses cumulative values (three levels relative to tau0 0.5: low < 0.4, high > 0.6)
            def centroids(spec):
                lo, hi, step = spec["range"]; u = np.linspace(lo, hi, int(round((hi - lo) / step)) + 1)
                return {term: float(fuzz.centroid(u, fuzz.trapmf(u, p) if kind == "trap" else fuzz.trimf(u, p))) for term, (kind, p) in spec["terms"].items()}
            bands = {}
            for key, name in (("lambda", "lambda_out"), ("sigma", "noise")):
                cen = centroids(cfg["outputs"][name])
                vals = [float(r[key]) for r in act]
                bands[key] = sorted({min(cen, key=lambda t: abs(cen[t] - v)) for v in vals})
            bands["tau"] = sorted({"Low" if float(r["tau"]) < 0.4 else "High" if float(r["tau"]) > 0.6 else "Mid" for r in act})
            out["bands_visited"] = bands
            out["checks"]["two_bands_each"] = all(len(b) >= 2 for b in bands.values())
            names = [ru["name"] for ru in cfg["rules"]]
            share = {n: float(np.mean([float(r.get(f"act_{n}") or 0) > 0 for r in act])) for n in names}
            out["rule_activation_share"] = share
            optional = {"R3", "R9"}
            out["checks"]["rules_fire"] = all(share[n] > 0 for n in names if n not in optional)
        ok = all(out["checks"].values())
        out["passed"] = ok; passed &= ok
        res["runs"][f"{ctrl}_s{seed}"] = out
        print(f"[{ctrl} s{seed}] {run.name}: A_pi {out['acc_pi']:.4f} (reported only); min usage tail {out['min_usage_tail']:.1f}%; "
              + "; ".join(f"{k} {'ok' if v else 'FAIL'}" for k, v in out["checks"].items()))
        for k, v in tr.items():
            print(f"   {k:6s} ep4-19 {v['ep4_19']:.3f} | ep20-35 {v['ep20_35']:.3f} | ep36-50 {v['ep36_50']:.3f} | range [{v['min']:.3f}, {v['max']:.3f}]")
        if ctrl == "fuzzy":
            print(f"   failures {out['fuzzy_failures']}, modes {out['fuzzy_modes']}, bands {out['bands_visited']}")
            print("   rule share>0: " + ", ".join(f"{n} {s:.2f}" for n, s in out["rule_activation_share"].items()))
    res["behavior_passed"] = passed
    if args.round == 1:
        res["decision"] = "pass" if passed else "adjust"
        print(f"-> Round 1: {'PASS (freeze the controller)' if passed else 'ADJUST membership functions or windows (not the rule directions), record the reason, and run the next round'}")
        return res
    fz = [{"acc_pi": res["runs"][f"fuzzy_s{s}"]["acc_pi"]} for s in SEEDS]
    fx = [store.get(args.nw_dir, "moe", s, sig="0.4", epochs=CHOSEN["epochs"]) for s in SEEDS]
    if not all(fx):
        sys.exit(f"[Error] fixed σ0 0.4, {CHOSEN['epochs']}-epoch runs (seeds 0/1/2) incomplete in {args.nw_dir}")
    d, mean = paired(fz, fx)
    res["paired_vs_fixed"] = {"fixed": [r["name"] for r in fx], "fixed_acc": [r["acc_pi"] for r in fx], "d": d, "mean": mean,
                              "non_inferior": mean >= -x}
    print(f"   paired vs fixed σ0 0.4: d_k {', '.join(pp(v) for v in d)}; mean {pp(mean)} (need >= -X {-x * 100:.2f} pp)")
    if args.round >= 3:
        # Round 3 (10.3, fixed 2026-11-06): differs from the previous round only in the σ of R5, R8, R9; the main verdict is the same-seed pairing with the previous round's fuzzy
        prev = {}
        for run in sorted(Path(args.stage8_dir).iterdir()):
            cp = run / "config.json"
            if run.is_dir() and cp.exists() and (run / "summary.json").exists() and run.name.endswith(f"_r{args.round - 1}"):
                c = json.loads(cp.read_text(encoding="utf-8"))
                if c.get("controller") == "fuzzy" and c["epochs"] == CHOSEN["epochs"]:
                    prev[int(c["seed"])] = {"name": run.name, "acc_pi": json.loads((run / "summary.json").read_text(encoding="utf-8"))["best_val_acc_pi"]}
        if not all(s in prev for s in SEEDS):
            sys.exit(f"[Error] round {args.round - 1} fuzzy runs (seeds 0/1/2) incomplete in {args.stage8_dir}")
        dp, mp = paired(fz, [prev[s] for s in SEEDS])
        res["paired_vs_prev_round"] = {"prev": [prev[s]["name"] for s in SEEDS], "prev_acc": [prev[s]["acc_pi"] for s in SEEDS], "d": dp, "mean": mp}
        print(f"   paired vs round {args.round - 1} fuzzy: d_k {', '.join(pp(v) for v in dp)}; mean {pp(mp)} (need >= +X {x * 100:.2f} pp)")
        res["decision"] = ("sigma_cause" if mean >= -x else "sigma_partial") if mp >= x else "sigma_not_cause"
        print(f"-> Round {args.round}: {res['decision']} (behavior checks reported only)")
        return res
    ok = passed and mean >= -x
    # If round 2 fails, stop tuning (10.3, fixed 2026-11-03); proceed to later stages with the round-2 version and report it as is
    res["decision"] = "pass" if ok else "stop"
    print(f"-> Round {args.round}: {'PASS (freeze the controller)' if ok else 'NOT PASSED: stop adjusting; freeze this round and report (10.3)'}")
    return res


def epoch_profile(run_dir, subset_dir):
    """Per-epoch minimum weighted usage (val_usage_weighted, the validation result of that epoch) and top-1 share of the weakest expert (val_routing.npz, π-weighted)."""
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    with open(run_dir / "history.csv", newline="") as f:
        hist = list(csv.DictReader(f))
    epochs = [int(r["epoch"]) for r in hist]
    usage = [min(json.loads(r["val_usage_weighted"])) for r in hist]
    top1 = None
    npz = run_dir / "val_routing.npz"
    if npz.exists():
        pi = load_pi(cfg, subset_dir)
        with open(subset_dir / cfg["val_list"], newline="") as f:
            y = np.array([int(r["class_idx"]) for r in csv.DictReader(f)])
        n_c = np.bincount(y, minlength=len(pi))
        t1 = np.load(npz)["topk"][:, :, 0].astype(int)
        top1 = [min(np.sum(pi * np.bincount(y[t == e], minlength=len(pi)) / n_c) * 100 for e in range(cfg["num_experts"])) for t in t1]
    return epochs, usage, top1


def fmt1(v):
    return "-" if v is None else f"{v:.1f}"


def part_9a(store, args, x):
    ref = store.get(args.nw_dir, "moe", 0, sig=SIGMA6B, epochs=CHOSEN["epochs"])
    if ref is None:
        sys.exit(f"[Error] Reference run (fixed σ0 {SIGMA6B}, seed 0, {CHOSEN['epochs']} epochs) not found in {args.nw_dir}")
    print(f"[Reference] fixed σ0 {SIGMA6B} seed 0: A_pi {ref['acc_pi']:.4f}")
    res = {"reference": {"name": ref["name"], "acc_pi": ref["acc_pi"]}, "to_run": [], "runs": {}, "decision": "pending"}
    found = {}
    for key, kind, mult, lam, sig, tau in CANDS9A:
        r = store.get(args.stage9_dir, "moe", 0, lam=lam, sig=sig, tau=tau, epochs=CHOSEN["epochs"])
        if r is None:
            res["to_run"].append({"candidate": key, "lambda0": lam, "sigma0": sig, "tau0": tau, "seed": 0})
        else:
            found[key] = (kind, mult, r)
    if res["to_run"]:
        print(f"[To run] {res['to_run']}"); return res
    for key, (kind, mult, r) in found.items():
        ep, usage, top1 = epoch_profile(Path(r["dir"]), Path(args.subset_dir))
        low = [e for e, u in zip(ep, usage) if u < MIN_USAGE]
        tail = usage[-LAST_K:]
        n_after10 = sum(e > 10 for e in low)
        crit = {"tail_collapse": r["min_usage_tail"] < MIN_USAGE, "mid_collapse": n_after10 >= 5,
                "acc_drop": ref["acc_pi"] - r["acc_pi"] >= x}
        out = {"name": r["name"], "kind": kind, "multiple": mult, "acc_pi": r["acc_pi"], "best_epoch": r["best_epoch"],
               "d_vs_ref": r["acc_pi"] - ref["acc_pi"], "criteria": crit, "bad": any(crit.values()),
               "level": 1 if (crit["tail_collapse"] or crit["mid_collapse"]) else 2 if crit["acc_drop"] else None,
               "first_low_epoch": low[0] if low else None, "n_low_ep1_11": sum(e <= 11 for e in low), "n_low_ep12_50": sum(e >= 12 for e in low),
               "n_low_after10": n_after10, "recovered": bool(low) and min(tail) >= MIN_USAGE,
               "min_usage_tail": r["min_usage_tail"], "top1_share_min_tail": r.get("top1_share_min_tail"),
               "n_top1_low": None if top1 is None else sum(v < MIN_USAGE for v in top1), "nmi": r.get("nmi")}
        res["runs"][key] = out
        print(f"{key:14s}: A_pi {r['acc_pi']:.4f} ({pp(out['d_vs_ref'])}), min usage tail {r['min_usage_tail']:.1f}%, "
              f"<5% epochs {out['n_low_ep1_11']} (ep1-11) / {out['n_low_ep12_50']} (ep12-50), first {out['first_low_epoch']}, "
              f"recovered {out['recovered']}, top-1 share min tail {fmt1(out['top1_share_min_tail'])}%, top-1 <5% epochs {out['n_top1_low']}"
              + ("; BAD (" + ", ".join(k for k, v in crit.items() if v) + ")" if out["bad"] else ""))
    bad = [(k, o) for k, o in res["runs"].items() if o["bad"]]
    bad.sort(key=lambda ko: (ko[1]["level"], ko[1]["kind"], math.inf if ko[1]["multiple"] is None else ko[1]["multiple"],
                             -ko[1]["n_low_after10"] if ko[1]["level"] == 1 else ko[1]["acc_pi"]))
    res["bad"] = [k for k, _ in bad]
    res["selected_9b"] = res["bad"][:MAX_9B]
    res["decision"] = "9b" if bad else "no_9b"
    print(f"-> Bad starts (ordered): {res['bad'] or 'none'}; 9b candidates (max {MAX_9B}): {res['selected_9b'] or 'none'}"
          + ("" if bad else "; fixed is insensitive to these starts, no 9b (10.4)"))
    return res


def fuzzy_runs(d, lam, sig, tau, fz_cfg, subset_dir):
    """Fuzzy runs in d with start (λ0, σ0, τ0), controller config content identical to fz_cfg, and otherwise the CHOSEN setting (R4, no aux warmup), returned by seed."""
    out = {}
    for run in (sorted(Path(d).iterdir()) if Path(d).is_dir() else []):
        cp = run / "config.json"
        if not (run.is_dir() and cp.exists() and (run / "summary.json").exists()):
            continue
        c = json.loads(cp.read_text(encoding="utf-8"))
        if not (c.get("controller") == "fuzzy" and c.get("fuzzy_config") == fz_cfg and c.get("router") == "R4"
                and feq(c["lambda0"], float(lam)) and feq(c["sigma0"], float(sig)) and feq(c["tau0"], float(tau))
                and feq(c.get("aux_warmup_frac", 1 / 3), 0.0) and c["aug"] == CHOSEN["aug"] and feq(c["lr"], CHOSEN["lr"])
                and feq(c["weight_decay"], CHOSEN["wd"]) and c["epochs"] == CHOSEN["epochs"]):
            continue
        seed = int(c["seed"])
        if seed in out:
            sys.exit(f"[Error] Multiple fuzzy runs match seed {seed}: {out[seed]['name']}, {run.name}")
        summ = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        out[seed] = {"name": run.name, "dir": str(run), "acc_pi": summ["best_val_acc_pi"], "best_epoch": summ["best_epoch"],
                     **routing_report(run, Path(subset_dir))}
    return out


def usage_flags(r, subset_dir):
    """Criteria (1)(2) (10.4) and imbalance timing of one run; criterion (1) is the same as the exclusion condition."""
    ep, usage, _ = epoch_profile(Path(r["dir"]), Path(subset_dir))
    low = [e for e, u in zip(ep, usage) if u < MIN_USAGE]
    n_after10 = sum(e > 10 for e in low)
    return {"tail_collapse": r["min_usage_tail"] < MIN_USAGE, "mid_collapse": n_after10 >= 5,
            "first_low_epoch": low[0] if low else None, "n_low_after10": n_after10,
            "recovered": bool(low) and min(usage[-LAST_K:]) >= MIN_USAGE, "min_usage_tail": r["min_usage_tail"]}


def back_in_range(run_dir):
    """Epoch from which λ, σ stay within the fuzzy output universe (report only); None if still outside the universe at the last epoch."""
    with open(Path(run_dir) / "history.csv", newline="") as f:
        hist = list(csv.DictReader(f))
    out = {}
    for k, (lo, hi) in FUZZY_RANGE.items():
        first = None
        for r in reversed(hist):
            if not lo - 1e-9 <= float(r[k]) <= hi + 1e-9:
                break
            first = int(r["epoch"])
        out[k] = first
    return out


def part_9b(store, args, x):
    sel_path = Path(args.study9a)
    if not sel_path.exists():
        sys.exit(f"[Error] {sel_path} not found; run study9a_ext first")
    sel = json.loads(sel_path.read_text(encoding="utf-8"))
    if sel.get("decision") != "9b":
        sys.exit(f"[Error] {sel_path}: decision is '{sel.get('decision')}', not '9b'")
    starts = {k: (lam, sig, tau) for k, _, _, lam, sig, tau in CANDS9A}
    fz_cfg = json.loads(Path(args.fuzzy_config).read_text(encoding="utf-8"))
    base_fx = [store.get(args.nw_dir, "moe", s, sig=SIGMA6B, epochs=CHOSEN["epochs"]) for s in SEEDS]
    if not all(base_fx):
        sys.exit(f"[Error] fixed σ0 {SIGMA6B}, {CHOSEN['epochs']}-epoch runs (seeds 0/1/2) incomplete in {args.nw_dir}")
    base_fz = fuzzy_runs(args.stage8_dir, "1.5", SIGMA6B, "0.5", fz_cfg, args.subset_dir)
    base_acc = ", ".join(f"{r['acc_pi']:.4f}" for r in base_fx)
    print(f"[Baseline] fixed σ0 {SIGMA6B}: {base_acc}; "
          f"fuzzy ({args.fuzzy_config}) seeds {sorted(base_fz)} in {args.stage8_dir}")
    res = {"selected": sel["selected_9b"], "baseline_fixed": {"names": [r["name"] for r in base_fx], "acc_pi": [r["acc_pi"] for r in base_fx]},
           "to_run": [], "starts": {}, "decision": "pending"}
    for key in sel["selected_9b"]:
        lam, sig, tau = starts[key]
        todo = lambda ctrl, seeds: [{"candidate": key, "controller": ctrl, "lambda0": lam, "sigma0": sig, "tau0": tau, "seed": s} for s in seeds]
        fx = [store.get(args.stage9_dir, "moe", s, lam=lam, sig=sig, tau=tau, epochs=CHOSEN["epochs"]) for s in SEEDS]
        miss = [s for s, r in zip(SEEDS, fx) if r is None]
        if miss:
            res["to_run"] += todo("fixed", miss); continue
        # Step 1: confirm "bad"
        flags = [usage_flags(r, args.subset_dir) for r in fx]
        d, mean = paired(fx, base_fx)
        n_usage = sum(f["tail_collapse"] or f["mid_collapse"] for f in flags)
        confirmed = mean <= -x or n_usage >= 2
        out = {"lambda0": lam, "sigma0": sig, "tau0": tau,
               "fixed": {"names": [r["name"] for r in fx], "acc_pi": [r["acc_pi"] for r in fx], "d_vs_baseline": d, "mean_vs_baseline": mean,
                         "n_usage_criterion": n_usage, "n_excluded": sum(f["tail_collapse"] for f in flags), "usage": flags},
               "confirmed": confirmed, "verdict": None}
        res["starts"][key] = out
        print(f"{key:14s} fixed: d vs baseline {', '.join(pp(v) for v in d)}; mean {pp(mean)} (<= -X {-x * 100:.2f} pp?); "
              f"usage criterion in {n_usage}/3 seeds -> {'CONFIRMED bad' if confirmed else 'not bad over 3 seeds'}")
        if not confirmed:
            out["verdict"] = "not_bad"; continue
        # Step 2: fuzzy minus fixed (same bad start); runs are not dropped from accuracy because of exclusion, exclusion is reported separately
        fz = fuzzy_runs(args.stage9_dir, lam, sig, tau, fz_cfg, args.subset_dir)
        miss = [s for s in SEEDS if s not in fz]
        if miss:
            res["to_run"] += todo("fuzzy", miss); continue
        fzl = [fz[s] for s in SEEDS]
        ffl = [usage_flags(r, args.subset_dir) for r in fzl]
        d2, mean2 = paired(fzl, fx)
        out["verdict"] = "better" if mean2 >= x else "worse" if mean2 <= -x else "tie"
        dz, meanz = paired(fzl, [base_fz.get(s) for s in SEEDS])
        out["fuzzy"] = {"names": [r["name"] for r in fzl], "acc_pi": [r["acc_pi"] for r in fzl], "d_vs_fixed": d2, "mean_vs_fixed": mean2,
                        "n_excluded": sum(f["tail_collapse"] for f in ffl), "usage": ffl,
                        "d_vs_fuzzy_baseline": dz, "mean_vs_fuzzy_baseline": meanz,
                        "back_in_range": [back_in_range(r["dir"]) for r in fzl]}
        print(f"{'':14s} fuzzy - fixed: d {', '.join(pp(v) for v in d2)}; mean {pp(mean2)} -> {out['verdict'].upper()}; "
              f"excluded seeds fixed {out['fixed']['n_excluded']}/3, fuzzy {out['fuzzy']['n_excluded']}/3")
        print(f"{'':14s} (reported only) fuzzy - fuzzy baseline: mean {'-' if meanz is None else pp(meanz)}; "
              f"λ, σ in range from epoch {[(b['lambda'], b['sigma']) for b in out['fuzzy']['back_in_range']]}; "
              f"first <5% epoch {[f['first_low_epoch'] for f in ffl]}, recovered {[f['recovered'] for f in ffl]}")
    if res["to_run"]:
        print(f"[To run] {res['to_run']}")
    else:
        res["decision"] = "done"
        print(f"-> 9b: " + "; ".join(f"{k} {o['verdict']}" for k, o in res["starts"].items()))
    return res


def main():
    ap = argparse.ArgumentParser(description="Summarize the Pilot_Study.md reruns (stages 3b, 4, 6, 7, 8, 9a, 9b)")
    ap.add_argument("--part", choices=["x", "3b", "4", "4s", "6", "6b", "7", "8", "9a", "9b"], required=True)
    ap.add_argument("--step", choices=list(STEPS), help="Stage 4 step")
    ap.add_argument("--nw-dir", default="../runs/stage6c", help="Fixed MoE runs without aux warmup (R4 and R0)")
    ap.add_argument("--pilot-dir", default="../runs/pilot", help="Single-model runs")
    ap.add_argument("--auxfree-dir", default="../runs/stage6", help="auxfree runs")
    ap.add_argument("--subset-dir", default="../datasets/subsets")
    ap.add_argument("--stage8-dir", default="../runs/stage8", help="Stage 8 trial runs (fuzzy, schedule)")
    ap.add_argument("--round", type=int, default=1, help="Stage 8 trial round")
    ap.add_argument("--stage9-dir", default="../runs/stage9", help="Stage 9 runs from bad starts (fixed, fuzzy)")
    ap.add_argument("--study9a", default="../runs/stage9/study9a_ext.json", help="Stage 9a result with the 9b starts (selected_9b)")
    ap.add_argument("--fuzzy-config", default="fuzzy_v2_r3.json", help="Frozen fuzzy controller for stage 9b (stage 8 round 3, adopted 2026-11-06)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.part == "4" and not args.step:
        ap.error("--part 4 requires --step")
    store = Store(args)
    x, center = compute_x(store, args)
    acc = [r["acc_pi"] for r in center]
    print(f"[Study {args.part}{' ' + args.step if args.step else ''}] X = {x * 100:.2f} pp "
          f"(center seeds {', '.join(f'{a:.4f}' for a in acc)}; range {(max(acc) - min(acc)) * 100:.2f} pp)")
    body = {"x": lambda: {}, "3b": lambda: part_3b(store, args, x, center), "4": lambda: part_4(store, args, x, center), "4s": lambda: part_4s(store, args, x),
            "6": lambda: part_6(store, args, x), "6b": lambda: part_6b(store, args, x), "8": lambda: part_8(store, args, x), "7": lambda: part_7(store, args, x),
            "9a": lambda: part_9a(store, args, x), "9b": lambda: part_9b(store, args, x)}[args.part]()
    result = {"part": args.part, "args": vars(args), "X": x, "center_acc": acc, **body}
    print("\nNote: X is a predefined tolerance, not a significance bound.")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
