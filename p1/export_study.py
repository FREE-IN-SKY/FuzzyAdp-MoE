#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_study.py

Copy the runs used by analysis/Pilot_Study.md from runs/ to a standalone public record directory (default ../release/pilot_study/) and generate an index.
Reads runs/ only; the original directories are not changed. The manifests (MAIN, SUPERSEDED, SUMMARIES) are the single source of "which runs each section of the document uses".

- Directories are named by document stage: runs/stage2_noise_baseline/, runs/stage3b_router/ ...; the old-condition runs of Appendix B are in superseded/B1_calibration/ ...
  The historical directory names under runs/ (pilot, router, stage6c, etc.) do not appear in public paths.
- Each run is copied only once, into the stage where it first appears; later stages reference that location in INDEX.md.
- Run directories keep their original names (which encode the configuration); console logs are copied as console.log / eval.log in the run directory.
- .pth files are not copied by default (not present on the analysis laptop; about 37 MB each). Add --with-checkpoints on the experiment machine to copy them too.
- Required runs that have not finished are marked pending in INDEX.md without an error (error with --strict). After the reruns finish, run this script again to update.

Usage (in p1/):
    python export_study.py [--out ../release/pilot_study] [--overwrite] [--with-checkpoints] [--strict]
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from summarize_pilot import fmt


def moe(router="R4", lam="1.5", sig="0.2", tau="0.5", aw0=True, aug="B", lr="1e-3", wd="1e-4", ep=70, seed=0, ctrl="fixed", gamma="1e-3"):
    tau_tag = "" if tau == "0.5" else f"_tau{tau}"
    g_tag = "" if (ctrl != "auxfree" or gamma == "1e-3") else f"_g{gamma}"
    return f"moe-{ctrl}_{router}_lam{lam}_sig{sig}{tau_tag}{'_aw0' if aw0 else ''}{g_tag}_aug{aug}_lr{lr}_wd{wd}_e{ep}_s{seed}"


def single(aug="B", lr="1e-3", wd="1e-4", ep=70, seed=0):
    return f"single_aug{aug}_lr{lr}_wd{wd}_e{ep}_s{seed}"


S3 = (0, 1, 2)
EP = 50  # selection of stage 4.3; the wd step and stages 5 ~ 7 use this setting
CENTER = [("stage6c", moe(seed=s), True) for s in S3]  # starting setting (70 epoch)
CENTER_EP = [("stage6c", moe(ep=EP, seed=s), True) for s in S3]
STAGE4_MOE = [dict(aug="A")] + [dict(lr=v) for v in ("1e-4", "3e-4", "3e-3")] + [dict(ep=100)] \
             + [dict(wd=v, ep=EP) for v in ("0.01", "0.05", "0.1")]
STAGE6 = [dict(lam="0.5"), dict(lam="4.5"), dict(sig="0.1"), dict(sig="0.4"), dict(tau="0.25"), dict(tau="1.0")]
AUXFREE = dict(lam="0", sig="0", aw0=False, ctrl="auxfree")

# (public directory name or None, document section, [(source directory under runs/, run name, required)]); not required = exists only when a rule triggers (confirmation runs, combined configuration, etc.).
# Sections whose directory name is None copy no runs; they only reference in INDEX.md runs already copied by other sections.
MAIN = [
    ("stage2_noise_baseline", "Stage 2: noise baseline and X", CENTER + [("pilot", single(), True)]),
    ("stage3a_aux_warmup", "Stage 3a: removing the aux loss warmup", [("router", moe(aw0=False, seed=s), True) for s in S3] + CENTER),
    ("stage3b_router", "Stage 3b: R4 vs R0", [("stage6c", moe(router="R0", seed=s), True) for s in S3] + CENTER),
    ("stage4_training", "Stage 4: training hyperparameters",
     [("pilot", single(aug="A"), True)] + [("pilot", single(lr=v), True) for v in ("1e-4", "3e-4", "3e-3")]
     + [("pilot", single(wd=v, ep=EP), True) for v in ("1e-4", "0.01", "0.05", "0.1")]
     + [("pilot", single(wd=v, ep=EP, seed=s), True) for v in ("1e-4", "0.01") for s in (1, 2)]
     + [("stage6c", moe(seed=0, **c), True) for c in STAGE4_MOE] + [("stage6c", moe(seed=s, **c), False) for c in STAGE4_MOE for s in (1, 2)]
     + [("stage6c", moe(ep=140, seed=s), False) for s in S3] + CENTER_EP + CENTER),
    (None, "Stage 5: router learning-rate diagnosis (reanalyzed data)", CENTER_EP + [("stage6c", moe(ep=EP, seed=0, **c), True) for c in STAGE6]),
    ("stage6_fixed_calibration", "Stage 6: fixed baseline calibration", [("stage6c", moe(ep=EP, seed=s, **c), True) for c in STAGE6 for s in S3]
     + [("stage6c", moe(ep=EP, lam=l, sig=g, tau=t, seed=s), False) for l in ("0.5", "4.5") for g in ("0.1", "0.4") for t in ("0.5",) for s in S3]
     + CENTER_EP),
    # σ 0.8 always runs; the others run only after the previous item passes. The 70-epoch seed 0 of σ 0.4 is also listed in Appendix B.7; it is saved only once, here
    ("stage6b_recheck", "Stage 6b: recheck after σ0 = 0.4", [("stage6c", moe(ep=EP, sig="0.8", seed=s), True) for s in S3]
     + [("stage6c", moe(ep=70, sig="0.4", seed=s), s == 0) for s in S3]
     + [("stage6c", moe(ep=EP, sig="0.4", wd="0.05", seed=s), False) for s in S3]
     + [("stage6c", moe(ep=EP, sig="0.4", lr=v, seed=s), False) for v in ("3e-4", "3e-3") for s in S3]
     + [("stage6c", moe(ep=EP, sig="0.4", seed=s), True) for s in S3]),
    # Stage 8 trial runs: one fuzzy and one schedule run per round (directory name _r<round>), at most 3 rounds; round 1 always runs. The frozen fuzzy seeds 1, 2 and the 70-epoch control are listed separately
    ("stage8_controller", "Stage 8: controller trial runs (fuzzy, schedule)",
     [("stage8", moe(ep=EP, sig="0.4", ctrl=c, seed=0) + "_r1", True) for c in ("fuzzy", "schedule")]
     + [("stage8", moe(ep=EP, sig="0.4", ctrl="fuzzy", seed=s), True) for s in (1, 2)]
     + [("stage8", moe(ep=EP, sig="0.4", ctrl="fuzzy", seed=s) + "_r2", False) for s in S3]
     + [("stage8", moe(ep=70, sig="0.4", ctrl="fuzzy", seed=s), False) for s in S3]),
    ("stage7_auxfree", "Stage 7: auxfree", [("stage6", moe(ep=EP, gamma=g, seed=0, **AUXFREE), True) for g in ("1e-3", "1e-4", "0.01")]
     + [("stage6", moe(ep=EP, gamma=g, seed=s, **AUXFREE), False) for g in ("1e-3", "1e-4", "0.01") for s in (1, 2)]),
]

# Document Appendix B: old-condition runs that were rerun or bypassed. (public directory name, section, source directory, list of run names or "*")
SUPERSEDED = [
    ("B1_calibration", "B.1 Routing default calibration (former stage −1)", "calib", "*"),
    ("B2_noise_baseline_R0", "B.2 Noise baseline (former stage 0)", "pilot", [moe(router="R0", aw0=False, seed=s) for s in S3]),
    ("B3_training_R0", "B.3 MoE side of the training hyperparameters (former stages 1 ~ 4)", "pilot",
     [moe(router="R0", aw0=False, aug="A")] + [moe(router="R0", aw0=False, lr=v) for v in ("1e-4", "3e-4", "3e-3")]
     + [moe(router="R0", aw0=False, ep=v) for v in (50, 100)] + [moe(router="R0", aw0=False, wd=v) for v in ("0.01", "0.05", "0.1")]),
    ("B4_router_with_warmup", "B.4 Router stage (former stage 5)", "router", [moe(router="R0", aw0=False, seed=s) for s in (1, 2)]),
    ("B5_repro", "B.5 Router learning-rate checkpoint (reproduction run)", "repro", [moe(router="R0", aw0=False, seed=0)]),
    ("B6_fixed_calibration_with_warmup", "B.6 Fixed baseline calibration (former stage 6 A)", "stage6",
     [moe(aw0=False, seed=0, **c) for c in STAGE6] + [moe(aw0=False, lam="4.5", seed=s) for s in (1, 2)]),
    ("B7_70_epoch/fixed_calibration", "B.7 70-epoch runs before the switch to 50 epochs: fixed calibration", "stage6c",
     [moe(seed=0, **c) for c in STAGE6] + [moe(lam="4.5", seed=s) for s in (1, 2)]),
    ("B7_70_epoch/auxfree", "B.7 70-epoch runs before the switch to 50 epochs: auxfree", "stage6", [moe(gamma="1e-3", seed=0, **AUXFREE)]),
    ("B7_70_epoch/single_wd", "B.7 70-epoch runs before the switch to 50 epochs: single wd", "pilot", [single(wd=v) for v in ("0.01", "0.05", "0.1")]),
]

# Summary results: (source directory, file name, public file name)
SUMMARIES = {
    "main": [("stage6c", "stage6c.json", "stage3a_aux_warmup.json"), ("stage6c", "study3b.json", "stage3b_router.json")]
            + [("stage6c", f"study4_{s}.json", f"stage4_{s}.json") for s in ("aug", "lr", "epochs", "wd")]
            + [("pilot", "study4_wd_single.json", "stage4_wd_single.json")]
            + [("stage6c", "study6.json", "stage6_fixed_calibration.json"), ("stage6c", "study6b.json", "stage6b_recheck.json"),
               ("stage8", "study8_r1.json", "stage8_round1.json"), ("stage8", "study8_r2.json", "stage8_round2.json"),
               ("stage6", "study7.json", "stage7_auxfree_gamma.json")],
    "superseded": [("calib", "step1.json", "B1_calibration_step1.json"), ("calib", "step2_checkA.json", "B1_calibration_step2_checkA.json"),
                   ("calib", "step2.json", "B1_calibration_step2.json"), ("pilot", "threshold.json", "B2_threshold_0.43pp.json"),
                   ("pilot", "stage0.json", "B2_noise_baseline_R0.json")]
                  + [("pilot", f"stage{i}.json", f"B3_training_R0_{n}.json") for i, n in ((1, "aug"), (2, "lr"), (3, "epochs"), (4, "wd"))]
                  + [("router", "stage5.json", "B4_router_with_warmup.json"), ("router", "compare_R0_s1.json", "B4_repro_check_R0_s1.json"),
                     ("router", "compare_R0_s2.json", "B4_repro_check_R0_s2.json"), ("stage6", "stage6a.json", "B6_fixed_calibration_with_warmup.json"),
                     ("stage6", "stage6a.json_bak", "B6_fixed_calibration_with_warmup_before_lambda_extension.json"),
                     ("stage6c", "stage6a_nw.json", "B6_fixed_calibration_old_rule_no_warmup.json"),
                     ("stage6", "stage6b.json", "B7_auxfree_collapse_check_70_epoch.json")],
}


def describe(run_dir):
    """One INDEX.md row: condition, training setting, code version, best A_pi."""
    c = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    s = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if c.get("model") == "single":
        cond = "single"
    else:
        cond = (f"{c.get('controller')}, {c.get('router', 'R0')}, warmup {'0' if not c.get('aux_warmup_frac') else 'E/3'}, "
                f"λ {c.get('lambda0')}, σ {c.get('sigma0')}, τ {c.get('tau0')}" + (f", γ {c.get('auxfree_gamma')}" if c.get("controller") == "auxfree" else ""))
    commit = ((c.get("env") or {}).get("git_commit") or "")[:7] or "not recorded"
    return {"cond": cond, "train": f"{c.get('aug')}, {fmt(float(c['lr']))}, {fmt(float(c['weight_decay']))}, {c.get('epochs')} ep, s{c.get('seed')}",
            "commit": commit, "acc": f"{s['best_val_acc_pi']:.4f} ({s['best_epoch']})"}


def main():
    ap = argparse.ArgumentParser(description="Export the runs used by analysis/Pilot_Study.md into a public record")
    ap.add_argument("--runs-root", default="../runs")
    ap.add_argument("--out", default="../release/pilot_study")
    ap.add_argument("--doc", default="../analysis/Pilot_Study.md")
    ap.add_argument("--overwrite", action="store_true", help="Delete and recreate --out if it exists")
    ap.add_argument("--with-checkpoints", action="store_true", help="Also copy *.pth (experiment machine only)")
    ap.add_argument("--strict", action="store_true", help="Fail if a required run is missing")
    args = ap.parse_args()
    root, out = Path(args.runs_root).resolve(), Path(args.out).resolve()
    if out == root or root in out.parents:
        sys.exit(f"[Error] --out must be outside {root} (runs/ is synced from the experiment machine)")
    if out.exists():
        if not args.overwrite:
            sys.exit(f"[Error] {out} exists; pass --overwrite to recreate it")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    placed = {}  # (source directory, run name) -> public path (relative to out)
    pending = []

    def put(group, stage_dir, src_dir, name):
        """Copy one run (on first appearance) and return its public path."""
        key = (src_dir, name)
        if key in placed:
            return placed[key], False
        src, rel = root / src_dir / name, Path(group) / stage_dir / name
        shutil.copytree(src, out / rel, ignore=None if args.with_checkpoints else shutil.ignore_patterns("*.pth"))
        for suf, dst in ((".log", "console.log"), (".eval.log", "eval.log")):
            lp = root / src_dir / "logs" / f"{name}{suf}"
            if lp.exists():
                shutil.copy2(lp, out / rel / dst)
        placed[key] = rel
        return rel, True

    lines = ["# Pilot study public record", "",
             "Generated by `p1/export_study.py` by copying from the experiment run directories (generated, do not edit by hand); it accompanies `Pilot_Study.md` in the same directory. "
             "Each run is stored once, in the directory of the stage where it first appears; later stages reference that location.", "",
             "Columns: Condition = controller, router, aux warmup (0 = none, E/3 = ×0.1 for the first 1/3 of training), λ0, σ0, τ0 (auxfree also lists γ); "
             "Training = augmentation, lr, wd, epochs, seed; Best A_pi is on the validation set, with the epoch in parentheses; Code version is the git commit on the experiment machine. "
             f"{'Includes' if args.with_checkpoints else 'Excludes'} .pth. Each run directory contains: config.json, history.csv, summary.json, val_eval_last/, "
             "val_routing.npz (MoE), console.log, eval.log (MoE).", "", "## Runs used in the main text", ""]
    for stage_dir, section, specs in MAIN:
        lines += [f"### {section}", "", "| Run | Condition | Training | Best A_pi | Code version |", "| --- | --- | --- | --- | --- |"]
        for src_dir, name, required in specs:
            src = root / src_dir / name
            if not (src / "summary.json").exists():
                if required:
                    pending.append(name)
                    lines.append(f"| `{name}` | pending | | | |")
                continue
            if stage_dir is None:
                rel = placed.get((src_dir, name))
                if rel is None:
                    rel, _ = put("runs", "stage6_fixed_calibration" if "stage6c" == src_dir else "misc", src_dir, name)
                cell = f"see `{rel.as_posix()}`"
            else:
                rel, new = put("runs", stage_dir, src_dir, name)
                cell = f"`{rel.as_posix()}`" if new else f"see `{rel.as_posix()}`"
            d = describe(src)
            lines.append(f"| {cell} | {d['cond']} | {d['train']} | {d['acc']} | {d['commit']} |")
        lines.append("")

    lines += ["## Appendix B: old-condition runs that were rerun or bypassed (not used for conclusions in the main text)", ""]
    for stage_dir, section, src_dir, names in SUPERSEDED:
        if names == "*":
            names = sorted(p.name for p in (root / src_dir).iterdir() if (p / "config.json").exists())
        lines += [f"### {section}", "", "| Run | Condition | Training | Best A_pi | Code version |", "| --- | --- | --- | --- | --- |"]
        for name in names:
            src = root / src_dir / name
            if not (src / "summary.json").exists():
                lines.append(f"| `{name}` | missing | | | |"); continue
            rel, _ = put("superseded", stage_dir, src_dir, name)
            d = describe(src)
            lines.append(f"| `{rel.as_posix()}` | {d['cond']} | {d['train']} | {d['acc']} | {d['commit']} |")
        lines.append("")

    lines += ["## Summary results", ""]
    for group, files in SUMMARIES.items():
        for src_dir, fn, dst_name in files:
            p = root / src_dir / fn
            if p.exists():
                dst = out / "summaries" / ("superseded" if group == "superseded" else "") / dst_name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dst)
                lines.append(f"- `{dst.relative_to(out).as_posix()}`")
    lines.append("")
    for extra, dst in ((root / "env_lock.json", out / "env" / "env_lock.json"), (Path(args.doc), out / "Pilot_Study.md"),
                       *((Path(__file__).resolve().parent / fn, out / "controller" / fn) for fn in ("fuzzy_v2.json", "fuzzy_v2_r2.json"))):
        if extra.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(extra, dst)
    if pending:
        lines += ["## Required runs not yet completed", ""] + [f"- `{p}`" for p in dict.fromkeys(pending)] + [""]
    (out / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")

    n_main = sum(1 for v in placed.values() if v.parts[0] == "runs"); n_sup = sum(1 for v in placed.values() if v.parts[0] == "superseded")
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 2 ** 20
    print(f"Exported to {out}: {n_main} main runs, {n_sup} superseded runs, {len(set(pending))} required runs pending, {size:.0f} MB")
    if pending and args.strict:
        sys.exit(f"[Error] {len(set(pending))} required runs are missing (see INDEX.md)")


if __name__ == "__main__":
    main()
