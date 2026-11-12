#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
estimate_e0.py

Estimate the provisional training budget E0 from the logs of old experiments. Only reads files, does not modify anything.

Rules (fixed before the pilot study starts):
    E0 = round up to a multiple of 5 ( 1.2 * max(epoch of the best validation accuracy of each old experiment) )
    Candidate budgets: 0.7 * E0 and 1.3 * E0, rounded as floor(x + 0.5), using integer arithmetic (7*E0+5)//10 and (13*E0+5)//10 to avoid floating-point error

Each old experiment must provide a CSV log (save_csv_log/.../*_train_*.csv, with epoch and val_acc columns),
and optionally a txt log (*_train_time_log.txt), used to read where the learning rate drops and to check whether training was resumed from a checkpoint.

Usage example:
    python estimate_e0.py \
        --run single  "old_experiments/single/save_csv_log/efficientnet_b0/efficientnet_b0_train_300e_224x224.csv" \
                      "old_experiments/single/save_csv_log/efficientnet_b0/efficientnet_b0_300e_224x224_train_time_log.txt" \
        --run fixed   "old_experiments/MoE/.../xxx.csv" "old_experiments/MoE/.../xxx_log.txt" \
        --run fuzzy   "old_experiments/fuzzy/.../xxx.csv" "old_experiments/fuzzy/.../xxx_log.txt"
"""

import argparse
import csv
import math
import re
import sys

LR_PATTERN = re.compile(r"(?:当前学习率|Current learning rates)\s*:?\s*\[([^\]]+)\]")
RESUME_PATTERN = re.compile(r"已加载模型权重|Loaded model weights|Loaded weights from|历史最佳验证精度|Previous best validation")
EPOCH_PATTERN = re.compile(r"Epoch\s+(\d+)\s*/\s*\d+")


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or "val_acc" not in rows[0] or "epoch" not in rows[0]:
        sys.exit(f"[Error] {path} is empty or missing the epoch or val_acc column")
    epochs = [int(float(r["epoch"])) for r in rows]
    accs = [float(r["val_acc"]) for r in rows]
    return epochs, accs


def read_log(path):
    """Return (list of the first epochs after each learning-rate drop, whether a resume-from-checkpoint record appears)."""
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    resumed = bool(RESUME_PATTERN.search(text))
    drops, prev, epoch = [], None, None
    for line in text.splitlines():
        m = EPOCH_PATTERN.search(line)
        if m:
            epoch = int(m.group(1))
            continue
        m = LR_PATTERN.search(line)
        if m:
            lr = float(m.group(1).split(",")[0])
            # the learning rate in the log is the value after the epoch ends and the scheduler updates; the new learning rate takes effect from the next epoch
            if prev is not None and lr < prev * 0.999 and epoch is not None:
                drops.append(epoch + 1)
            prev = lr
    return drops, resumed


def main():
    ap = argparse.ArgumentParser(description="Estimate the provisional epoch budget E0 from previous experiment logs")
    ap.add_argument("--run", nargs="+", action="append", required=True, metavar=("NAME CSV", "LOG"),
                    help="Name, CSV log path, and optional text log path")
    ap.add_argument("--margin", type=float, default=1.2)
    args = ap.parse_args()

    best_epochs = []
    print(f"{'Run':10s} {'Epochs':>10s} {'Best epoch':>10s} {'Best val_acc':>12s}  First epoch after each learning rate drop / Notes")
    for item in args.run:
        if len(item) not in (2, 3):
            sys.exit("[Error] Each --run requires 2 or 3 arguments: name, CSV path, [text log path]")
        name, csv_path = item[0], item[1]
        epochs, accs = read_csv(csv_path)
        notes = []
        if epochs != list(range(1, len(epochs) + 1)):
            notes.append("Epoch numbers are not consecutive from 1; this may be a resumed run or multiple runs in one file")
        best_i = max(range(len(accs)), key=lambda i: accs[i])  # on ties take the earliest
        best_epochs.append(epochs[best_i])
        drops = "-"
        if len(item) == 3:
            d, resumed = read_log(item[2])
            drops = str(d) if d else "None"
            if resumed:
                notes.append("The log records loading previous weights; this may be a resumed run")
        print(f"{name:10s} {len(epochs):>10d} {epochs[best_i]:>10d} {accs[best_i]:>12.4f}  {drops}"
              + (f"  [Warning] {'; '.join(notes)}" if notes else ""))

    e0 = int(math.ceil(args.margin * max(best_epochs) / 5.0) * 5)
    print()
    print(f"Maximum best epoch: {max(best_epochs)}")
    print(f"E0 = ceil({args.margin} x {max(best_epochs)} / 5) x 5 = {e0}")
    print(f"Candidate budgets: 0.7 x E0 = {(7 * e0 + 5) // 10}, E0 = {e0}, 1.3 x E0 = {(13 * e0 + 5) // 10}")
    print(f"With aux-warmup-frac = 1/3, auxiliary loss warmup lasts {math.floor(e0 / 3 + 0.5)} epochs (20 in the original code)")


if __name__ == "__main__":
    main()
