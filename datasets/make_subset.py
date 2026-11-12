#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_subset.py

Generate the file lists for the experiments from the RSCD per-day split lists (MiviaLab/Road-Surface-Dataset-per-day-split).
Only generates CSV file lists and a manifest; does not copy, move, or modify any image.

Outputs:
    train_s{k}.csv   one training subset per data seed k, round(full-set count * 0.04) per class
    val.csv          fixed validation subset, max(round(full-set count * 0.005), 100) per class, using a separate validation seed
    test.csv         all images of the per-day split test list
    manifest.json    class mapping, quotas, target distribution pi, seeds, checksums, date coverage, known issues, self-check results

Usage example (run from the project root):
    python datasets/make_subset.py \
        --split-dir datasets/Road-Surface-Dataset-per-day-split \
        --root "datasets/RSCD dataset-1million" \
        --out-dir datasets/subsets
"""

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SOURCE_FILES = {"train": "train.csv", "val": "vali_20k.csv", "test": "test_50k.csv"}
TRAIN_RATIO = 0.04
VAL_RATIO = 0.005
VAL_MIN_PER_CLASS = 100
# Separate random number streams for the training and validation subsets, so the two do not use the same seed sequence
STREAM_TRAIN = 1
STREAM_VAL = 2


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


IRREGULAR_NAMES = []  # file names whose number is not immediately followed by '-', e.g. "...468wet-dry-asphalt-smooth.jpg"


def parse_name(rel_path: str):
    """Take the number and the date (first 8 digits) from the leading consecutive digits of the file name. Exits with an error on parse failure."""
    fname = rel_path.split("/")[-1]
    m = re.match(r"^(\d+)", fname)
    if not m or len(m.group(1)) < 8:
        sys.exit(f"[Error] Cannot parse the identifier from the filename: {rel_path}")
    num = m.group(1)
    if not fname[len(num):].startswith("-"):
        IRREGULAR_NAMES.append(rel_path)
    date = num[:8]
    try:
        datetime.strptime(date, "%Y%m%d")
    except ValueError:
        sys.exit(f"[Error] The first 8 filename characters are not a valid date: {rel_path}")
    return num, date


def label_from_filename(rel_path: str, num: str):
    """Take the text after the first '-' following the number as the file-name label, strip the extension, replace '-' with '_'."""
    fname = rel_path.split("/")[-1]
    rest = fname[len(num):]
    if "-" not in rest:
        return None
    label = rest.split("-", 1)[1].rsplit(".", 1)[0]
    return label.replace("-", "_")


def validate_sources(src):
    """Pre-sampling check: paths are unique within each source list; each row's class matches the file-name label; under train/<class>/ it matches the directory."""
    errors = []
    for split, rows in src.items():
        dup = [p for p, n in Counter(r[0] for r in rows).items() if n > 1]
        if dup:
            errors.append(f"Source manifest {split} contains {len(dup)} duplicate paths, e.g. {dup[:5]}")
        name_mis, dir_mis = [], []
        for rel, cls, _, num in rows:
            lab = label_from_filename(rel, num)
            if lab != cls:
                name_mis.append((rel, cls, lab))
            parts = rel.split("/")
            if len(parts) == 3 and parts[0] == "train" and parts[1] != cls:
                dir_mis.append((rel, cls))
        if name_mis:
            errors.append(f"Source manifest {split} has {len(name_mis)} rows with class/filename label mismatches, "
                          f"e.g. (path, class, filename label) {name_mis[:5]}")
        if dir_mis:
            errors.append(f"Source manifest {split} has {len(dir_mis)} rows with class/train/<class>/ directory mismatches, "
                          f"e.g. (path, class) {dir_mis[:5]}")
    return errors


def parse_seed_list(text: str):
    items = [s.strip() for s in text.split(",")]
    if not text.strip() or any(s == "" for s in items):
        sys.exit("[Error] --train-seeds requires at least one comma-separated seed with no empty entries, e.g. 0,1,2,3")
    seeds = []
    for s in items:
        if not s.isdigit():
            sys.exit(f"[Error] Seed '{s}' in --train-seeds is not a nonnegative integer")
        seeds.append(int(s))
    if len(set(seeds)) != len(seeds):
        sys.exit("[Error] Duplicate seeds in --train-seeds")
    return seeds


def load_source(split_dir: Path):
    """Read the three source lists, return {subset: [(path, class name, date, number), ...]}."""
    data = {}
    for split, fname in SOURCE_FILES.items():
        path = split_dir / fname
        if not path.exists():
            sys.exit(f"[Error] Source manifest not found: {path}")
        rows = []
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                rel = r["image"].strip()
                cls = r["class"].strip().replace("-", "_")
                num, date = parse_name(rel)
                rows.append((rel, cls, date, num))
        # Sort, so the result does not depend on the row order in the source files
        rows.sort(key=lambda x: x[0])
        data[split] = rows
    return data


def rng_for(seed: int, stream: int, class_idx: int) -> np.random.Generator:
    """Each class uses its own random number generator, so the result does not depend on the class iteration order."""
    return np.random.default_rng(np.random.SeedSequence([seed, stream, class_idx]))


def sample_per_class(rows, quotas, class_to_idx, seed, stream):
    by_class = defaultdict(list)
    for row in rows:
        by_class[row[1]].append(row)
    picked = []
    for cls, quota in quotas.items():
        pool = by_class[cls]  # already sorted by path
        if quota > len(pool):
            sys.exit(f"[Error] Class {cls} requires {quota} images, but only {len(pool)} are available")
        idx = rng_for(seed, stream, class_to_idx[cls]).choice(len(pool), size=quota, replace=False)
        picked.extend(pool[i] for i in sorted(idx))
    picked.sort(key=lambda x: x[0])
    return picked


def write_list(path: Path, rows, class_to_idx):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "class_name", "class_idx", "date"])
        for rel, cls, date, _ in rows:
            w.writerow([rel, cls, class_to_idx[cls], date])


def class_counts(rows, classes):
    c = Counter(r[1] for r in rows)
    return {cls: c[cls] for cls in classes}


def date_coverage(rows, classes):
    per = defaultdict(Counter)
    for _, cls, date, _ in rows:
        per[cls][date] += 1
    return {cls: {"n_dates": len(per[cls]), "per_date": dict(sorted(per[cls].items()))} for cls in classes}


def main():
    ap = argparse.ArgumentParser(description="Generate training/validation/test lists from the RSCD per-day split")
    ap.add_argument("--split-dir", required=True, help="Per-day split repository directory containing train.csv, vali_20k.csv, and test_50k.csv")
    ap.add_argument("--root", default=None, help="Local dataset root containing train/, vali_20k/, and test_50k/; if provided, check class directories and image existence")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--train-seeds", default="0,1,2,3,4,5", help="Comma-separated training subset data seeds; 0-2 for pilot runs, 3-5 for formal runs")
    ap.add_argument("--val-seed", type=int, default=2026, help="Validation sampling seed, independent of the training seed")
    ap.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output files")
    args = ap.parse_args()

    split_dir = Path(args.split_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    root = Path(args.root).expanduser() if args.root else None
    train_seeds = parse_seed_list(args.train_seeds)
    if args.val_seed < 0:
        sys.exit("[Error] --val-seed must be a nonnegative integer")

    out_files = [f"train_s{k}.csv" for k in train_seeds] + ["val.csv", "test.csv", "manifest.json"]
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = [f for f in out_files if (out_dir / f).exists()]
    if existing and not args.overwrite:
        sys.exit(f"[Error] Output files already exist: {existing}. Pass --overwrite to replace them")

    # ---------- Read source lists ----------
    src = load_source(split_dir)
    src_errors = validate_sources(src)
    if src_errors:
        print("[Validation failed] Source manifest validation failed; no files were written:")
        for e in src_errors:
            print("  -", e)
        sys.exit(1)
    all_rows = [r for s in src for r in src[s]]
    classes = sorted({r[1] for r in all_rows})
    if len(classes) != 27:
        sys.exit(f"[Error] Expected 27 classes, found {len(classes)}: {classes}")
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # ---------- Check against local folders ----------
    checks = {}
    if root is not None:
        local = sorted(p.name for p in (root / "train").iterdir() if p.is_dir())
        if local != classes:
            sys.exit(f"[Error] Manifest classes do not match local train directories.\n  Manifest: {classes}\n  Local: {local}")
        checks["class_names_match_local_train_folders"] = True
    else:
        print("[Info] No --root provided; skipping local class directory and image existence checks")
        checks["class_names_match_local_train_folders"] = "skipped (no --root)"

    # ---------- Quotas ----------
    total = Counter(r[1] for r in all_rows)
    # Note: Python's round is banker's rounding (round half to even); the rule is written into the manifest
    train_quota = {c: round(total[c] * TRAIN_RATIO) for c in classes}
    val_quota = {c: max(round(total[c] * VAL_RATIO), VAL_MIN_PER_CLASS) for c in classes}
    n_train = sum(train_quota.values())
    pi = {c: train_quota[c] / n_train for c in classes}

    # ---------- Sampling ----------
    train_sets = {k: sample_per_class(src["train"], train_quota, class_to_idx, k, STREAM_TRAIN) for k in train_seeds}
    val_set = sample_per_class(src["val"], val_quota, class_to_idx, args.val_seed, STREAM_VAL)
    test_set = list(src["test"])

    # ---------- Self-check ----------
    def paths(rows):
        return {r[0] for r in rows}

    def dates(rows):
        return {r[2] for r in rows}

    errors = []
    # Between source lists
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        if paths(src[a]) & paths(src[b]):
            errors.append(f"Source manifests {a} and {b} share paths")
        if dates(src[a]) & dates(src[b]):
            errors.append(f"Source manifests {a} and {b} share dates")
    # No duplicates within each subset, count equals the quota
    for k, rows in train_sets.items():
        if len(paths(rows)) != len(rows):
            errors.append(f"Duplicate paths in train_s{k}")
        if class_counts(rows, classes) != train_quota:
            errors.append(f"Per-class counts in train_s{k} do not match quotas")
        # Each training subset is compared with the validation set and the test set separately (overlap between different training subsets is allowed)
        for name, other in [("val", val_set), ("test", test_set)]:
            if paths(rows) & paths(other):
                errors.append(f"train_s{k} and {name} share paths")
            if dates(rows) & dates(other):
                errors.append(f"train_s{k} and {name} share dates")
    if len(paths(val_set)) != len(val_set):
        errors.append("Duplicate paths in val")
    if class_counts(val_set, classes) != val_quota:
        errors.append("Per-class counts in val do not match quotas")
    if paths(val_set) & paths(test_set):
        errors.append("val and test share paths")
    if dates(val_set) & dates(test_set):
        errors.append("val and test share dates")

    # Local image existence (only checks the images used by this output)
    if root is not None:
        used = set().union(*(paths(r) for r in train_sets.values()), paths(val_set), paths(test_set))
        missing = sorted(p for p in used if not (root / p).is_file())
        if missing:
            errors.append(f"Missing {len(missing)} local images, e.g. {missing[:5]}")
        checks["all_listed_files_exist_locally"] = not missing
    else:
        checks["all_listed_files_exist_locally"] = "skipped (no --root)"

    if errors:
        print("[Validation failed] No files were written:")
        for e in errors:
            print("  -", e)
        sys.exit(1)

    checks.update({
        "each_source_list_has_unique_paths": True,
        "each_row_class_matches_filename_label": True,
        "each_row_class_matches_train_folder_when_applicable": True,
        "source_splits_disjoint_by_path_and_date": True,
        "each_train_subset_disjoint_from_val_and_test_by_path_and_date": True,
        "val_disjoint_from_test_by_path_and_date": True,
        "per_class_counts_equal_quotas": True,
        "overlap_between_train_subsets": "allowed, not checked",
        "scope_note": "These checks establish separation by path and filename date only; they do not deduplicate image contents. "
                      "Timestamps in filenames may indicate cropping time rather than acquisition time.",
    })

    # ---------- Known issue: one numeric ID maps to multiple classes ----------
    id_classes = defaultdict(set)
    id_rows = defaultdict(list)
    for split in src:
        for rel, cls, _, num in src[split]:
            id_classes[num].add(cls)
            id_rows[num].append((split, rel, cls))
    val_paths = paths(val_set)
    train_union = set().union(*(paths(r) for r in train_sets.values()))
    test_paths = paths(test_set)
    conflicts = []
    for num, cl in sorted(id_classes.items()):
        if len(cl) > 1:
            conflicts.append({
                "id": num,
                "records": [
                    {"source": s, "path": rel, "class": cls,
                     "in_val_subset": rel in val_paths,
                     "in_any_train_subset": rel in train_union,
                     "in_test": rel in test_paths}
                    for s, rel, cls in id_rows[num]
                ],
            })

    # ---------- Write files ----------
    for k, rows in train_sets.items():
        write_list(out_dir / f"train_s{k}.csv", rows, class_to_idx)
    write_list(out_dir / "val.csv", val_set, class_to_idx)
    write_list(out_dir / "test.csv", test_set, class_to_idx)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "make_subset.py",
        "source": {
            "repository": "https://github.com/MiviaLab/Road-Surface-Dataset-per-day-split",
            "files": {split: {"file": fname, "rows": len(src[split]),
                              "sha256": sha256_of(split_dir / fname)}
                      for split, fname in SOURCE_FILES.items()},
            "class_name_normalization": "Replace '-' with '_' in the manifest class column; do not use the original label column",
        },
        "class_to_idx": class_to_idx,
        "rules": {
            "train_quota": f"round(full_dataset_class_count * {TRAIN_RATIO})",
            "val_quota": f"max(round(full_dataset_class_count * {VAL_RATIO}), {VAL_MIN_PER_CLASS})",
            "rounding": "Python built-in round (round half to even)",
            "class_totals_definition": "Sum of rows for the class across train.csv, vali_20k.csv, and test_50k.csv",
            "sampling": "Sort paths within each class, then sample without replacement using numpy.random.default_rng(SeedSequence([seed, stream, class_idx]))",
            "streams": {"train": STREAM_TRAIN, "val": STREAM_VAL},
            "test": "Use all images in the test manifest",
        },
        "seeds": {
            "train_data_seeds": train_seeds,
            "seed_usage": "Seeds 0-2 are for pilot experiments; seeds 3-5 are for formal runs. Scheme A: run k uses train_s{k}.csv and training seed k; all methods share the same (list, seed) pairs",
            "val_seed": args.val_seed,
        },
        "class_totals": {c: total[c] for c in classes},
        "train_quota": train_quota,
        "val_quota": val_quota,
        "sizes": {
            **{f"train_s{k}": len(r) for k, r in train_sets.items()},
            "val": len(val_set),
            "test": len(test_set),
        },
        "target_distribution_pi": {
            "definition": f"pi_c = train_quota_c / {n_train}, shared by all runs",
            "values": pi,
            "usage": "Controller inputs (weighted accuracy A_pi; aggregate weighted expert usage before computing Gini and minimum usage) and checkpoint selection",
        },
        "date_coverage": {
            "val": date_coverage(val_set, classes),
            "test": date_coverage(test_set, classes),
            **{f"train_s{k}": {c: v["n_dates"] for c, v in date_coverage(r, classes).items()}
               for k, r in train_sets.items()},
        },
        "known_issues": {
            "same_id_multiple_classes": conflicts,
            "irregular_filenames": {
                "description": "The numeric identifier is not immediately followed by '-'; extra text (e.g. 'wet', 'dry') may conflict with the label. Use the manifest class column as the class label",
                "count": len(IRREGULAR_NAMES),
                "records": [
                    {"path": p,
                     "in_val_subset": p in val_paths,
                     "in_any_train_subset": p in train_union,
                     "in_test": p in test_paths}
                    for p in sorted(IRREGULAR_NAMES)
                ],
            },
            "note": "Original records are retained as requested; no records are deleted or relabeled",
        },
        "self_checks": checks,
        "output_sha256": {f: sha256_of(out_dir / f) for f in out_files if f != "manifest.json"},
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # ---------- Summary ----------
    print(f"Number of classes: {len(classes)}")
    for k, r in train_sets.items():
        print(f"train_s{k}.csv: {len(r)} images")
    print(f"val.csv: {len(val_set)} images (validation seed {args.val_seed})")
    print(f"test.csv: {len(test_set)} images")
    print(f"Number of identifiers assigned to multiple classes: {len(conflicts)}")
    for c in conflicts:
        for rec in c["records"]:
            print(f"  {rec['path']}  In validation subset: {rec['in_val_subset']}")
    irr = [p for p in IRREGULAR_NAMES]
    print(f"Irregular filenames: {len(irr)} records; {sum(p in val_paths for p in irr)} in validation, "
          f"{sum(p in train_union for p in irr)} in any training subset, {sum(p in test_paths for p in irr)} in test")
    print(f"Validation passed, Output directory: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
