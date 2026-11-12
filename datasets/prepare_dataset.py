#!/usr/bin/env python3
"""Convert RSCD image names and the per-day split CSVs by a fixed mapping. Does not depend on subset outputs; depends only on the standard library."""

import argparse
import csv
import hashlib
import io
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile


BASE_DIR = Path(__file__).resolve().parent
# MiviaLab/Road-Surface-Dataset-per-day-split, commit 505f613728107d78a890b16b18fad8002faf790c.
SOURCE_CSV = {
    "train.csv": {
        "upstream_sha256": "edb0687e277505958f9eafb6cac1d39536fa76c618c6dd953850dd7512dbd75d",
        "sha256": "14dfab9a1b661c3206203e04de3ab4e8730a5857cba40b21ba9eafe668826589",
        "rows": 678281, "changes": 54,
    },
    "vali_20k.csv": {
        "upstream_sha256": "27ecfcc3d586db7034d12da95e665ed2eefca03a842befc37803c6b5dfae4b91",
        "sha256": "5247612b61d4cb3d9413e97526bc63f24276c439c1dfbb66507ffb3c2881cb03",
        "rows": 290227, "changes": 4,
    },
    "test_50k.csv": {
        "upstream_sha256": "f2501a2d798d5430ff38fa34aba95e1ccecc38fb37dc2c13ff083b82510f8e4b",
        "sha256": "b43b36b4424facfef8ee4b2b80eac88430935ef8552e60ba1336bf4ffe24ff18",
        "rows": 59793, "changes": 1,
    },
}


def sha256_of(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def relative_path(value):
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"Unsafe relative path: {value}")
    return path


def load_mapping():
    with (BASE_DIR / "filename_mapping.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 59:
        raise ValueError("filename_mapping.csv must contain 59 records")
    old_paths, new_paths = set(), set()
    for row in rows:
        old, new = relative_path(row["old_path"]), relative_path(row["new_path"])
        if old == new or old.parent != new.parent:
            raise ValueError(f"Mappings must only rename files within the same directory: {old} -> {new}")
        if str(old) in old_paths or str(new) in new_paths:
            raise ValueError("Duplicate paths in the mapping")
        old_paths.add(str(old))
        new_paths.add(str(new))
        if len(row["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in row["sha256"]):
            raise ValueError(f"Invalid SHA-256: {old}")
    if old_paths & new_paths:
        raise ValueError("Source and destination paths overlap in the mapping")
    return rows


def plan_changes(root, mapping):
    """Check all mappings, accepting original or already-corrected names; does not modify files."""
    changes = []
    for row in mapping:
        old, new = root / row["old_path"], root / row["new_path"]
        for path in (old, new):
            if not path.resolve().is_relative_to(root):
                raise ValueError(f"Mapped path is outside the dataset directory: {path}")
        old_exists, new_exists = os.path.lexists(old), os.path.lexists(new)
        if old_exists and new_exists:
            raise ValueError(f"Both source and destination exist; refusing to overwrite: {old} -> {new}")
        if not old_exists and not new_exists:
            raise ValueError(f"Neither source nor destination exists: {old} / {new}")
        current = old if old_exists else new
        if current.is_symlink() or not current.is_file():
            raise ValueError(f"Mapped images must be regular files: {current}")
        if sha256_of(current) != row["sha256"]:
            raise ValueError(f"Image SHA-256 mismatch: {current}")
        if old_exists:
            changes.append((old, new))
    return changes


def convert_csv(data, mapping):
    """Replace only the image field, keep the field order and other values, use the CRLF line endings of the project CSVs."""
    replacements = {row["old_path"]: row["new_path"] for row in mapping}
    reader = csv.DictReader(io.StringIO(data.decode("utf-8"), newline=""))
    if reader.fieldnames != ["image", "label", "class"]:
        raise ValueError("Source CSV must contain exactly the image,label,class columns in that order")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=reader.fieldnames)
    writer.writeheader()
    count = 0
    for row in reader:
        if row["image"] in replacements:
            row["image"] = replacements[row["image"]]
            count += 1
        writer.writerow(row)
    return output.getvalue().encode("utf-8"), count


def plan_csv_files(split_dir, mapping):
    """Identify original/converted CSVs, validate the conversion result and backup conflicts in memory; does not write files."""
    plans = []
    for name, info in SOURCE_CSV.items():
        path = split_dir / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Source CSV must be a regular file: {path}")
        original = path.read_bytes()
        digest = hashlib.sha256(original).hexdigest()
        data, count = original, 0
        if digest != info["sha256"]:
            if digest != info["upstream_sha256"]:
                raise ValueError(f"Unknown source CSV version; SHA-256 mismatch: {path}")
            data, count = convert_csv(original, mapping)
            if count != info["changes"] or hashlib.sha256(data).hexdigest() != info["sha256"]:
                raise ValueError(f"Converted CSV does not match the pinned manifest: {path}")
            backup = path.with_name(path.name + ".bak")
            if os.path.lexists(backup):
                if backup.is_symlink() or not backup.is_file() or sha256_of(backup) != digest:
                    raise ValueError(f"CSV backup conflict; refusing to overwrite: {backup}")
        plans.append({"path": path, "data": data, "original": original,
                      "original_hash": digest, "target_hash": info["sha256"],
                      "count": count, "rows": info["rows"]})
    return plans


def verify_source_paths(root, changes, csv_plans):
    """Validate all converted source paths and record counts, allowing the original names of images to be converted in this run."""
    pending = {str(new.relative_to(root)): old for old, new in changes}
    total, missing, examples = 0, 0, []
    for plan in csv_plans:
        count = 0
        with io.StringIO(plan["data"].decode("utf-8"), newline="") as f:
            for row in csv.DictReader(f):
                rel = str(relative_path(row["image"]))
                current = pending.get(rel, root / rel)
                if not current.is_file():
                    missing += 1
                    if len(examples) < 5:
                        examples.append(rel)
                count += 1
        if count != plan["rows"]:
            raise ValueError(f"Source manifest row count mismatch: {plan['path']}")
        total += count
    if missing:
        raise ValueError(f"Missing images in the manifests: {missing}; examples: {examples}")
    return total


def apply_changes(changes):
    """Create a hard link in the same directory, then delete the old name; fail if the target already exists, no overwrite."""
    for old, new in changes:
        # os.rename can overwrite the target on Unix; the exclusive creation of link avoids this data loss.
        os.link(old, new)
        old.unlink()


def stage_file(path, data):
    """Write and sync a temp file in the target's directory, preserving the original file's permissions."""
    staged = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as f:
            staged = Path(f.name)
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        staged.chmod(path.stat().st_mode & 0o777)
        return staged
    except BaseException:
        if staged is not None:
            staged.unlink(missing_ok=True)
        raise


def apply_preparation(image_changes, csv_plans):
    """First prepare all CSV temp files and original backups, then rename the images and atomically replace the CSVs one by one."""
    staged_files = []
    try:
        for plan in csv_plans:
            if not plan["count"]:
                continue
            path = plan["path"]
            if sha256_of(path) != plan["original_hash"]:
                raise ValueError(f"CSV changed after preflight: {path}")
            staged = stage_file(path, plan["data"])
            staged_files.append((plan, staged))
            backup = path.with_name(path.name + ".bak")
            backup_staged = stage_file(path, plan["original"])
            try:
                try:
                    os.link(backup_staged, backup)
                except FileExistsError:
                    if backup.is_symlink() or not backup.is_file() or sha256_of(backup) != plan["original_hash"]:
                        raise ValueError(f"CSV backup conflict; refusing to overwrite: {backup}")
            finally:
                backup_staged.unlink()
        apply_changes(image_changes)
        for plan, staged in staged_files:
            if sha256_of(plan["path"]) != plan["original_hash"]:
                raise ValueError(f"CSV changed after preflight: {plan['path']}")
            os.replace(staged, plan["path"])
            if sha256_of(plan["path"]) != plan["target_hash"]:
                raise ValueError(f"Written CSV SHA-256 mismatch: {plan['path']}")
    finally:
        for _, staged in staged_files:
            staged.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Correct RSCD image filenames and per-day split CSV paths using a fixed mapping.")
    parser.add_argument("--root", required=True, help="Image root directory containing train/, vali_20k/, and test_50k/")
    parser.add_argument("--split-dir", default=str(BASE_DIR / "Road-Surface-Dataset-per-day-split"),
                        help="Directory of original or corrected per-day split CSVs; defaults to the project directory")
    parser.add_argument("--check-only", action="store_true", help="Validate and report pending image and CSV conversions without writing files")
    args = parser.parse_args()
    try:
        root = Path(args.root).expanduser().resolve()
        for name in ("train", "vali_20k", "test_50k"):
            if not (root / name).is_dir():
                raise ValueError(f"Missing image directory: {root / name}")
        mapping = load_mapping()
        changes = plan_changes(root, mapping)
        split_dir = Path(args.split_dir).expanduser().resolve()
        csv_plans = plan_csv_files(split_dir, mapping)
        print(f"All 59 mappings validated: {len(changes)} pending renames, {59 - len(changes)} already corrected", flush=True)
        for plan in csv_plans:
            print(f"{plan['path'].name}: {plan['count']} paths pending conversion", flush=True)
        print("Validating CSV hashes and all source image paths...", flush=True)
        total = verify_source_paths(root, changes, csv_plans)
        print(f"Source manifests validated: {total:,} records")
        csv_count = sum(plan["count"] for plan in csv_plans)
        if args.check_only:
            print(f"Check only; no files changed. Pending renames: {len(changes)}, CSV paths pending conversion: {csv_count}")
            return
        apply_preparation(changes, csv_plans)
        print(f"Preparation complete: renamed {len(changes)} images and converted {csv_count} CSV paths.")
        print("Next: run make_subset.py with the same --root and --split-dir and specify --out-dir to generate subsets.")
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
