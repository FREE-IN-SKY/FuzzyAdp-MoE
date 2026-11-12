# RSCD Dataset: Reproduction Guide and Local Changes

## Reproduce from the original data

The required order is **download original data -> run `prepare_dataset.py` -> run `make_subset.py`**. No `subsets/` directory or existing `manifest.json` is required. Preparation uses `filename_mapping.csv` and the fixed source CSV hashes, row counts, and replacement counts in its `SOURCE_CSV` configuration.

### 1. Download and arrange the inputs

Download the **RSCD classification dataset** from the [official dataset website](https://thu-rsxd.com/rscd/) and extract it. Obtain the three per-day split CSVs from upstream commit [`505f613728107d78a890b16b18fad8002faf790c`](https://github.com/MiviaLab/Road-Surface-Dataset-per-day-split/tree/505f613728107d78a890b16b18fad8002faf790c). Start with this layout:

```text
datasets/
├── prepare_dataset.py
├── make_subset.py
├── filename_mapping.csv
├── README.md
├── RSCD dataset-1million/
│   ├── train/
│   ├── vali_20k/
│   └── test_50k/
└── Road-Surface-Dataset-per-day-split/
    ├── train.csv
    ├── vali_20k.csv
    └── test_50k.csv
```

The two scripts and mapping are project files. Images and source CSVs are inputs; `subsets/` is an output created later. Keep the mapping beside `prepare_dataset.py`.

If you have not downloaded the source CSVs, run the following from the project root. The subshell stops if the destination already exists or a download fails, so it does not overwrite an existing source directory:

```bash
(
  set -e
  mkdir datasets/Road-Surface-Dataset-per-day-split
  for name in train.csv vali_20k.csv test_50k.csv; do
    curl -fL "https://raw.githubusercontent.com/MiviaLab/Road-Surface-Dataset-per-day-split/505f613728107d78a890b16b18fad8002faf790c/$name" \
      -o "datasets/Road-Surface-Dataset-per-day-split/$name"
  done
)
```

If you already have the project's corrected source CSVs, skip downloading them. Preparation accepts both the pinned originals and the corrected versions, including a mixture of whole original/corrected CSV files.

### 2. Prepare images and source CSVs

Use Python 3.9 or newer. This step uses only the Python standard library and was validated with Python 3.11.15. Run from the project root:

```bash
# Optional preview: perform all checks without writing anything.
python datasets/prepare_dataset.py \
  --root "datasets/RSCD dataset-1million" \
  --split-dir datasets/Road-Surface-Dataset-per-day-split \
  --check-only

# Apply the image filename and CSV path corrections.
python datasets/prepare_dataset.py \
  --root "datasets/RSCD dataset-1million" \
  --split-dir datasets/Road-Surface-Dataset-per-day-split
```

The script corrects 59 image filenames and the corresponding `image` fields in the source CSVs: 54 in `train.csv`, 4 in `vali_20k.csv`, and 1 in `test_50k.csv`. Labels, classes, row order, image contents, and split assignments are unchanged. Existing corrected images/CSVs are accepted and skipped, so rerunning is safe.

Before changing files, the script checks all 59 mapped image hashes, recognizes each source CSV by its original or corrected SHA-256, verifies the transformed CSV bytes against fixed target hashes, checks backup conflicts for CSVs requiring conversion, and verifies all 1,028,301 source image paths. It does not read, create, or validate any experiment subsets.

Original CSVs are backed up beside themselves as `train.csv.bak`, `vali_20k.csv.bak`, and `test_50k.csv.bak`. An identical backup is accepted; a conflicting backup is never overwritten. Already corrected CSVs do not create new backups. `--check-only` reports pending changes without creating backups or temporary files.

For inputs elsewhere, pass their actual locations with `--root` and `--split-dir`, then use those same locations with `make_subset.py`. The image root must directly contain `train/`, `vali_20k/`, and `test_50k/`. Relative command-line paths are interpreted from the current working directory. If `--split-dir` is omitted in preparation, it defaults to `Road-Surface-Dataset-per-day-split/` beside the preparation script.

### 3. Generate the experiment subsets

Run this step **after preparation succeeds**. `make_subset.py` creates `subsets/` and `manifest.json`; it does not need previous subset outputs. It requires NumPy. The reference results below were validated with Python 3.11.15 and NumPy 2.4.6:

```bash
# Install in your chosen compatible Python environment if needed.
python -m pip install numpy==2.4.6

python datasets/make_subset.py \
  --split-dir datasets/Road-Surface-Dataset-per-day-split \
  --root "datasets/RSCD dataset-1million" \
  --out-dir datasets/subsets \
  --train-seeds 0,1,2,3,4,5 \
  --val-seed 2026
```

With the pinned inputs and these seeds, the output is:

| File | Records | Usage |
|---|---:|---|
| `train_s0.csv` | 41,130 | Pilot experiments, seed 0 |
| `train_s1.csv` | 41,130 | Pilot experiments, seed 1 |
| `train_s2.csv` | 41,130 | Pilot experiments, seed 2 |
| `train_s3.csv` | 41,130 | Formal run 1, seed 3 |
| `train_s4.csv` | 41,130 | Formal run 2, seed 4 |
| `train_s5.csv` | 41,130 | Formal run 3, seed 5 |
| `val.csv` | 5,810 | Fixed validation set, data seed 2026 |
| `test.csv` | 59,793 | Complete per-day test split |
| `manifest.json` | - | Class mapping, sampling rules, seeds, checks, and hashes |

Each CSV row contains `path,class_name,class_idx,date`. Resolve `path` relative to the image root, not the project root or the `subsets/` directory. Use `class_idx` with the generated manifest's `class_to_idx`, rather than the original source CSV's `label` values. The directory containing an image does not determine its experiment split: follow the CSV.

Training lists may overlap with one another; they represent separate runs. Recorded checks establish separation from validation/test by path and filename date, not image-content deduplication or independently verified acquisition dates. Dataset preparation alone does not guarantee identical model training results.

If output files already exist, `make_subset.py` refuses to overwrite them by default. To compare a regeneration, choose a fresh output directory such as `datasets/subsets_regenerated`; deletion of the previous directory is not required. Sampling sorts paths before drawing samples, so changing filenames or paths inside source CSVs can change the chosen images even with identical seeds. Moving the outer dataset folder does not change those relative paths.

### Verify the generated lists

The following optional check compares the generated CSVs against previously validated reference hashes. It does not require a previous `subsets/` directory. Run it after step 3; change `out` if you used another output directory:

```bash
python - <<'PY'
import hashlib
from pathlib import Path

out = Path("datasets/subsets")
expected = {
    "train_s0.csv": "f33e3fdabaeb30068df758697a35d8f3ad290173a237ba4cc02891c16df02d90",
    "train_s1.csv": "a369eedd6b083ac0585f6934ec94a0fce9ba64514f09c6017350688e6ae28e78",
    "train_s2.csv": "d4f084a14ddb4f183ef8518707eefc7a16d43f6f844217d8016ab91c7ce64a5a",
    "train_s3.csv": "3b4c059eeb6fd69362653d0ce92967972fac55a2ee71e2042113986fcba05818",
    "train_s4.csv": "17c8ca5023305b5f6a69a90e3fe3bf5a3c10de6c3404a998da056c0c4a28e83c",
    "train_s5.csv": "3daf1d39db3b77e6d881eb8dfb53d4feca9e5e4788136f0fb9b519ef76a8c62a",
    "val.csv": "b64e1760532927a135e7d117e8fa1631e6464133ba259a3b4c25718ed53c844b",
    "test.csv": "426caab2a8cb2d4824ec782d367b0c2ea83b7ac6a99b9b8ec103b1838472c008",
}
for name, digest in expected.items():
    actual = hashlib.sha256((out / name).read_bytes()).hexdigest()
    if actual != digest:
        raise SystemExit(f"Mismatch: {name}")
    print(f"OK: {name}")
PY
```

A new `manifest.json` has a new creation timestamp. Compare CSV hashes rather than expecting the entire JSON file to be byte-identical across runs.

### Interrupted conversions and validation scope

Preparation stages CSV replacements and original backups first, then renames images and replaces each CSV atomically using a temporary file in the same directory. Image renaming and backup installation use exclusive hard-link creation to prevent overwriting existing targets. Both filesystems must support hard links. Do not modify the image or CSV directories concurrently.

The whole batch is not a single transaction. Completed changes and backups are retained after interruption. Rerunning accepts mixed states across images and whole CSV files. Unknown CSV versions, including manually or partially edited individual CSVs, are rejected. If interruption leaves both the old and new name for one image, the next run reports a conflict; verify both against the mapping hash before manually resolving it. Abrupt termination can leave hidden CSV temporary files, which are not used as input on rerun. After restoring backups, run preparation again before generating or using subsets.

These checks validate all source paths, fixed CSV bytes, and the contents of the 59 mapped images. They do not decode or hash every image, or establish that the dataset has no duplicate image contents.

### Mapping provenance

[`filename_mapping.csv`](filename_mapping.csv) contains all 59 original relative paths, corrected relative paths, and image SHA-256 values. Original names were recovered from the three CSV files at upstream commit [`505f613728107d78a890b16b18fad8002faf790c`](https://github.com/MiviaLab/Road-Surface-Dataset-per-day-split/tree/505f613728107d78a890b16b18fad8002faf790c), not inferred from corrected names.

On 2026-10-19, row-by-row comparison against that commit confirmed exactly 54/4/1 changed `image` fields in train/validation/test. All other fields and row order matched. Image hashes were computed from the local corrected images. The original image archive was not re-downloaded for this verification; preparation stops if a downloaded mapped image differs from its recorded hash.

To share this workflow, provide the two scripts, `filename_mapping.csv`, and this README. Users download the images and pinned source CSVs, prepare them, then generate their own subsets. Existing corrected CSVs and reference subsets may be shared optionally; neither reference subsets nor their manifest are prerequisites.

## 1. Renaming Original RSCD Image Files

Corrected 59 malformed filenames by removing the extra `wet`, `wetasphalt`, or `dry` string between the numeric identifier and the class name. The corrected format is `numeric_identifier-class_name.jpg`.

Example:

```text
Before: 20220709134314820wet-dry-asphalt-smooth.jpg
After:  20220709134314820-dry-asphalt-smooth.jpg
```

| Original directory | Files renamed |
|---|---:|
| train | 57 |
| vali_20k | 1 |
| test_50k | 1 |
| Total | 59 |

The removed strings were `wet` in 56 filenames, `wetasphalt` in 2 filenames, and `dry` in 1 filename. Numeric identifiers, class names, file extensions, and parent directories were preserved.

Destination filenames were checked for conflicts before renaming. SHA-256 comparisons confirmed that the contents of all 59 files remained unchanged. Image counts were also unchanged.

## 2. Updating Paths in the Per-Day Split CSV Files

Updated the `image` field in the three CSV files under `Road-Surface-Dataset-per-day-split`, replacing the old paths for these 59 images with their renamed paths.

| Manifest | Paths updated | Total records |
|---|---:|---:|
| train.csv | 54 | 678,281 |
| vali_20k.csv | 4 | 290,227 |
| test_50k.csv | 1 | 59,793 |
| Total | 59 | 1,028,301 |

Only the `image` field was changed. The `label` and `class` fields, row order, record counts, and split assignments were preserved.

Images were not moved into new directories for the per-day split. Split membership is defined by the CSV manifests, whose paths still point to the original RSCD image directories. This explains why the update counts differ between the original directories and the per-day split manifests.

Post-update validation covered all records:

- Every image path in all three CSV files resolves to a local file.
- No duplicate image paths were found within any individual CSV file.
- Row-by-row comparisons confirmed that only the intended `image` fields changed.

## 3. Confirmed Duplicate Image with Conflicting Labels

The following two files have identical contents but are labeled `dry_concrete_slight` and `dry_concrete_severe`, respectively:

```text
train/dry_concrete_slight/202204252233248-dry-concrete-slight.jpg
train/dry_concrete_severe/202204252233248-dry-concrete-severe.jpg
```

Both files were verified to exist on 2026-10-19. Each is 8,293 bytes in size and has the following SHA-256 hash:

```text
20dae81618df0a7516963b43c8ecb07ca56d0f62f44ea5e5a4ff855025fafa2d
```

This is a conflict in the road unevenness label assigned to the same image. The filenames and hashes alone do not establish which class is correct.

Status: documented only, pending verification. Both image files and their existing labels are retained. No deletion, renaming, relabeling, or changes to the corresponding CSV records have been made to resolve this conflict.

Unique paths in the CSV files do not imply unique image contents. This record confirms only the pair above and does not establish that the entire dataset is free of other duplicate images.

## 4. Upstream Sources

- [Official RSCD dataset website](https://thu-rsxd.com/rscd/).
- [Road-Surface-Dataset-per-day-split](https://github.com/MiviaLab/Road-Surface-Dataset-per-day-split).
