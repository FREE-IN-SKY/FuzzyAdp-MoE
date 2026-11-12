# FuzzyAdp-MoE

Code and experiment records for the paper **"Edge-AI Road Surface Classification with Fuzzy-Adaptive Mixture-of-Experts"** (IEEE AIoT 2026).

FuzzyAdp-MoE is a sparse mixture-of-experts (MoE) on a partitioned EfficientNet-B0 backbone with a cosine router. A Mamdani fuzzy controller adapts the routing temperature τ, the routing noise σ, and the auxiliary-loss weight λ once per epoch from validation balance and accuracy signals. Inference uses the same graph as a fixed-coefficient MoE. The repository compares it with a monolithic EfficientNet-B0, Fixed MoE, a linear schedule, and auxiliary-loss-free balancing on the per-day split of the Road Surface Classification Dataset (RSCD), and measures CPU-only inference on four edge boards.

## Repository layout

```text
FuzzyAdp-MoE/
├── p1/                     training, evaluation, analysis, and summary code
│   ├── common.py           data loading, augmentation, metrics, LR schedule, environment check
│   ├── train_single.py     monolithic EfficientNet-B0
│   ├── train_moe.py        MoE with --controller fixed | schedule | fuzzy | auxfree
│   ├── evaluate.py         validation / test evaluation, dense expert evaluation (--dense-experts)
│   ├── analyze_experts.py  expert function analysis (masking, CKA, agreement) -> experts.json
│   ├── summarize_*.py      pilot-stage and final summaries; summarize_final.py writes analysis/final/
│   ├── export_study.py     builds pilot_study/ from the run directories
│   ├── env_lock.py         shows or updates runs/env_lock.json
│   ├── fuzzy_v2.json       controller, design round 1
│   ├── fuzzy_v2_r2.json    controller, design round 2
│   ├── fuzzy_v2_r3.json    controller, design round 3 (used in the main experiments)
│   └── shell/              run_calib.sh, run_pilot.sh, run_extra.sh
├── datasets/               data preparation and the exact subset lists (see datasets/README.md)
│   └── subsets/            train_s0..s5.csv, val.csv, test.csv, manifest.json
├── runs/                   records of the main experiments and of the pilot runs not in pilot_study/
│   ├── env_lock.json       environment of the experiment machine
│   ├── final/              33 training runs (batches A and B, seeds 3-5), test evaluations, dense expert analysis
│   ├── final_timing/       15 timing runs (batch C, diagnostics off, display on)
│   ├── final_timing_c1_screenoff/   first timing batch of seed 3 (display off); kept, not used for reported times
│   ├── stage9/             pilot: poor starting points of fixed and fuzzy, and the σ = 0 runs (Supplementary S2.7, S4.3)
│   ├── oracle/             pilot: winner-take-all loss runs (Supplementary S4.4)
│   └── stage6/, stage6c/, stage8/   dense expert evaluations of pilot runs (Supplementary S4.3)
├── pilot_study/            public record of the pilot study (Supplementary S1, S2)
│   ├── Pilot_Study.md      pilot protocol, decisions, and results
│   ├── INDEX.md            every run used, with its stage, condition, and result
│   ├── runs/, superseded/  run records by stage
│   ├── summaries/          stage summaries written by the summary scripts
│   ├── controller/         controller configurations of design rounds 1 and 2
│   └── env/                environment lock
├── analysis/final/         final_tables.md and final_summary.json (numbers of the paper tables)
└── deploy/                 edge-device measurements (see deploy/README.md)
    ├── references/         CPU fp32 reference predictions used for the on-board consistency check
    └── results/            raw latency and power results per board, summary.md
```

### What each run directory contains

| File | Content |
|---|---|
| `config.json` | all arguments, data sizes, derived settings, the running environment (`env`), and the git commit of the code |
| `history.csv` | per-epoch losses, accuracies, timing; for MoE also routing statistics, controller outputs, and `diag_*` training diagnostics |
| `summary.json` | best epoch, best validation `acc_pi`, total time |
| `val_eval_last/metrics.json`, `test_eval_{best,last}/metrics.json` | overall, per-class, and per-date metrics and the confusion matrix |
| `val_dense_best/metrics.json` | dense evaluation of every expert (MoE runs used in the expert analysis) |
| `logs/` | console logs of training and evaluation |

Raw records are published unchanged. They contain absolute paths and host names of the machines used, and a few Chinese text fields: the `_comment` of the controller configuration copied into `config.json`, and the `why` and `excluded` fields of some early pilot summaries.

### Not included

- Model checkpoints (`best.pth`, `last.pth`) in the git tree. The `best.pth` of the 15 runs of batch A (monolithic, Fixed, Schedule, Aux-free, FuzzyAdp-MoE; seeds 3, 4, 5) are attached to the GitHub Release of this repository as `FuzzyAdp-MoE_best_seed3.tar.gz`, `FuzzyAdp-MoE_best_seed4.tar.gz`, and `FuzzyAdp-MoE_best_seed5.tar.gz`, with `SHA256SUMS.txt`. The seed-3 models are the models of the edge measurements. Extract the archives at the repository root (`tar xzf FuzzyAdp-MoE_best_seed3.tar.gz`); the files land in `runs/final/<run>/best.pth`, where `p1/evaluate.py` and the `deploy/` scripts expect them. The checkpoints of batch B (poor starting points) are not released.
- Per-image prediction files (`predictions.csv`) of the training runs, the routing traces (`val_routing.npz`), and the dense expert outputs (`dense_experts.npz`). `pilot_study/INDEX.md` lists `val_routing.npz` among the run files; it is not part of this release. The summary tables built from these files are included.
- The RSCD images and the three upstream per-day split CSVs. `datasets/README.md` explains how to download and prepare them.

## Code version

This repository was exported from our development repository, which is not public, at its commit `e11e220`. The commit hashes in `config.json`, `pilot_study/INDEX.md`, and the supplementary material refer to that repository. The main experiments were produced from commit `411e887`, and the pilot reruns of Supplementary S4.4 and the σ = 0 runs of seeds 1 and 2 from commit `96521d2`. Between `96521d2` and this release, the Python code of `p1/`, `datasets/`, and `deploy/` differs only in comments and docstrings (checked by comparing the abstract syntax trees), except `p1/summarize_final.py`, `p1/export_study.py`, and `deploy/summarize_device.py`, whose report text was changed from Chinese to English. For this release, the `_comment` field of the three controller configurations was translated to English; the controller reads no other field than before. The two Chinese strings left in `p1/estimate_e0.py` are regular expressions that match old Chinese log lines.

## Environment

All training and evaluation ran on one machine with an NVIDIA GeForce RTX 4070 Laptop GPU. The versions are recorded in `runs/env_lock.json`:

| Package | Version |
|---|---|
| Python | 3.12.12 |
| PyTorch | 2.10.0+cu128 (CUDA 12.8, cuDNN 91002) |
| torchvision | 0.25.0 |
| timm | 1.0.22 |
| NumPy | 2.2.6 |
| Pillow | 12.0.0 |
| scikit-fuzzy | 0.5.0 |

```bash
conda create -n fuzzyadp-moe python=3.12.12 pip -y
conda activate fuzzyadp-moe
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install timm==1.0.22 numpy==2.2.6 pillow==12.0.0 scikit-fuzzy==0.5.0 scipy networkx
```

`train_moe.py`, `train_single.py`, and `evaluate.py` compare the running environment with `runs/env_lock.json` before they write any output and stop on a mismatch (Python, the packages above, CUDA, cuDNN, driver, and GPU model). On a different machine, either create your own lock with `python p1/env_lock.py --update --reason "<reason>"` or pass `--env-lock none` to skip the check. Seeds and cuDNN determinism are set, but results are not guaranteed to be bit-identical across GPUs, drivers, or library versions.

The default training uses ImageNet-pretrained weights from timm, which are downloaded on first use.

## Data

Follow [datasets/README.md](datasets/README.md): download the RSCD images and the pinned per-day split CSVs, run `datasets/prepare_dataset.py` (fixes 59 malformed file names), then `datasets/make_subset.py`. The subset lists of this paper are already in `datasets/subsets/`, and their SHA-256 checksums are in `datasets/subsets/manifest.json` and in `datasets/README.md`.

The run scripts expect the images in `datasets/RSCD dataset-1million/` (containing `train/`, `vali_20k/`, `test_50k/`).

| List | Images | Use |
|---|---:|---|
| `train_s0.csv` – `train_s2.csv` | 41,130 each | pilot study, seeds 0–2 |
| `train_s3.csv` – `train_s5.csv` | 41,130 each | main experiments, seeds 3–5 |
| `val.csv` | 5,810 | validation (model selection and controller input) |
| `test.csv` | 59,793 | complete per-day test split, used once after all settings were frozen |

## Reproducing the main experiments

The runner works from `p1/` and skips runs that already have a `summary.json`. `DRY_RUN=1` only lists what would run.

```bash
cd p1
# Batch A (5 methods) and batch B (3 poor starting points x fixed, fuzzy), seeds 3, 4, 5 -> runs/final/
for b in A1 A2 A3 B1 B2 B3; do BATCH=$b LAM0=1.5 SIG0=0.4 shell/run_pilot.sh formal; done
# Batch C: timing runs with diagnostics off -> runs/final_timing/
for b in C1 C2 C3; do BATCH=$b LAM0=1.5 SIG0=0.4 shell/run_pilot.sh formal; done
# Test evaluation of best.pth and last.pth of the 33 runs (each checkpoint once)
LAM0=1.5 SIG0=0.4 shell/run_pilot.sh final-test
# Dense expert evaluation of the 12 MoE runs of batch A -> runs/final/experts.json
LAM0=1.5 SIG0=0.4 shell/run_pilot.sh final-experts
# Tables of the paper -> analysis/final/final_tables.md, final_summary.json
python summarize_final.py
```

The settings of the main experiments (fixed inside the runner): augmentation B, peak learning rate 1e-3, 50 epochs, 3 warm-up epochs, router R4, no auxiliary-loss warm-up; weight decay 1e-4 for the MoE variants and 0.01 for the monolithic model; Fixed MoE λ0 = 1.5, σ0 = 0.4, τ0 = 0.5; FuzzyAdp-MoE with `p1/fuzzy_v2_r3.json`. `python p1/train_moe.py --help` lists all options.

The pilot stages are run with the other `run_pilot.sh` subcommands listed at the top of that script; `pilot_study/Pilot_Study.md` describes each stage and its decision rule. Some script comments refer to `analysis/Pilot_Study.md` and `analysis/Pilot_Plan.md` of the development repository; the public version of the first is `pilot_study/Pilot_Study.md`. `Pilot_Study.md` also refers to working notes of the development repository (`Pilot_Plan.md`, `TODO.md`, `Status_2026-11-09.md`, `E1_Competitive_Loss.md`), which are not public. `pilot_study/Pilot_Study.md` is an English translation of the original Chinese record.

## Where the paper numbers come from

| Paper | Source in this repository |
|---|---|
| Classification results, robustness, routing table, training time | `analysis/final/final_tables.md` (built from `runs/final/`, `runs/final_timing/`) |
| Expert function analysis, main experiments | `runs/final/experts.json` |
| Fuzzy controller specification (Supplementary S1) | `p1/fuzzy_v2_r3.json`; earlier rounds in `p1/fuzzy_v2.json`, `p1/fuzzy_v2_r2.json` |
| Pilot study (Supplementary S2) | `pilot_study/` |
| Pilot σ sweep and winner-take-all loss (Supplementary S4) | `runs/stage9/`, `runs/oracle/`, dense evaluations under `runs/stage6*/`, `runs/stage8/`, `runs/stage9/` |
| Edge measurements | `deploy/results/summary.md` and the raw files under `deploy/results/<board>/` |

## Edge-device measurements

`deploy/` contains the TorchScript export, latency, and power scripts for Raspberry Pi 4B, Raspberry Pi 5, Jetson Xavier NX, and Jetson Orin Nano (CPU only, batch 1). See [deploy/README.md](deploy/README.md).

## License

MIT, see [LICENSE](LICENSE). The RSCD dataset and the per-day split are subject to their own terms; see their upstream pages linked in `datasets/README.md`.
