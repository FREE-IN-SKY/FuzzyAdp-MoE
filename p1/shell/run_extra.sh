#!/usr/bin/env bash
# Follow-up runs on the experiment machine (analysis/Pilot_Study.md 13.6, 13.7 row 7; on 2026-11-10 the author decided to run both). Run only after all of C1 ~ C3 in section 11 are done, and after git pull.
#   1. Complete the σ sweep: fixed σ0 0 seeds 1, 2 (mix loss), output to runs/stage9/, same name and settings as the existing seed 0;
#      seed 0 only gets the dense evaluation (the original evaluation is under dense_1109/ on the laptop, not in runs/).
#   2. oracle σ0 0 seeds 0, 1, 2 (--cls-loss oracle), output to runs/oracle/. E2b on the laptop (seeds 0 ~ 2) only sets the direction; the numbers in the paper come from here.
# Other settings same as σ0 0 seed 0 in runs/stage9: fixed, R4, λ0 1.5, τ0 0.5, no aux warmup, augmentation B, lr 1e-3, wd 1e-4, 50 epoch, lr warmup 3 epochs, batch 64,
# train_s<seed>.csv, diagnostics on. After each training: evaluate last.pth on val.csv (val_eval_last/, same as run_pilot.sh), then dense evaluation of best.pth (val_dense_best/).
# Finally run analyze_experts.py for each group: runs/stage9/experts_sigma0.json, runs/oracle/experts.json. Validation set only, descriptive only.
# Completed steps are skipped; stop with an error when a directory exists but is incomplete; stop on any failed step. 5 trainings in total, about 6 hours.
#
#   cd p1 && DRY_RUN=1 shell/run_extra.sh     # only list the steps to do
#   cd p1 && shell/run_extra.sh
set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")/.."

ROOT="../datasets/RSCD dataset-1million"
SUBSET="../datasets/subsets"
DRY_RUN=${DRY_RUN:-0}

die() { echo "[Error] $*" >&2; exit 1; }

python -c "import torch, timm" 2>/dev/null || die "The current python cannot import torch/timm; run conda activate mypy first"
[[ -d "$ROOT" ]] || die "Dataset directory not found: $ROOT"
python -c "from common import check_env; check_env()" || die "The environment does not match runs/env_lock.json (differences above); nothing was run"
grep -q -- '"--cls-loss"' train_moe.py || die "train_moe.py has no --cls-loss; git pull first (needs b66899f or later)"
[[ -z "$(git status --porcelain -- . ../datasets/subsets)" ]] || die "p1/ or datasets/subsets/ has uncommitted changes; commit or stash them first"
shopt -s nullglob; c_done=(../runs/final_timing/*/summary.json); shopt -u nullglob; n_c=${#c_done[@]}
[[ "$n_c" -ge 15 || "${FORCE:-0}" == 1 ]] || die "runs/final_timing has $n_c/15 finished timing runs; finish C1 ~ C3 first (the timing batches need the machine alone)"

# run_one <output dir> <name> <cls-loss> <seed> : training + val_eval_last + dense evaluation
run_one() {
  local out=$1 name=$2 loss=$3 seed=$4 dir="$1/$2"
  [[ "$DRY_RUN" == 1 ]] || mkdir -p "$out/logs"
  if [[ -f "$dir/summary.json" ]]; then
    echo "[Skip] $name training is already complete"
  else
    [[ -d "$dir" ]] && die "$dir exists without summary.json and may have been interrupted; inspect and delete it, then rerun"
    if [[ "$DRY_RUN" == 1 ]]; then echo "[Dry run] train $out/$name (cls-loss $loss, seed $seed)"; else
      echo "[Run] $name  ($(date '+%F %T'))"
      python train_moe.py --controller fixed --cls-loss "$loss" --router R4 --tau0 0.5 --lambda0 1.5 --sigma0 0 --aux-warmup-frac 0 \
        --root "$ROOT" --subset-dir "$SUBSET" --train-list "train_s${seed}.csv" --aug B --lr 1e-3 --weight-decay 1e-4 \
        --epochs 50 --warmup-epochs 3 --batch-size 64 --seed "$seed" --out-dir "$dir" 2>&1 | tee "$out/logs/$name.log" \
        || die "$name training failed; see $out/logs/$name.log. Delete $dir before rerunning"
      [[ -f "$dir/summary.json" ]] || die "$name did not complete; see $out/logs/$name.log"
    fi
  fi
  if [[ ! -f "$dir/val_eval_last/metrics.json" ]]; then
    [[ -e "$dir/val_eval_last" ]] && die "$dir/val_eval_last exists without metrics.json; inspect and delete it, then rerun"
    if [[ "$DRY_RUN" == 1 ]]; then echo "[Dry run] val eval $name/last.pth"; else
      python evaluate.py --checkpoint "$dir/last.pth" --root "$ROOT" --subset-dir "$SUBSET" --list val.csv \
        --out-dir "$dir/val_eval_last" > "$out/logs/$name.eval.log" 2>&1 || die "$name evaluation failed; see $out/logs/$name.eval.log"
    fi
  fi
  dense "$out" "$name"
}

# dense <output dir> <name> : dense evaluation of best.pth on val.csv
dense() {
  local out=$1 name=$2 ev="$1/$2/val_dense_best"
  [[ "$DRY_RUN" == 1 ]] || mkdir -p "$out/logs"
  if [[ -f "$ev/dense_experts.npz" ]]; then echo "[Skip] $name/val_dense_best is already complete"; return 0; fi
  [[ -e "$ev" ]] && die "$ev exists without dense_experts.npz and may have been interrupted; inspect and delete it, then rerun"
  if [[ "$DRY_RUN" == 1 ]]; then echo "[Dry run] dense eval $name/best.pth"; return 0; fi
  [[ -f "$out/$name/best.pth" ]] || die "$out/$name/best.pth not found"
  echo "[Dense] $name/best.pth  ($(date '+%F %T'))"
  python evaluate.py --checkpoint "$out/$name/best.pth" --root "$ROOT" --subset-dir "$SUBSET" --list val.csv --dense-experts \
    --out-dir "$ev" > "$out/logs/$name.dense.log" 2>&1 || die "$name dense evaluation failed; see $out/logs/$name.dense.log"
}

analyze() {  # analyze <output dir> <output file> <names...>
  local out=$1 file=$2; shift 2
  if [[ "$DRY_RUN" == 1 ]]; then echo "[Dry run] analyze_experts -> $out/$file ($# runs)"; return 0; fi
  python analyze_experts.py --runs-dir "$out" --names "$@" --subset-dir "$SUBSET" --out "$out/$file" \
    2>&1 | tee "$out/logs/analyze_experts_${file%.json}.log" || die "analyze_experts failed for $out/$file"
}

# 1. Complete the σ sweep (runs/stage9)
S9=../runs/stage9
sig=()
for s in 0 1 2; do
  name=moe-fixed_R4_lam1.5_sig0_aw0_augB_lr1e-3_wd1e-4_e50_s$s
  sig+=("$name")
  if [[ "$s" == 0 ]]; then
    [[ -f "$S9/$name/summary.json" ]] || die "$S9/$name (the existing sigma-0 seed 0) not found"
    dense "$S9" "$name"
  else
    run_one "$S9" "$name" mix "$s"
  fi
done
analyze "$S9" experts_sigma0.json "${sig[@]}"

# 2. oracle (runs/oracle)
OR=../runs/oracle
orc=()
for s in 0 1 2; do
  name=moe-fixed_R4_oracle_lam1.5_sig0_aw0_augB_lr1e-3_wd1e-4_e50_s$s
  orc+=("$name")
  run_one "$OR" "$name" oracle "$s"
done
analyze "$OR" experts.json "${orc[@]}"

echo "[$(date '+%F %T')] run_extra done"
