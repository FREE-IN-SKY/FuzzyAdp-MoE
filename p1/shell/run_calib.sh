#!/usr/bin/env bash
# run_calib.sh
#
# Runner for step 2 of stage −1 (calibration of routing defaults); design in analysis/Pilot_Plan.md.
# Step 1 (λ ∈ {0.01, 0.1, 0.5, 1.5}, σ=0.1, seed 0) was done with the old command; directories are runs/calib/lam<λ>_sig0.1 (no seed suffix).
#
# Usage (run under p1/; calling it with the full path from another directory also works, the script switches to p1/):
#   shell/run_calib.sh checkA   # part A: λ ∈ {0.1, 0.5, 1.5}, σ=0.1, seeds 1/2, 6 runs in total; then do the exclusion check
#   shell/run_calib.sh partB    # part B: λ ∈ {0.1, 1.5} × σ ∈ {0, 0.05, 0.2} × seeds 0/1/2, 18 runs in total; then make the final selection
#   shell/run_calib.sh final    # only make the final selection (8 configs)
# With DRY_RUN=1, only list the combinations to run; no training, no analysis.
#
# Completed runs (with summary.json) are skipped automatically; if a directory exists but has no summary.json, stop with an error; check and delete it, then rerun.
# partB runs only after the part A check passes (λ=0.5 still excluded); otherwise stop for discussion.

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")/.."

ROOT="../datasets/RSCD dataset-1million"
SUBSET="../datasets/subsets"
OUT="../runs/calib"
EPOCHS=20
WARMUP=3
TAU0=0.5
DRY_RUN="${DRY_RUN:-0}"

preflight() {
  python -c "import torch, timm" 2>/dev/null || { echo "[Error] The current python cannot import torch/timm; run conda activate mypy first" >&2; exit 1; }
  [[ -d "$ROOT" ]] || { echo "[Error] Dataset directory not found: $ROOT" >&2; exit 1; }
  [[ -f "$SUBSET/manifest.json" ]] || { echo "[Error] $SUBSET/manifest.json not found" >&2; exit 1; }
  python -c "from common import check_env; check_env()" \
    || { echo "[Error] The environment does not match runs/env_lock.json (differences above); no training was run" >&2; exit 1; }
  mkdir -p "$OUT/logs"
}

# calib_run <λ> <σ> <seed>
calib_run() {
  local lam=$1 sig=$2 seed=$3
  local name="lam${lam}_sig${sig}_s${seed}"
  local dir="$OUT/$name"
  if [[ -f "$dir/summary.json" ]]; then echo "[Skip] $name is already complete"; return 0; fi
  if [[ -d "$dir" ]]; then
    echo "[Error] $dir exists without summary.json and may have been interrupted; inspect and delete it, then rerun" >&2; exit 1
  fi
  if [[ "$DRY_RUN" == "1" ]]; then echo "[Dry run] $name"; return 0; fi
  echo "[Run] $name  ($(date '+%F %T'))"
  if ! python train_moe.py --controller fixed --root "$ROOT" --subset-dir "$SUBSET" --train-list "train_s${seed}.csv" \
      --aug B --lr 1e-3 --weight-decay 1e-4 --epochs "$EPOCHS" --warmup-epochs "$WARMUP" --seed "$seed" \
      --tau0 "$TAU0" --lambda0 "$lam" --sigma0 "$sig" --out-dir "$dir" 2>&1 | tee "$OUT/logs/$name.log"; then
    echo "[Error] $name training failed; see $OUT/logs/$name.log. Delete $dir before rerunning" >&2; exit 1
  fi
  [[ -f "$dir/summary.json" ]] || { echo "[Error] $name did not complete; see $OUT/logs/$name.log" >&2; exit 1; }
  python evaluate.py --checkpoint "$dir/last.pth" --root "$ROOT" --subset-dir "$SUBSET" --list val.csv \
    --out-dir "$dir/val_eval_last" > "$OUT/logs/$name.eval.log" 2>&1 \
    || { echo "[Error] $name evaluation failed; see $OUT/logs/$name.eval.log" >&2; exit 1; }
}

# The step 1 runs (seed 0, σ=0.1) have no seed suffix
step1_dir() { echo "$OUT/lam$1_sig0.1"; }

check_a() {
  python analyze_calib.py "$(step1_dir 0.1)" "$(step1_dir 0.5)" "$(step1_dir 1.5)" \
    "$OUT"/lam{0.1,0.5,1.5}_sig0.1_s{1,2} --out "$OUT/step2_checkA.json"
  python - "$OUT/step2_checkA.json" <<'PY'
import json, sys
g = [g for g in json.load(open(sys.argv[1]))["groups"] if abs(g["lambda0"] - 0.5) < 1e-9][0]
if g["excluded"]:
    print(f"\n[Check passed] lambda0=0.5 is still excluded ({'; '.join(g['excluded'])}); shell/run_calib.sh partB can be run")
else:
    print("\n[STOP: discuss before continuing] lambda0=0.5 is not excluded over 3 seeds; do not run partB until deciding whether to add it to the sigma grid")
PY
}

require_check_a() {
  [[ -f "$OUT/step2_checkA.json" ]] || { echo "[Error] $OUT/step2_checkA.json not found; run shell/run_calib.sh checkA first" >&2; exit 1; }
  python - "$OUT/step2_checkA.json" <<'PY' || exit 1
import json, sys
g = [g for g in json.load(open(sys.argv[1]))["groups"] if abs(g["lambda0"] - 0.5) < 1e-9][0]
if not g["excluded"]:
    sys.exit("[Error] The part A check did not pass (lambda0=0.5 is not excluded); per the plan, stop and discuss instead of running partB")
PY
}

final() {
  python analyze_calib.py "$(step1_dir 0.1)" "$(step1_dir 1.5)" "$OUT"/lam{0.1,1.5}_sig0.1_s{1,2} \
    "$OUT"/lam{0.1,1.5}_sig{0,0.05,0.2}_s{0,1,2} --out "$OUT/step2.json"
}

stage="${1:-}"
case "$stage" in
  checkA)
    [[ "$DRY_RUN" == "1" ]] || preflight
    for lam in 0.1 0.5 1.5; do for seed in 1 2; do calib_run "$lam" 0.1 "$seed"; done; done
    [[ "$DRY_RUN" == "1" ]] || check_a
    ;;
  partB)
    [[ "$DRY_RUN" == "1" ]] || { preflight; require_check_a; }
    for lam in 0.1 1.5; do for sig in 0 0.05 0.2; do for seed in 0 1 2; do
      calib_run "$lam" "$sig" "$seed"
    done; done; done
    [[ "$DRY_RUN" == "1" ]] || final
    ;;
  final)
    final
    ;;
  *)
    echo "Usage: shell/run_calib.sh checkA | partB | final   (DRY_RUN=1 for a dry run)" >&2
    exit 1
    ;;
esac
