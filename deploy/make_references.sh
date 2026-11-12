#!/usr/bin/env bash
# make_references.sh (laptop, conda env edge-ai): use the frozen p1/evaluate.py on CPU (fp32) to generate reference predictions on val.csv for the 5 best.pth of seed 3
# -> deploy/references/<run>/predictions.csv. The on-board bench_latency.py uses it to check the agreement rate (>= 0.999).
# Why CPU: the experiment machine's GPU autocast (fp16) reference has about 0.3% routing flips against the on-board CPU fp32 predictions (measured 0.9972 on 2026-11-09), which does not reach 0.999.
set -euo pipefail
cd "$(dirname "$0")/../p1"
PY=${PY:-/home/my/miniconda3/envs/edge-ai/bin/python}
CKPT_ROOT=${CKPT_ROOT:-../runs/final}   # relative to p1/; on the laptop, when checkpoints are unpacked in final_best_pth/, use CKPT_ROOT=../final_best_pth/runs/final
for d in single_augB_lr1e-3_wd0.01_e50_s3 moe-fixed_R4_lam1.5_sig0.4_aw0_augB_lr1e-3_wd1e-4_e50_s3 \
         moe-schedule_R4_lam1.5_sig0.4_aw0_augB_lr1e-3_wd1e-4_e50_s3 moe-auxfree_R4_lam0_sig0_aw0_augB_lr1e-3_wd1e-4_e50_s3 \
         moe-fuzzy_R4_lam1.5_sig0.4_aw0_augB_lr1e-3_wd1e-4_e50_s3; do
  out=../deploy/references/$d
  if [[ -f $out/predictions.csv ]]; then echo "[skip] $d"; continue; fi
  [[ -f $CKPT_ROOT/$d/best.pth ]] || { echo "[Error] $CKPT_ROOT/$d/best.pth missing"; exit 1; }
  echo "[ref] $d"
  CUDA_VISIBLE_DEVICES= "$PY" evaluate.py --checkpoint $CKPT_ROOT/$d/best.pth --root "../datasets/RSCD dataset-1million" \
    --subset-dir ../datasets/subsets --list val.csv --env-lock none --num-workers 4 --out-dir "$out"
done
echo "references done"
