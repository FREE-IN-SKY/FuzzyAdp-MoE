#!/usr/bin/env bash
# run_pilot.sh
#
# Runner for pilot study stages 0 to 4; design in analysis/Pilot_Plan.md. Each stage first runs any missing runs (follow-up runs), then calls
# summarize_pilot.py to make the selection by the predefined rules; results are written to runs/pilot/stage*.json.
#
# Usage (run under p1/; calling it with the full path from another directory also works, the script switches to p1/). λ0, σ0 are the results of stage −1, passed via environment variables:
#   LAM0=<λ0> SIG0=<σ0> shell/run_pilot.sh base                       # stage 0: MoE seeds 0/1/2 + single seed 0, compute X
#   LAM0=<λ0> SIG0=<σ0> shell/run_pilot.sh aug                        # stage 1: augmentation A / B
#   LAM0=<λ0> SIG0=<σ0> shell/run_pilot.sh lr <AUG>                   # stage 2: lr 1e-4 / 3e-4 / 1e-3 / 3e-3
#   LAM0=<λ0> SIG0=<σ0> shell/run_pilot.sh epochs <AUG> <LR> [WD]     # stage 3: 50 / 70 / 100 (MoE only); WD defaults to 1e-4,
#                                                                   #   passed only when the stage 4 conditional recheck selects the smaller lr and the epoch is re-selected retroactively
#   LAM0=<λ0> SIG0=<σ0> shell/run_pilot.sh epochs-ext <AUG> <LR> [WD] # stage 3 extension: 140, only when the extension rule is triggered
#   LAM0=<λ0> SIG0=<σ0> shell/run_pilot.sh wd <AUG> <LR> <EP>         # stage 4: wd 1e-4 / 0.01 / 0.05 / 0.1
#   LAM0=<λ0> SIG0=<σ0> shell/run_pilot.sh lr-recheck <AUG> <LR> <EP> <WD>  # stage 4 conditional recheck: the selected LR vs the next smaller candidate
#                                                                   #   (1e-3 vs 3e-4, 3e-3 vs 1e-3)
#   LAM0=1.5 SIG0=0.2 shell/run_pilot.sh repro                       # reproduction run: the config of stage 0 seed 0, output to runs/repro/,
#                                                                   #   compared bit-for-bit with stage 0 automatically after finishing, exits non-zero on failure (see Pilot_Plan.md "Reproduction run")
#   LAM0=1.5 SIG0=0.2 shell/run_pilot.sh router                      # stage 5: R0 seeds 1/2 (first compared bit-for-bit with stage 0, continues only if passed)
#                                                                   #   + R4 seeds 0/1/2, output to runs/router/; R0 seed 0 taken from runs/repro/
#   LAM0=1.5 SIG0=0.2 shell/run_pilot.sh fixed6                      # stage 6 A: seed 0 of two candidates each for λ, σ, τ on R4, then follow-up runs by the rules
#                                                                   #   (listed by summarize_stage6.py), output to runs/stage6/; center point taken from runs/router/
#   LAM0=1.5 SIG0=0.2 shell/run_pilot.sh fixed6-ext                  # λ 4.5 extension of stage 6 A: follow-up runs of λ 4.5 seeds 1, 2, then re-summarize
#   LAM0=1.5 SIG0=0.2 shell/run_pilot.sh nowarmup-check              # stage 6 C step 1: center point without aux warmup, seeds 0/1/2, output to runs/stage6c/,
#                                                                   #   paired comparison with the center point in runs/router/
#   LAM0=1.5 SIG0=0.2 shell/run_pilot.sh fixed6nw                    # stage 6 C step 2: after adopting no warmup, rerun stage 6 A (including the λ 4.5 extension)
#   LAM0=1.5 SIG0=0.2 shell/run_pilot.sh study3b | study4 | study4wd | study4single | study6 | study7   # reruns of analysis/Pilot_Study.md
#                                                                   #   (stages 3b, 4, 6, 7), run in this order; rules in summarize_study.py.
#                                                                   #   study6 stops after selecting σ0 = 0.4; then run study6b (recheck), then study7:
#   LAM0=1.5 SIG0=0.4 shell/run_pilot.sh study6b                     # stage 6b: σ 0.8, 70 epoch, wd 0.05, lr on both sides, stops if not passed
#   ROUND=1 LAM0=1.5 SIG0=0.4 shell/run_pilot.sh study8               # stage 8 trial run, round ROUND: fuzzy (fuzzy_v2.json) and schedule, seed 0 each,
#                                                                   #   50 epoch; then summarize_study.py --part 8 gives the verdict by control behavior
#   ROUND=2 LAM0=1.5 SIG0=0.4 shell/run_pilot.sh study8               # round 2: fuzzy_v2_r2.json, fuzzy seeds 0/1/2; verdict adds a 3-seed pairing
#   ROUND=3 LAM0=1.5 SIG0=0.4 shell/run_pilot.sh study8               # round 3: fuzzy_v2_r3.json (σ of R5, R8, R9 changed to medium), seeds 0/1/2;
#                                                                   #   verdict is the pairing with round 2 fuzzy (tests the σ hypothesis, 10.3)
#   LAM0=1.5 SIG0=0.4 shell/run_pilot.sh study9a                      # stage 9a: 6 bad starting points for fixed (λ0 0.17, 13.5; σ0 0, 1.6; τ0 0.125, 2.0),
#                                                                   #   seed 0, 50 epoch; then summarize_study.py --part 9a gives the "bad" verdict and the 9b candidates
#   LAM0=1.5 SIG0=0.4 shell/run_pilot.sh study9a_ext                  # stage 9a extension: λ0 0, 40.5; σ0 3.2; combinations (λ0, σ0, τ0) = (0, 0, 0.125),
#                                                                   #   (40.5, 3.2, 2.0), (0, 3.2, 2.0); seed 0; then verdict over 12 candidates (study9a_ext.json)
#   LAM0=1.5 SIG0=0.4 shell/run_pilot.sh study9b                      # stage 9b: bad starting points selected by study9a_ext.json; step 1 fixed seeds 1, 2,
#                                                                   #   after "bad" is confirmed, step 2 fuzzy (fuzzy_v2_r3.json, directory name suffixed with _r3) seeds 0/1/2
#                                                                   #   (study9b_r3.json; results of the round 2 version are in study9b.json, not overwritten)
#   BATCH=<A1..C3> LAM0=1.5 SIG0=0.4 shell/run_pilot.sh formal         # formal experiment (analysis/Pilot_Study.md 11.2), indices 1/2/3 are seeds 3/4/5:
#                                                                   #   A main comparison (single, fixed, schedule, auxfree, fuzzy), B robustness (3 bad starting points × fixed, fuzzy),
#                                                                   #   C timing (same as A, diagnostics off, output to runs/final_timing/); A, B output to runs/final/
#   LAM0=1.5 SIG0=0.4 shell/run_pilot.sh final-test                   # after formal experiment A, B are done: best.pth, last.pth of the 33 runs each evaluated once on test.csv
#   LAM0=1.5 SIG0=0.4 shell/run_pilot.sh final-experts                # expert function analysis of the 12 MoE runs of batch A (validation set, --dense-experts; experts.json)
#   LAM0=1.5 SIG0=0.4 shell/run_pilot.sh study8-seeds                 # after stage 8 is frozen: fuzzy (fuzzy_v2.json) seeds 1, 2, 50 epoch, without round suffix
#                                                                   #   study4 stops at the epoch step (selection changed from 70 to 50);
#                                                                   #   study4wd, study6, study7 are run at 50 epoch
#   LAM0=1.5 SIG0=0.2 shell/run_pilot.sh auxfree6                    # stage 6 B: auxfree (R4, seed 0, σ=0, λ=0), τ0 taken from the final selection of A
# With DRY_RUN=1, only list the combinations to run; no training, no summarizing.
#
# Each stage lists the full candidate set; completed runs (with summary.json) are skipped automatically, so reuse needs no manual judgment;
# if a directory exists but has no summary.json, stop with an error; check and delete it, then rerun. lr, wd accept only fixed spellings (e.g. 1e-3, 0.05),
# to keep directory names consistent. After a MoE run completes, evaluate last.pth on val.csv, output val_eval_last/ (for routing analysis).

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")/.."

ROOT="../datasets/RSCD dataset-1million"
SUBSET="../datasets/subsets"
OUT="../runs/pilot"
ROUTER=R0  # changed to R4 only in the router, fixed6, auxfree6 stages; not read from environment variables, to avoid other stages being changed by accident
CONTROLLER=fixed  # changed to auxfree only in the auxfree6 stage
AUX_WARMUP=""  # empty = training script default (aux coefficient multiplied by 0.1 for the first 1/3 of training); 0 = no warmup (from stage 6 C on), directory name suffixed with _aw0
STAGE6_EP=70  # number of epochs for stage6_run; changed to 50 for study6 (selection of stage 4.3)
GAMMA=1e-3  # bias update step size of auxfree; when not 1e-3, directory name suffixed with _g<γ> (changed only in study7)
FUZZY_CONFIG=fuzzy_v2.json  # controller config of fuzzy (study8)
TAG=""  # directory name suffix (round _r<ROUND> of study8)
DIAG=1  # 1 = training diagnostics on (default); 0 = --no-diagnostics, and no validation-set evaluation after training (only for the timing batch C of formal)
LAST_RUN=""  # name of the run directory most recently handled by pilot_run (used by formal to collect the runs of this batch)
NAME_ONLY=0  # 1 = pilot_run only computes the directory name (LAST_RUN) and returns, no checks, no training (used by final-test, final-experts to list the formal runs)
BATCH_RUNS=()  # run directory names of this batch collected by formal_batch
WARMUP=3
TAU0=0.5
DRY_RUN="${DRY_RUN:-0}"
LAM0="${LAM0:-}"
SIG0="${SIG0:-}"

die() { echo "[Error] $*" >&2; exit 1; }

check_value() {  # check_value <name> <value> <allowed values...>
  local label=$1 v=$2; shift 2
  for a in "$@"; do [[ "$v" == "$a" ]] && return 0; done
  die "$label must be one of {$*} (exact spelling), got: $v"
}

preflight() {
  python -c "import torch, timm" 2>/dev/null || die "The current python cannot import torch/timm; run conda activate mypy first"
  [[ -d "$ROOT" ]] || die "Dataset directory not found: $ROOT"
  [[ -f "$SUBSET/manifest.json" ]] || die "$SUBSET/manifest.json not found"
  python -c "from common import check_env; check_env()" \
    || die "The environment does not match runs/env_lock.json (differences above); no training was run"
  mkdir -p "$OUT/logs"
  local calib="../runs/calib/step2.json"
  if [[ -f "$calib" ]]; then
    python - "$calib" "$LAM0" "$SIG0" <<'PY'
import json, math, sys
p = json.load(open(sys.argv[1]))["pick"]
lam, sig = float(sys.argv[2]), float(sys.argv[3])
if not p or not (math.isclose(p["lambda0"], lam) and math.isclose(p["sigma0"], sig)):
    print(f"[Warning] LAM0={sys.argv[2]}, SIG0={sys.argv[3]} differ from the stage -1 selection {p}; make sure this is intentional")
PY
  else
    echo "[Warning] $calib (the stage -1 final selection) not found; check where LAM0 and SIG0 come from"
  fi
}

# pilot_run <single|moe> <augmentation> <lr> <wd> <epochs> <seed>
pilot_run() {
  local model=$1 aug=$2 lr=$3 wd=$4 ep=$5 seed=$6 name
  if [[ "$model" == single ]]; then
    name="single_aug${aug}_lr${lr}_wd${wd}_e${ep}_s${seed}"
  else
    local tau_tag=""  # τ0 is written into the directory name only when it is not 0.5; directory names of existing runs stay unchanged
    if [[ "$TAU0" != 0.5 ]]; then tau_tag="_tau${TAU0}"; fi
    local aw_tag=""
    if [[ "$AUX_WARMUP" == 0 ]]; then aw_tag="_aw0"; fi
    local g_tag=""
    if [[ "$CONTROLLER" == auxfree && "$GAMMA" != 1e-3 ]]; then g_tag="_g${GAMMA}"; fi
    name="moe-${CONTROLLER}_${ROUTER}_lam${LAM0}_sig${SIG0}${tau_tag}${aw_tag}${g_tag}_aug${aug}_lr${lr}_wd${wd}_e${ep}_s${seed}${TAG}"
  fi
  LAST_RUN=$name
  [[ "$NAME_ONLY" == 1 ]] && return 0
  local dir="$OUT/$name"
  if [[ -f "$dir/summary.json" ]]; then echo "[Skip] $name is already complete"; return 0; fi
  [[ -d "$dir" ]] && die "$dir exists without summary.json and may have been interrupted; inspect and delete it, then rerun"
  if [[ "$DRY_RUN" == "1" ]]; then echo "[Dry run] $name"; return 0; fi
  echo "[Run] $name  ($(date '+%F %T'))"
  local common=(--root "$ROOT" --subset-dir "$SUBSET" --train-list "train_s${seed}.csv" --aug "$aug" --lr "$lr"
                --weight-decay "$wd" --epochs "$ep" --warmup-epochs "$WARMUP" --seed "$seed" --out-dir "$dir")
  local cmd=(python train_single.py)
  [[ "$model" == moe ]] && cmd=(python train_moe.py --controller "$CONTROLLER" --router "$ROUTER" --tau0 "$TAU0" --lambda0 "$LAM0" --sigma0 "$SIG0")
  [[ "$model" == moe && "$AUX_WARMUP" == 0 ]] && cmd+=(--aux-warmup-frac 0)
  [[ "$model" == moe && "$CONTROLLER" == auxfree ]] && cmd+=(--auxfree-gamma "$GAMMA")
  [[ "$model" == moe && "$CONTROLLER" == fuzzy ]] && cmd+=(--fuzzy-config "$FUZZY_CONFIG")
  [[ "$DIAG" == 0 ]] && cmd+=(--no-diagnostics)
  if ! "${cmd[@]}" "${common[@]}" 2>&1 | tee "$OUT/logs/$name.log"; then
    die "$name training failed; see $OUT/logs/$name.log. Delete $dir before rerunning"
  fi
  [[ -f "$dir/summary.json" ]] || die "$name did not complete; see $OUT/logs/$name.log"
  if [[ "$model" == moe && "$DIAG" == 1 ]]; then
    python evaluate.py --checkpoint "$dir/last.pth" --root "$ROOT" --subset-dir "$SUBSET" --list val.csv \
      --out-dir "$dir/val_eval_last" > "$OUT/logs/$name.eval.log" 2>&1 \
      || die "$name evaluation failed; see $OUT/logs/$name.eval.log"
  fi
}

summarize() {  # summarize <output name> <summarize_pilot.py args...>
  local out=$1; shift
  [[ "$DRY_RUN" == "1" ]] && return 0
  python summarize_pilot.py --pilot-dir "$OUT" --tau0 "$TAU0" --lambda0 "$LAM0" --sigma0 "$SIG0" "$@" --out "$OUT/$out"
}

require_json() {  # require_json <file> <python condition expression, variable d is the JSON> <failure message>
  [[ -f "$1" ]] || die "$1 not found; run the corresponding stage first"
  python - "$1" "$2" "$3" <<'PY' || exit 1
import json, math, sys
d = json.load(open(sys.argv[1]))
close = lambda a, b: a is not None and math.isclose(float(a), float(b), rel_tol=1e-6)
if not eval(sys.argv[2]):
    sys.exit("[Error] " + sys.argv[3])
PY
}

compare_runs() {  # compare_runs <new run dir> <reference run dir> <result JSON>: bit-for-bit comparison, exits 0 when passed (only reads the two run directories)
  python - "$1" "$2" "$3" <<'PY'
import csv, json, os, sys
new, old, out = sys.argv[1:4]
res = {"new": new, "reference": old, "passed": False, "first_difference": None, "checked": {}}

def finish(passed, msg):
    res["passed"], res["first_difference"] = passed, (None if passed else msg)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(("[Repro passed] " if passed else "[Repro FAILED] ") + msg)
    sys.exit(0 if passed else 1)

for d in (new, old):
    if not os.path.exists(os.path.join(d, "summary.json")):
        finish(False, f"summary.json not found in {d}")
rows = {k: list(csv.DictReader(open(os.path.join(d, "history.csv"), newline=""))) for k, d in (("new", new), ("old", old))}
if len(rows["new"]) != len(rows["old"]):
    finish(False, f"history.csv has {len(rows['new'])} epochs in the repro run but {len(rows['old'])} in the reference")
cols = [c for c in rows["old"][0] if c not in ("train_time_s", "val_time_s")]
missing = [c for c in cols if c not in rows["new"][0]]
if missing:
    finish(False, f"columns missing in the repro run: {missing}")
for a, b in zip(rows["old"], rows["new"]):
    for c in cols:
        if a[c] != b[c]:
            finish(False, f"history.csv epoch {a['epoch']}, column {c}: reference {a[c]!r}, repro {b[c]!r}")
res["checked"]["history_columns"] = len(cols)
res["checked"]["epochs"] = len(rows["old"])
sn, so = (json.load(open(os.path.join(d, "summary.json"))) for d in (new, old))
for k in ("best_epoch", "best_val_acc_pi"):
    if sn.get(k) != so.get(k):
        finish(False, f"summary.json {k}: reference {so.get(k)!r}, repro {sn.get(k)!r}")
mp = [os.path.join(d, "val_eval_last", "metrics.json") for d in (new, old)]
if not all(os.path.exists(x) for x in mp):
    finish(False, f"val_eval_last/metrics.json missing: {[x for x in mp if not os.path.exists(x)]}")
mn, mo = (json.load(open(x)) for x in mp)
for k in ("acc", "acc_pi", "balanced_acc", "macro_f1"):
    if mn.get(k) != mo.get(k):
        finish(False, f"val_eval_last/metrics.json {k}: reference {mo.get(k)!r}, repro {mn.get(k)!r}")
finish(True, f"{len(cols)} history columns over {len(rows['old'])} epochs, summary and evaluation metrics are identical to {old}")
PY
}

stage6_run() {  # stage6_run <λ> <σ> <τ> <seed>: one Fixed MoE run of stage 6 A (augmentation B, lr 1e-3, wd 1e-4, $STAGE6_EP epoch)
  local LAM0=$1 SIG0=$2 TAU0=$3
  pilot_run moe B 1e-3 1e-4 "$STAGE6_EP" "$4"
}

study6b_run() {  # study6b_run <σ> <lr> <wd> <epochs> <seed>: one Fixed MoE run of stage 6b (augmentation B, λ 1.5, τ 0.5)
  local LAM0=1.5 SIG0=$1 TAU0=0.5
  pilot_run moe B "$2" "$3" "$4" "$5"
}

study_summary() {  # study_summary <part> <output JSON> [other args]: calls summarize_study.py (rules of Pilot_Study.md)
  local part=$1 out=$2; shift 2
  python summarize_study.py --part "$part" --nw-dir ../runs/stage6c --auxfree-dir ../runs/stage6 "$@" --out "$out"
}

study_field() {  # study_field <JSON> <field>: prints one top-level field of the summary result
  python -c 'import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"
}

final_check() {  # final_check <batch> <batch dir> <batch A dir> <run names...>: check at the end of each formal experiment batch (Pilot_Study.md 11.2)
  python - "$@" <<'PY'
import csv, json, sys
from pathlib import Path
batch, out, ref, names = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4:]
cols = ("train_cls_loss", "train_acc", "val_loss", "val_acc", "val_acc_pi", "val_balanced_acc", "val_macro_f1")
errs, commits = [], set()
for n in names:
    d = out / n
    if not (d / "summary.json").exists():
        errs.append(f"{n}: no summary.json in {out}")
        continue
    c = json.loads((d / "config.json").read_text(encoding="utf-8"))
    s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    with open(d / "history.csv", newline="") as f:
        h = list(csv.DictReader(f))
    e = c.get("env", {})
    commits.add(str(e.get("git_commit")))
    if c["epochs"] != 50 or len(h) != 50:
        errs.append(f"{n}: config epochs {c['epochs']}, {len(h)} epochs logged (expected 50)")
    if e.get("git_dirty") is not False:
        errs.append(f"{n}: git_dirty = {e.get('git_dirty')}")
    line = f"{n}: best val A_pi {s['best_val_acc_pi']:.4f} (epoch {s['best_epoch']}), {s.get('total_time_min')} min"
    if batch.startswith("C"):  # timing run: accuracy columns must be bit-identical to the matching A run, otherwise this timing is invalid
        r = ref / n
        if not (r / "summary.json").exists():
            errs.append(f"{n}: the matching batch A run {r} is missing; cannot verify this timing run")
        else:
            with open(r / "history.csv", newline="") as f:
                hr = list(csv.DictReader(f))
            # history.csv of single writes train_loss, MoE writes train_cls_loss
            hk, rk = set(h[0] if h else ()), set(hr[0] if hr else ())
            ck = [("train_loss" if k == "train_cls_loss" and k not in hk else k) for k in cols]
            miss = [k for k in ck if k not in hk or k not in rk]
            bad = [k for k in ck if k not in miss and [row[k] for row in h] != [row[k] for row in hr]]
            if miss:
                errs.append(f"{n}: column(s) {miss} missing in {d} or {r}; cannot verify this timing run")
            elif bad:
                errs.append(f"{n}: differs from {r} in {bad}; this timing run is invalid")
            else:
                line += "; accuracy columns identical to batch A"
    print(line)
if len(commits) > 1:
    errs.append(f"runs come from different commits: {sorted(commits)}")
print(f"Batch {batch}: {len(names)} runs; commit {', '.join(sorted(commits))}")
if errs:
    print("\n".join("[Error] " + x for x in errs), file=sys.stderr)
    sys.exit(1)
PY
}

formal_batch() {  # formal_batch <A|B> <seed>: runs of one formal experiment batch (Pilot_Study.md 11.2), method order fixed; directory names collected into BATCH_RUNS
  local kind=$1 seed=$2 spec
  ROUTER=R4; AUX_WARMUP=0; TAG=""; GAMMA=1e-3; FUZZY_CONFIG=fuzzy_v2_r3.json
  BATCH_RUNS=()
  if [[ "$kind" == B ]]; then
    for spec in "0 0.4 0.5" "0 0 0.125" "40.5 3.2 2.0"; do
      read -r LAM0 SIG0 TAU0 <<< "$spec"
      for CONTROLLER in fixed fuzzy; do pilot_run moe B 1e-3 1e-4 50 "$seed"; BATCH_RUNS+=("$LAST_RUN"); done
    done
  else
    pilot_run single B 1e-3 0.01 50 "$seed"; BATCH_RUNS+=("$LAST_RUN")
    LAM0=1.5; SIG0=0.4; TAU0=0.5
    for CONTROLLER in fixed schedule; do pilot_run moe B 1e-3 1e-4 50 "$seed"; BATCH_RUNS+=("$LAST_RUN"); done
    CONTROLLER=auxfree; LAM0=0; SIG0=0
    pilot_run moe B 1e-3 1e-4 50 "$seed"; BATCH_RUNS+=("$LAST_RUN")
    CONTROLLER=fuzzy; LAM0=1.5; SIG0=0.4
    pilot_run moe B 1e-3 1e-4 50 "$seed"; BATCH_RUNS+=("$LAST_RUN")
  fi
  CONTROLLER=fixed; LAM0=1.5; SIG0=0.4; TAU0=0.5
}

formal_all() {  # formal_all <A|B>...: list all run directory names of these batches (seeds 3, 4, 5) into BATCH_RUNS, no training
  local all=() kind seed
  NAME_ONLY=1
  for kind in "$@"; do for seed in 3 4 5; do formal_batch "$kind" "$seed"; all+=("${BATCH_RUNS[@]}"); done; done
  NAME_ONLY=0
  BATCH_RUNS=("${all[@]}")
}

study_todo() {  # study_todo <JSON> <field names...>: one line per pending run, output in the given field order
  python -c 'import json, sys; [print(*(r[k] for k in sys.argv[2:])) for r in json.load(open(sys.argv[1]))["to_run"]]' "$@"
}

stage="${1:-}"
[[ -n "$stage" ]] || die "Missing stage argument; see the usage comment at the top of this script"
[[ -n "$LAM0" && -n "$SIG0" ]] || die "Set LAM0 and SIG0 (the stage -1 results) first, e.g. LAM0=0.1 SIG0=0.05 shell/run_pilot.sh base"
if [[ "$stage" == repro ]]; then OUT="../runs/repro"; fi  # reproduction run kept separately, to avoid duplicating the stage 0 run configs in runs/pilot/
if [[ "$stage" == router ]]; then OUT="../runs/router"; fi  # same reason: R0 seeds 1/2 have the same config as stage 0
if [[ "$stage" == fixed6 || "$stage" == fixed6-ext || "$stage" == auxfree6 ]]; then OUT="../runs/stage6"; fi
if [[ "$stage" == nowarmup-check || "$stage" == fixed6nw ]]; then OUT="../runs/stage6c"; fi  # stage 6 C: runs without aux warmup kept separately
if [[ "$stage" == study3b || "$stage" == study4 || "$stage" == study4wd || "$stage" == study6 || "$stage" == study6b ]]; then OUT="../runs/stage6c"; fi  # reruns of Pilot_Study.md (no aux warmup)
if [[ "$stage" == study4single ]]; then OUT="../runs/pilot"; fi  # summarize_study.py reads single from here
if [[ "$stage" == study7 ]]; then OUT="../runs/stage6"; fi  # auxfree kept together with the existing γ = 1e-3 runs
if [[ "$stage" == study8 || "$stage" == study8-seeds ]]; then OUT="../runs/stage8"; fi
if [[ "$stage" == study9a || "$stage" == study9a_ext || "$stage" == study9b ]]; then OUT="../runs/stage9"; fi  # stage 9: fixed, fuzzy under bad starting points
if [[ "$stage" == formal || "$stage" == final-test || "$stage" == final-experts ]]; then  # formal experiment (Pilot_Study.md 11): A, B in runs/final/, C (timing) in runs/final_timing/
  OUT="../runs/final"
  if [[ "$stage" == formal ]]; then
    check_value BATCH "${BATCH:-}" A1 A2 A3 B1 B2 B3 C1 C2 C3
    if [[ "${BATCH:0:1}" == C ]]; then OUT="../runs/final_timing"; fi
  fi
  if [[ "$DRY_RUN" != "1" && -n "$(git status --porcelain -- . ../datasets/subsets)" ]]; then
    git status --porcelain -- . ../datasets/subsets
    die "p1/ or datasets/subsets/ has uncommitted or untracked files (listed above); the formal stages require a clean checkout"
  fi
fi
[[ "$DRY_RUN" == "1" ]] || preflight

case "$stage" in
  base)
    for s in 0 1 2; do pilot_run moe B 1e-3 1e-4 70 "$s"; done
    pilot_run single B 1e-3 1e-4 70 0
    [[ "$DRY_RUN" == "1" ]] || python analyze_calib.py "$OUT"/moe-fixed_R0_lam${LAM0}_sig${SIG0}_augB_lr1e-3_wd1e-4_e70_s{0,1,2} \
      | sed '/^-> Select/d; /^  vs\. runner-up/d; s/^\[Excluded\]/[STOP: discuss before continuing - routing recheck failed]/;
             s/^Note: .*/Note: stage 0 only rechecks routing (min usage, nmi, cos_gap, w1); no selection is made./'
    summarize stage0.json --stage base
    ;;
  aug)
    for m in single moe; do for a in A B; do pilot_run "$m" "$a" 1e-3 1e-4 70 0; done; done
    summarize stage1.json --stage aug
    ;;
  lr)
    aug="${2:-}"; check_value AUG "$aug" A B
    for m in single moe; do for lr in 1e-4 3e-4 1e-3 3e-3; do pilot_run "$m" "$aug" "$lr" 1e-4 70 0; done; done
    summarize stage2.json --stage lr --aug "$aug"
    ;;
  epochs)
    aug="${2:-}"; lr="${3:-}"; wd="${4:-1e-4}"
    check_value AUG "$aug" A B; check_value LR "$lr" 1e-4 3e-4 1e-3 3e-3; check_value WD "$wd" 1e-4 0.01 0.05 0.1
    for ep in 50 70 100; do pilot_run moe "$aug" "$lr" "$wd" "$ep" 0; done
    summarize stage3.json --stage epochs --aug "$aug" --lr "$lr" --wd "$wd"
    ;;
  epochs-ext)
    aug="${2:-}"; lr="${3:-}"; wd="${4:-1e-4}"
    check_value AUG "$aug" A B; check_value LR "$lr" 1e-4 3e-4 1e-3 3e-3; check_value WD "$wd" 1e-4 0.01 0.05 0.1
    [[ "$DRY_RUN" == "1" ]] || require_json "$OUT/stage3.json" \
      "d.get('extension_triggered') and d['args']['aug'] == '$aug' and close(d['args']['lr'], '$lr') and close(d['args']['wd'], '$wd')" \
      "stage3.json does not show the extension rule triggered for AUG=$aug, LR=$lr, WD=$wd; not running 140 epochs"
    pilot_run moe "$aug" "$lr" "$wd" 140 0
    summarize stage3.json --stage epochs --aug "$aug" --lr "$lr" --wd "$wd"
    ;;
  wd)
    aug="${2:-}"; lr="${3:-}"; ep="${4:-}"
    check_value AUG "$aug" A B; check_value LR "$lr" 1e-4 3e-4 1e-3 3e-3; check_value EP "$ep" 50 70 100 140
    for m in single moe; do for wd in 1e-4 0.01 0.05 0.1; do pilot_run "$m" "$aug" "$lr" "$wd" "$ep" 0; done; done
    summarize stage4.json --stage wd --aug "$aug" --lr "$lr" --epochs "$ep"
    ;;
  lr-recheck)
    aug="${2:-}"; lr="${3:-}"; ep="${4:-}"; wd="${5:-}"
    check_value AUG "$aug" A B; check_value LR "$lr" 1e-3 3e-3; check_value EP "$ep" 50 70 100 140
    check_value WD "$wd" 0.01 0.05 0.1
    case "$lr" in 1e-3) lower=3e-4 ;; 3e-3) lower=1e-3 ;; esac
    [[ "$DRY_RUN" == "1" ]] || require_json "$OUT/stage4.json" \
      "close(d['args']['lr'], '$lr') and close(d.get('final'), '$wd') and d['args']['aug'] == '$aug' and d['args']['epochs'] == $ep" \
      "The lr recheck runs only when stage 4 (AUG=$aug, LR=$lr, EP=$ep) selected wd=$wd"
    for m in single moe; do pilot_run "$m" "$aug" "$lower" "$wd" "$ep" 0; done
    summarize stage4_lr.json --stage lr-recheck --aug "$aug" --lr "$lr" --epochs "$ep" --wd "$wd"
    ;;
  repro)
    # Config identical to stage 0 seed 0; directory name also identical. After finishing, compare bit-for-bit with the same-named run in runs/pilot/ (read-only),
    # exit non-zero when not passed or when the old run is not found, so "repro && wd ..." can be used to run stage 4 right after
    [[ "$DRY_RUN" == "1" ]] || mkdir -p "$OUT/logs"
    pilot_run moe B 1e-3 1e-4 70 0
    [[ "$DRY_RUN" == "1" ]] && exit 0
    name="moe-fixed_R0_lam${LAM0}_sig${SIG0}_augB_lr1e-3_wd1e-4_e70_s0"
    compare_runs "$OUT/$name" "../runs/pilot/$name" "$OUT/compare.json" \
      || die "Repro comparison did not pass; stop and discuss before running stage 4 (see $OUT/compare.json)"
    ;;
  router)
    # Stage 5 (non-inferiority check of R4 vs R0, see Pilot_Plan.md). Settings fixed to the selections of stages 1 to 4, which is also the stage 0 config,
    # so R0 seeds 1/2 should be bit-identical to the same-named stage 0 runs in runs/pilot/; stop when not passed, do not run R4
    [[ "$DRY_RUN" == "1" ]] || mkdir -p "$OUT/logs"
    for s in 1 2; do
      pilot_run moe B 1e-3 1e-4 70 "$s"
      [[ "$DRY_RUN" == "1" ]] && continue
      name="moe-fixed_R0_lam${LAM0}_sig${SIG0}_augB_lr1e-3_wd1e-4_e70_s${s}"
      compare_runs "$OUT/$name" "../runs/pilot/$name" "$OUT/compare_R0_s${s}.json" \
        || die "R0 seed $s does not reproduce stage 0; stop and discuss before running R4 (see $OUT/compare_R0_s${s}.json)"
    done
    ROUTER=R4
    for s in 0 1 2; do pilot_run moe B 1e-3 1e-4 70 "$s"; done
    [[ "$DRY_RUN" == "1" ]] && exit 0
    python summarize_router.py --router-dir "$OUT" --repro-dir ../runs/repro --threshold ../runs/pilot/threshold.json \
      --tau0 "$TAU0" --lambda0 "$LAM0" --sigma0 "$SIG0" --out "$OUT/stage5.json"
    ;;
  fixed6)
    # Stage 6 A (see Pilot_Plan.md): settings fixed to the selections of stages 1 to 5. First run seed 0 of the 6 single-parameter candidates, then
    # summarize_stage6.py lists the runs needing follow-up by the rules (seeds 1/2, combined configs); after running them summarize again, until no more follow-up runs are needed
    ROUTER=R4
    [[ "$DRY_RUN" == "1" ]] || mkdir -p "$OUT/logs"
    for c in "0.5 0.2 0.5" "4.5 0.2 0.5" "1.5 0.1 0.5" "1.5 0.4 0.5" "1.5 0.2 0.25" "1.5 0.2 1.0"; do
      stage6_run $c 0
    done
    [[ "$DRY_RUN" == "1" ]] && exit 0
    for round in 1 2 3; do
      python summarize_stage6.py --part A --stage6-dir "$OUT" --router-dir ../runs/router --threshold ../runs/pilot/threshold.json \
        --lambda0 "$LAM0" --sigma0 "$SIG0" --out "$OUT/stage6a.json"
      todo=$(python -c 'import json, sys; [print(r["lambda"], r["sigma"], r["tau"], r["seed"]) for r in json.load(open(sys.argv[1]))["to_run"]]' "$OUT/stage6a.json")
      [[ -z "$todo" ]] && break
      [[ "$round" == 3 ]] && die "Runs still required after 3 rounds; inspect $OUT/stage6a.json"
      while read -r l s t k; do stage6_run "$l" "$s" "$t" "$k"; done <<< "$todo"
    done
    ;;
  fixed6-ext)
    # λ 4.5 extension of stage 6 A (rule exception of 2026-10-28, see Pilot_Plan.md): follow-up runs of λ 4.5 seeds 1, 2, verdict by the pairing rule; only this step is extended
    ROUTER=R4
    [[ "$DRY_RUN" == "1" ]] || mkdir -p "$OUT/logs"
    for s in 1 2; do stage6_run 4.5 0.2 0.5 "$s"; done
    [[ "$DRY_RUN" == "1" ]] && exit 0
    python summarize_stage6.py --part A --extend lambda=4.5 --stage6-dir "$OUT" --router-dir ../runs/router \
      --threshold ../runs/pilot/threshold.json --lambda0 "$LAM0" --sigma0 "$SIG0" --out "$OUT/stage6a.json"
    ;;
  nowarmup-check)
    # Stage 6 C step 1 (see Pilot_Plan.md): center point without aux warmup (R4, λ 1.5, σ 0.2, τ 0.5) seeds 0/1/2, paired by seed with the center point
    # with warmup in runs/router/; summarized automatically after running, giving whether to adopt "no warmup" and whether layer 2 is triggered
    ROUTER=R4; AUX_WARMUP=0
    [[ "$DRY_RUN" == "1" ]] || mkdir -p "$OUT/logs"
    for s in 0 1 2; do stage6_run 1.5 0.2 0.5 "$s"; done
    [[ "$DRY_RUN" == "1" ]] && exit 0
    python summarize_stage6.py --part C --stage6-dir "$OUT" --ref-dir ../runs/router --aux-warmup-frac 0 \
      --threshold ../runs/pilot/threshold.json --lambda0 "$LAM0" --sigma0 "$SIG0" --out "$OUT/stage6c.json"
    ;;
  fixed6nw)
    # Stage 6 C step 2: after adopting "no warmup", rerun stage 6 A (same rules, including the λ 4.5 extension, i.e. λ 4.5 directly gets seeds 1/2);
    # center point taken from the 3 runs of nowarmup-check (also in runs/stage6c/)
    [[ -f "$OUT/stage6c.json" ]] || die "$OUT/stage6c.json not found; run nowarmup-check first"
    python -c 'import json, sys; sys.exit(0 if json.load(open(sys.argv[1]))["decision"] == "adopt no-warmup" else 1)' "$OUT/stage6c.json" \
      || die "Stage 6 C did not adopt the no-warmup setting (see $OUT/stage6c.json); fixed6nw is not applicable"
    ROUTER=R4; AUX_WARMUP=0
    for c in "0.5 0.2 0.5" "4.5 0.2 0.5" "1.5 0.1 0.5" "1.5 0.4 0.5" "1.5 0.2 0.25" "1.5 0.2 1.0"; do
      stage6_run $c 0
    done
    [[ "$DRY_RUN" == "1" ]] && exit 0
    for round in 1 2 3; do
      python summarize_stage6.py --part A --extend lambda=4.5 --stage6-dir "$OUT" --router-dir "$OUT" --aux-warmup-frac 0 \
        --threshold ../runs/pilot/threshold.json --lambda0 "$LAM0" --sigma0 "$SIG0" --out "$OUT/stage6a_nw.json"
      todo=$(python -c 'import json, sys; [print(r["lambda"], r["sigma"], r["tau"], r["seed"]) for r in json.load(open(sys.argv[1]))["to_run"]]' "$OUT/stage6a_nw.json")
      [[ -z "$todo" ]] && break
      [[ "$round" == 3 ]] && die "Runs still required after 3 rounds; inspect $OUT/stage6a_nw.json"
      while read -r l s t k; do stage6_run "$l" "$s" "$t" "$k"; done <<< "$todo"
    done
    ;;
  study3b)
    # Pilot_Study.md stage 3b: R0, no aux warmup, starting settings, seeds 0/1/2; paired with the R4 center point in runs/stage6c/
    ROUTER=R0; AUX_WARMUP=0
    [[ "$DRY_RUN" == "1" ]] || mkdir -p "$OUT/logs"
    for s in 0 1 2; do stage6_run 1.5 0.2 0.5 "$s"; done
    [[ "$DRY_RUN" == "1" ]] && exit 0
    study_summary 3b "$OUT/study3b.json"
    [[ "$(study_field "$OUT/study3b.json" decision)" == "confirm R4" ]] || die "Stage 3b did not confirm R4; stop and discuss (see $OUT/study3b.json)"
    ;;
  study4)
    # Pilot_Study.md stage 4: augmentation -> lr -> epoch step by step (MoE, R4, no aux warmup, seed 0); single reuses runs/pilot/.
    # The selection of the epoch step has changed from 70 to 50, so a rerun stops at this step; for the wd step see study4wd.
    # Each step: run candidates, summarize, add seeds 1/2 and summarize again when confirmation is needed; stop when the selection changes, do not go to the next step
    ROUTER=R4; AUX_WARMUP=0
    [[ "$DRY_RUN" == "1" ]] || mkdir -p "$OUT/logs"
    if [[ "$DRY_RUN" == "1" ]]; then  # only list the seed 0 candidates of each step (follow-up runs for confirmation are decided by the summary results)
      pilot_run moe A 1e-3 1e-4 70 0
      for lr in 1e-4 3e-4 3e-3; do pilot_run moe B "$lr" 1e-4 70 0; done
      for ep in 50 100; do pilot_run moe B 1e-3 1e-4 "$ep" 0; done
      exit 0
    fi
    for step in aug lr epochs; do
      js="$OUT/study4_${step}.json"
      for round in 1 2 3; do
        study_summary 4 "$js" --step "$step"
        todo=$(study_todo "$js" aug lr wd epochs seed)
        [[ -z "$todo" ]] && break
        [[ "$round" == 3 ]] && die "Runs still required after 3 rounds; inspect $js"
        while read -r a l w e k; do pilot_run moe "$a" "$l" "$w" "$e" "$k"; done <<< "$todo"
      done
      d=$(study_field "$js" decision)
      [[ "$d" == keep ]] || die "Stage 4 step $step: decision '$d'; stop and discuss before later steps (see $js)"
    done
    ;;
  study4wd)
    # Pilot_Study.md stage 4.4: wd at 50 epoch (seed 0). The 4 single runs are written to runs/pilot/ (summarize_study.py reads single from there);
    # MoE wd 1e-4 reuses the 50 epoch run of the epoch step. Add seeds 1/2 when confirmation is needed; stop when the selection changes
    ROUTER=R4; AUX_WARMUP=0
    if [[ "$DRY_RUN" != "1" ]]; then mkdir -p "$OUT/logs" ../runs/pilot/logs; fi
    for wd in 1e-4 0.01 0.05 0.1; do OUT=../runs/pilot pilot_run single B 1e-3 "$wd" 50 0; done
    for wd in 0.01 0.05 0.1; do pilot_run moe B 1e-3 "$wd" 50 0; done
    [[ "$DRY_RUN" == "1" ]] && exit 0
    js="$OUT/study4_wd.json"
    for round in 1 2 3; do
      study_summary 4 "$js" --step wd
      todo=$(study_todo "$js" aug lr wd epochs seed)
      [[ -z "$todo" ]] && break
      [[ "$round" == 3 ]] && die "Runs still required after 3 rounds; inspect $js"
      while read -r a l w e k; do pilot_run moe "$a" "$l" "$w" "$e" "$k"; done <<< "$todo"
    done
    d=$(study_field "$js" decision)
    [[ "$d" == keep ]] || die "Stage 4 step wd: decision '$d'; stop and discuss before later stages (see $js)"
    ;;
  study4single)
    # Pilot_Study.md stage 4.4 single confirmation: seeds 1/2 of wd 0.01 (selected by seed 0) and 1e-4, 50 epoch, paired together with seed 0; does not affect MoE
    [[ "$DRY_RUN" == "1" ]] || mkdir -p "$OUT/logs"
    for wd in 1e-4 0.01; do for s in 1 2; do pilot_run single B 1e-3 "$wd" 50 "$s"; done; done
    [[ "$DRY_RUN" == "1" ]] && exit 0
    study_summary 4s "$OUT/study4_wd_single.json"
    d=$(study_field "$OUT/study4_wd_single.json" decision)
    [[ "$d" == keep ]] || die "Stage 4 single-model wd: decision '$d'; stop and discuss (see $OUT/study4_wd_single.json)"
    ;;
  study6)
    # Pilot_Study.md stage 6 (50 epoch): 3 seeds each for the 6 candidates of λ / σ / τ, final values and acceptable ranges set by the pairing rule; follow-up runs of combined configs when needed.
    # The center point is the 3 runs of 50 epoch from the epoch step
    ROUTER=R4; AUX_WARMUP=0; STAGE6_EP=50
    [[ "$DRY_RUN" == "1" ]] || mkdir -p "$OUT/logs"
    for c in "0.5 0.2 0.5" "4.5 0.2 0.5" "1.5 0.1 0.5" "1.5 0.4 0.5" "1.5 0.2 0.25" "1.5 0.2 1.0"; do
      for s in 0 1 2; do stage6_run $c "$s"; done
    done
    [[ "$DRY_RUN" == "1" ]] && exit 0
    for round in 1 2 3; do
      study_summary 6 "$OUT/study6.json"
      todo=$(study_todo "$OUT/study6.json" lambda sigma tau seed)
      [[ -z "$todo" ]] && break
      [[ "$round" == 3 ]] && die "Runs still required after 3 rounds; inspect $OUT/study6.json"
      while read -r l s t k; do stage6_run "$l" "$s" "$t" "$k"; done <<< "$todo"
    done
    [[ "$(study_field "$OUT/study6.json" changed)" == False ]] \
      || die "Stage 6 final values differ from the stage-1 provisional values; stop and re-check stages 2 to 5 (see $OUT/study6.json)"
    ;;
  study6b)
    # Pilot_Study.md stage 6b (recheck after σ0 = 0.4): summarize_study.py gives the pending runs of each step in order; stop when not passed, later items are not run.
    # DRY_RUN lists the 10 runs for the case where all pass (70 epoch seed 0 already exists, shown as skipped); the lr confirmation follow-up runs are not included
    ROUTER=R4; AUX_WARMUP=0
    if [[ "$DRY_RUN" == "1" ]]; then
      for k in 0 1 2; do study6b_run 0.8 1e-3 1e-4 50 "$k"; done
      for k in 0 1 2; do study6b_run 0.4 1e-3 1e-4 70 "$k"; done
      for k in 0 1 2; do study6b_run 0.4 1e-3 0.05 50 "$k"; done
      for l in 3e-4 3e-3; do study6b_run 0.4 "$l" 1e-4 50 0; done
      exit 0
    fi
    mkdir -p "$OUT/logs"
    js="$OUT/study6b.json"
    for round in 1 2 3 4 5 6 7; do
      study_summary 6b "$js"
      todo=$(study_todo "$js" sigma lr wd epochs seed)
      [[ -z "$todo" ]] && break
      [[ "$round" == 7 ]] && die "Runs still required after 7 rounds; inspect $js"
      while read -r g l w e k; do study6b_run "$g" "$l" "$w" "$e" "$k"; done <<< "$todo"
    done
    d=$(study_field "$js" decision)
    [[ "$d" == keep ]] || die "Stage 6b: decision '$d'; stop and discuss (see $js)"
    ;;
  study8)
    # Pilot_Study.md stage 8 trial run: round ROUND (default 1), fuzzy (fuzzy_v2.json, current content recorded in config.json) and schedule, seed 0 each, 50 epoch;
    # directory name suffixed with _r<ROUND>. After running, summarize_study.py --part 8 gives the verdict by control behavior (validation accuracy not considered); after adjusting, increase ROUND by 1 and run again
    # From round 2 on (10.3): config is fuzzy_v2_r<ROUND>.json, only fuzzy seeds 0/1/2 are run (schedule unchanged, not rerun), verdict includes the 3-seed pairing with fixed σ0 0.4
    ROUTER=R4; AUX_WARMUP=0; TAU0=0.5; TAG="_r${ROUND:-1}"
    if [[ "${ROUND:-1}" -ge 2 ]]; then FUZZY_CONFIG="fuzzy_v2_r${ROUND}.json"; fi
    [[ -f "$FUZZY_CONFIG" ]] || die "$FUZZY_CONFIG not found"
    [[ "$DRY_RUN" == "1" ]] || mkdir -p "$OUT/logs"
    if [[ "${ROUND:-1}" -ge 2 ]]; then
      CONTROLLER=fuzzy
      for s in 0 1 2; do pilot_run moe B 1e-3 1e-4 50 "$s"; done
    else
      for c in fuzzy schedule; do CONTROLLER=$c; pilot_run moe B 1e-3 1e-4 50 0; done
    fi
    CONTROLLER=fixed
    [[ "$DRY_RUN" == "1" ]] && exit 0
    study_summary 8 "$OUT/study8_r${ROUND:-1}.json" --round "${ROUND:-1}"
    ;;
  study9a)
    # Pilot_Study.md 10.4 stage 9a: fixed, seed 0, 50 epoch; each run changes only one parameter (one step outward from the acceptable range on the stage 6 grid), the rest at the baseline λ0 1.5, σ0 0.4, τ0 0.5.
    # After running, summarize_study.py --part 9a gives the "bad" verdict and selects the 9b candidates (at most 3) in the pre-written order
    [[ "$LAM0" == 1.5 && "$SIG0" == 0.4 ]] || die "study9a expects the baseline LAM0=1.5 SIG0=0.4 (the candidates are set inside)"
    ROUTER=R4; AUX_WARMUP=0; TAG=""; CONTROLLER=fixed
    [[ "$DRY_RUN" == "1" ]] || mkdir -p "$OUT/logs"
    for spec in "0.17 0.4 0.5" "13.5 0.4 0.5" "1.5 0 0.5" "1.5 1.6 0.5" "1.5 0.4 0.125" "1.5 0.4 2.0"; do
      read -r LAM0 SIG0 TAU0 <<< "$spec"
      pilot_run moe B 1e-3 1e-4 50 0
    done
    LAM0=1.5; SIG0=0.4; TAU0=0.5
    [[ "$DRY_RUN" == "1" ]] && exit 0
    study_summary 9a "$OUT/study9a.json" --stage9-dir "$OUT"
    ;;
  study9a_ext)
    # Pilot_Study.md 10.4 stage 9a extension (A.1 item 18): single parameters λ0 0, 40.5, σ0 3.2 and three combinations; fixed, seed 0, 50 epoch.
    # After running, summarize_study.py --part 9a gives the verdict over 12 candidates (batch 1 plus the extension) and selects the 9b candidates (at most 3)
    [[ "$LAM0" == 1.5 && "$SIG0" == 0.4 ]] || die "study9a_ext expects the baseline LAM0=1.5 SIG0=0.4 (the candidates are set inside)"
    ROUTER=R4; AUX_WARMUP=0; TAG=""; CONTROLLER=fixed
    [[ "$DRY_RUN" == "1" ]] || mkdir -p "$OUT/logs"
    for spec in "0 0.4 0.5" "40.5 0.4 0.5" "1.5 3.2 0.5" "0 0 0.125" "40.5 3.2 2.0" "0 3.2 2.0"; do
      read -r LAM0 SIG0 TAU0 <<< "$spec"
      pilot_run moe B 1e-3 1e-4 50 0
    done
    LAM0=1.5; SIG0=0.4; TAU0=0.5
    [[ "$DRY_RUN" == "1" ]] && exit 0
    study_summary 9a "$OUT/study9a_ext.json" --stage9-dir "$OUT"
    ;;
  study9b)
    # Pilot_Study.md 10.4 stage 9b: bad starting points selected by study9a_ext.json (at most 3), 50 epoch. Step 1 fixed seeds 1, 2 (seed 0 reuses 9a);
    # after summarize_study.py --part 9b confirms "bad", step 2 runs fuzzy seeds 0/1/2 only for the confirmed starting points. The pending runs of each step come from to_run of the summary.
    # On 2026-11-06 the round 3 controller was adopted (A.1 item 21): fuzzy uses fuzzy_v2_r3.json, directory name suffixed with _r3 (the 9b fuzzy directories of the round 2 version have no suffix and are not reused),
    # summary written to study9b_r3.json; fixed runs do not depend on the controller and are reused as usual
    [[ "$LAM0" == 1.5 && "$SIG0" == 0.4 ]] || die "study9b expects the baseline LAM0=1.5 SIG0=0.4 (the starts are read from study9a_ext.json)"
    [[ -f "$OUT/study9a_ext.json" ]] || die "$OUT/study9a_ext.json not found; run study9a_ext first"
    ROUTER=R4; AUX_WARMUP=0; TAG=""; FUZZY_CONFIG=fuzzy_v2_r3.json; fz_tag="_r3"
    [[ -f "$FUZZY_CONFIG" ]] || die "$FUZZY_CONFIG not found"
    [[ "$DRY_RUN" == "1" ]] || mkdir -p "$OUT/logs"
    sum9b="$OUT/study9b${fz_tag}.json"
    [[ "$DRY_RUN" == "1" ]] && sum9b=$(mktemp)  # dry run only lists step 1, does not write runs/
    for round in 1 2 3; do
      study_summary 9b "$sum9b" --stage9-dir "$OUT" --study9a "$OUT/study9a_ext.json" --fuzzy-config "$FUZZY_CONFIG"
      todo=$(study_todo "$sum9b" controller lambda0 sigma0 tau0 seed)
      [[ -z "$todo" ]] && break
      [[ "$round" == 3 ]] && die "Runs still required after 3 rounds; inspect $sum9b"
      while read -r c l g t k; do
        CONTROLLER=$c; LAM0=$l; SIG0=$g; TAU0=$t; TAG=""
        [[ "$c" == fuzzy ]] && TAG=$fz_tag
        pilot_run moe B 1e-3 1e-4 50 "$k" < /dev/null
      done <<< "$todo"
      CONTROLLER=fixed; LAM0=1.5; SIG0=0.4; TAU0=0.5; TAG=""
      [[ "$DRY_RUN" == "1" ]] && { rm -f "$sum9b"; exit 0; }
    done
    echo "Stage 9b: decision $(study_field "$sum9b" decision) (see $sum9b)"
    ;;
  formal)
    # Pilot_Study.md 11.2 formal experiment, one seed per batch (BATCH indices 1/2/3 are seeds 3/4/5), method order fixed.
    # A: single (wd 0.01), fixed (1.5, 0.4, 0.5), schedule, auxfree (γ 1e-3), fuzzy (fuzzy_v2_r3.json); B: 3 bad starting points × fixed, fuzzy;
    # C: same as A, diagnostics off and no validation-set evaluation. At the end of each batch final_check checks completeness, commit hash, uncommitted changes; C is compared bit-for-bit with the matching A on the accuracy columns
    [[ "$LAM0" == 1.5 && "$SIG0" == 0.4 ]] || die "formal expects the baseline LAM0=1.5 SIG0=0.4 (the settings are set inside)"
    seed=$(( ${BATCH:1:1} + 2 ))
    [[ -f fuzzy_v2_r3.json ]] || die "fuzzy_v2_r3.json not found"
    if [[ "${BATCH:0:1}" == C ]]; then DIAG=0; fi
    [[ "$DRY_RUN" == "1" ]] || mkdir -p "$OUT/logs"
    kind=${BATCH:0:1}; [[ "$kind" == C ]] && kind=A  # the timing batch has the same composition as A
    formal_batch "$kind" "$seed"
    DIAG=1
    [[ "$DRY_RUN" == "1" ]] && exit 0
    final_check "$BATCH" "$OUT" ../runs/final "${BATCH_RUNS[@]}"
    ;;
  final-test)
    # Pilot_Study.md 11.3: after all 33 runs of A, B are done, evaluate best.pth and last.pth on test.csv (--final), output <run>/test_eval_{best,last}/.
    # Each checkpoint is evaluated only once: skip those that already have metrics.json, stop with an error when the directory exists but is incomplete
    [[ "$LAM0" == 1.5 && "$SIG0" == 0.4 ]] || die "final-test expects the baseline LAM0=1.5 SIG0=0.4"
    formal_all A B
    if [[ "$DRY_RUN" == "1" ]]; then
      for n in "${BATCH_RUNS[@]}"; do for ck in best last; do echo "[Dry run] test eval $n/$ck.pth -> $n/test_eval_$ck"; done; done
      exit 0
    fi
    final_check AB "$OUT" "$OUT" "${BATCH_RUNS[@]}" || die "Batches A and B are not complete or not consistent (see above); no test evaluation was run"
    for n in "${BATCH_RUNS[@]}"; do
      for ck in best last; do
        ev="$OUT/$n/test_eval_$ck"
        if [[ -f "$ev/metrics.json" ]]; then echo "[Skip] $n/test_eval_$ck is already complete"; continue; fi
        [[ -e "$ev" ]] && die "$ev exists without metrics.json and may have been interrupted; inspect and delete it, then rerun"
        echo "[Test] $n/$ck.pth  ($(date '+%F %T'))"
        python evaluate.py --checkpoint "$OUT/$n/$ck.pth" --root "$ROOT" --subset-dir "$SUBSET" --list test.csv --final \
          --out-dir "$ev" > "$OUT/logs/$n.test_$ck.log" 2>&1 || die "$n/$ck.pth test evaluation failed; see $OUT/logs/$n.test_$ck.log"
      done
    done
    echo "Test evaluation complete: ${#BATCH_RUNS[@]} runs x 2 checkpoints"
    ;;
  final-experts)
    # Pilot_Study.md 11.3 (D3) and analysis/TODO.md "Expert function analysis after the formal experiments": best.pth of the 12 MoE runs of batch A on val.csv,
    # run evaluate.py --dense-experts (output <run>/val_dense_best/), then analyze_experts.py summarizes into runs/final/experts.json. Descriptive only
    [[ "$LAM0" == 1.5 && "$SIG0" == 0.4 ]] || die "final-experts expects the baseline LAM0=1.5 SIG0=0.4"
    formal_all A
    moe_runs=()
    for n in "${BATCH_RUNS[@]}"; do [[ "$n" == moe-* ]] && moe_runs+=("$n"); done
    if [[ "$DRY_RUN" == "1" ]]; then
      for n in "${moe_runs[@]}"; do echo "[Dry run] dense experts $n/best.pth -> $n/val_dense_best"; done
      exit 0
    fi
    final_check A "$OUT" "$OUT" "${moe_runs[@]}" || die "Batch A is not complete or not consistent (see above); no expert analysis was run"
    for n in "${moe_runs[@]}"; do
      ev="$OUT/$n/val_dense_best"
      if [[ -f "$ev/dense_experts.npz" ]]; then echo "[Skip] $n/val_dense_best is already complete"; continue; fi
      [[ -e "$ev" ]] && die "$ev exists without dense_experts.npz and may have been interrupted; inspect and delete it, then rerun"
      echo "[Dense] $n/best.pth  ($(date '+%F %T'))"
      python evaluate.py --checkpoint "$OUT/$n/best.pth" --root "$ROOT" --subset-dir "$SUBSET" --list val.csv --dense-experts \
        --out-dir "$ev" > "$OUT/logs/$n.dense.log" 2>&1 || die "$n dense evaluation failed; see $OUT/logs/$n.dense.log"
    done
    python analyze_experts.py --runs-dir "$OUT" --names "${moe_runs[@]}" --subset-dir "$SUBSET" --out "$OUT/experts.json"
    ;;
  study8-seeds)
    # Pilot_Study.md after stage 8 is frozen: fuzzy (fuzzy_v2.json) seeds 1, 2, 50 epoch, directory name without the _r suffix (export_study.py names them this way).
    # The verdict (3-seed pairing with fixed σ0 0.4, 10.3) is made on the analysis laptop, not automatically here
    ROUTER=R4; AUX_WARMUP=0; TAU0=0.5; TAG=""
    [[ -f "$FUZZY_CONFIG" ]] || die "$FUZZY_CONFIG not found"
    [[ "$DRY_RUN" == "1" ]] || mkdir -p "$OUT/logs"
    CONTROLLER=fuzzy
    for s in 1 2; do pilot_run moe B 1e-3 1e-4 50 "$s"; done
    CONTROLLER=fixed
    ;;
  study7)
    # Pilot_Study.md stage 7 (50 epoch): auxfree γ ∈ {1e-4, 1e-3, 1e-2}, seed 0; when it deviates from 1e-3, add seeds 1/2 of the candidate γ and of 1e-3 for confirmation
    ROUTER=R4; CONTROLLER=auxfree; LAM0=0; SIG0=0; TAU0=0.5
    [[ "$DRY_RUN" == "1" ]] || mkdir -p "$OUT/logs"
    for g in 1e-3 1e-4 0.01; do GAMMA=$g; pilot_run moe B 1e-3 1e-4 50 0; done
    GAMMA=1e-3
    [[ "$DRY_RUN" == "1" ]] && exit 0
    for round in 1 2 3; do
      study_summary 7 "$OUT/study7.json"
      todo=$(study_todo "$OUT/study7.json" gamma seed)
      [[ -z "$todo" ]] && break
      [[ "$round" == 3 ]] && die "Runs still required after 3 rounds; inspect $OUT/study7.json"
      while read -r g k; do GAMMA=$g; pilot_run moe B 1e-3 1e-4 50 "$k"; done <<< "$todo"
      GAMMA=1e-3
    done
    d=$(study_field "$OUT/study7.json" decision)
    [[ "$d" == keep ]] || die "Stage 7: decision '$d'; stop and discuss (see $OUT/study7.json)"
    ;;
  auxfree6)
    # Stage 6 B: faithful reproduction of DeepSeek-V3 auxfree (σ=0, λ=0; --auxfree-lambda defaults to 0), τ0 taken from the final selection of A
    [[ -f "$OUT/stage6a.json" ]] || die "$OUT/stage6a.json not found; run fixed6 first"
    tau=$(python -c 'import json, sys; d = json.load(open(sys.argv[1])); assert not d["to_run"] and d["final"], "stage 6 A has no final selection"; print(d["final"]["tau"])' "$OUT/stage6a.json") \
      || die "Stage 6 A has no final selection yet (see $OUT/stage6a.json)"
    ROUTER=R4; CONTROLLER=auxfree
    [[ "$DRY_RUN" == "1" ]] || mkdir -p "$OUT/logs"
    LAM0_A=$LAM0; SIG0_A=$SIG0
    LAM0=0; SIG0=0; TAU0=$tau
    pilot_run moe B 1e-3 1e-4 70 0
    [[ "$DRY_RUN" == "1" ]] && exit 0
    python summarize_stage6.py --part B --stage6-dir "$OUT" --router-dir ../runs/router --threshold ../runs/pilot/threshold.json \
      --lambda0 "$LAM0_A" --sigma0 "$SIG0_A" --out "$OUT/stage6b.json"
    ;;
  *)
    die "Unknown stage: $stage; choose base | aug | lr | epochs | epochs-ext | wd | lr-recheck | repro | router | fixed6 | fixed6-ext | auxfree6 | nowarmup-check | fixed6nw | study3b | study4 | study4wd | study4single | study6 | study6b | study7 | study8 | study8-seeds | study9a | study9a_ext | study9b | formal | final-test | final-experts"
    ;;
esac
