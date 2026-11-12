#!/usr/bin/env bash
# setup_board.sh: set up the CPU inference environment (venv) on the board; first check the installed packages and versions, install only missing ones or ones with the wrong version.
#
#   bash deploy/setup_board.sh            # check and install
#   bash deploy/setup_board.sh --check    # check only, do not install
#   VENV=~/edgeai-venv bash deploy/setup_board.sh
#
# Version requirements: all four boards use torch 2.11.0, torchvision 0.26.0, timm 1.0.22, numpy 2.2.6, pillow 12.0.0 (Python >= 3.10, glibc >= 2.28).
# Not using torch 2.10.0 from the experiment machine's runs/env_lock.json: in its aarch64 wheel, the mimalloc linked by libc10.so contains ARMv8.1 LSE atomic instructions,
# so on the ARMv8.0 Cortex-A72 (Raspberry Pi 4B) import torch alone fails with Illegal instruction (pytorch/pytorch#174344);
# the fix is in 2.11.0; on 2026-11-09 import and an EfficientNet-B0 forward pass were tested and passed on the RPi 4B. The other three boards (A76, Carmel, A78AE) are all ARMv8.2.
# Version differences between the boards and the experiment machine are checked by bench_latency.py against the experiment machine's predictions.csv (agreement rate >= 0.999).
# With Python < 3.10 (e.g. 3.8 on JetPack 5 / Ubuntu 20.04) or glibc < 2.28, exit with an error, no more downgrading: downgrading would make the versions differ across boards and leak into cross-board comparisons.
# torch is installed from PyTorch's CPU index (aarch64 CPU wheels; since 2.11 the default aarch64 wheels on PyPI include CUDA); falls back to PyPI on failure.
set -euo pipefail

CHECK_ONLY=0
[[ "${1:-}" == "--check" ]] && CHECK_ONLY=1
VENV="${VENV:-$HOME/edgeai-venv}"
PY_SYS="${PYTHON:-python3}"

command -v "$PY_SYS" >/dev/null || { echo "[Error] $PY_SYS not found"; exit 1; }
PYV=$("$PY_SYS" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
ARCH=$(uname -m)
echo "== system python $PYV ($ARCH), venv $VENV =="

case "$PYV" in
  3.1[0-4]) ;;
  *) echo "[Error] Python $PYV: need 3.10 ~ 3.14 (torch 2.11.0 has no wheel for it)."
     echo "        Ubuntu 20.04 / JetPack 5: sudo apt install python3.10 python3.10-venv (deadsnakes PPA if missing), then"
     echo "        PYTHON=python3.10 bash deploy/setup_board.sh"
     exit 1 ;;
esac
GLIBC=$(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+$' || true)
if [[ -n "$GLIBC" ]] && ! awk -v g="$GLIBC" 'BEGIN { split(g, a, "."); exit !(a[1] > 2 || (a[1] == 2 && a[2] >= 28)) }'; then
  echo "[Error] glibc $GLIBC < 2.28: the torch 2.11.0 aarch64 wheel (manylinux_2_28) cannot run here (JetPack 4 / Ubuntu 18.04?)"; exit 1
fi
TORCH=2.11.0; TV=0.26.0; NUMPY=2.2.6; PILLOW=12.0.0; TIMM=1.0.22
echo "== glibc ${GLIBC:-unknown}; target torch $TORCH torchvision $TV timm $TIMM numpy $NUMPY pillow $PILLOW =="

if [[ ! -x "$VENV/bin/python" ]]; then
  if [[ "$CHECK_ONLY" == 1 ]]; then echo "[Missing] venv $VENV does not exist"; exit 2; fi
  "$PY_SYS" -m venv "$VENV" || { echo "[Error] venv failed; on Debian/Ubuntu: sudo apt install python3-venv"; exit 1; }
fi
PIP="$VENV/bin/pip"
PYB="$VENV/bin/python"

# current version -> list of packages to install
need=()
check_pkg() {  # check_pkg <import name> <pip name> <required version>
  local have
  have=$("$PYB" - "$1" <<'PY' 2>/dev/null || true
import sys, importlib
try:
    m = importlib.import_module(sys.argv[1]); print(getattr(m, "__version__", "?"))
except Exception:
    print("")
PY
)
  have=${have%%+*}   # strip suffixes such as +cpu
  if [[ "$have" == "$3" ]]; then echo "[OK]      $2 $have"; else echo "[Install] $2: have '${have:-none}', need $3"; need+=("$2==$3"); fi
}
check_pkg numpy numpy "$NUMPY"
check_pkg PIL pillow "$PILLOW"
check_pkg torch torch "$TORCH"
check_pkg torchvision torchvision "$TV"
check_pkg timm timm "$TIMM"

if [[ ${#need[@]} -eq 0 ]]; then
  echo "== all packages present at the required versions =="
else
  if [[ "$CHECK_ONLY" == 1 ]]; then echo "== missing/mismatched: ${need[*]} (check only, nothing installed) =="; exit 2; fi
  "$PIP" install --upgrade pip >/dev/null
  torch_pkgs=(); other=()
  for p in "${need[@]}"; do case "$p" in torch==*|torchvision==*) torch_pkgs+=("$p");; *) other+=("$p");; esac; done
  if [[ ${#torch_pkgs[@]} -gt 0 ]]; then
    echo "== installing ${torch_pkgs[*]} from the PyTorch CPU index =="
    "$PIP" install "${torch_pkgs[@]}" --index-url https://download.pytorch.org/whl/cpu \
      || { echo "[Warn] CPU index failed, trying PyPI"; "$PIP" install "${torch_pkgs[@]}"; }
  fi
  if [[ ${#other[@]} -gt 0 ]]; then echo "== installing ${other[*]} =="; "$PIP" install "${other[@]}"; fi
fi

echo "== verification =="
"$PYB" - <<'PY'
import torch, torchvision, timm, numpy, PIL
print(f"torch {torch.__version__}  torchvision {torchvision.__version__}  timm {timm.__version__}  numpy {numpy.__version__}  pillow {PIL.__version__}")
print(f"threads {torch.get_num_threads()}  cuda {torch.cuda.is_available()}")
x = torch.randn(1, 3, 224, 224); m = timm.create_model("efficientnet_b0", pretrained=False).eval()
with torch.no_grad(): print("efficientnet_b0 forward ok:", tuple(m(x).shape))
PY
HERE=$(cd "$(dirname "$0")" && pwd)
if [[ "$CHECK_ONLY" == 0 ]]; then
  BOARD="${BOARD:-$(hostname)}"
  "$PYB" "$HERE/board_env.py" --out "$HERE/results/$BOARD/env.json" >/dev/null && echo "env saved to deploy/results/$BOARD/env.json (BOARD=$BOARD)"
fi
echo "== done. activate with: source $VENV/bin/activate =="
