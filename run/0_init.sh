#!/usr/bin/env bash
# Set up everything the phase scripts need. Safe to run any number of times:
# each check only does work when something is missing or out of date.
#   run/0_init.sh
set -u
RW="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RW" || exit 1
if [ -t 1 ]; then B=$'\e[1m'; D=$'\e[2m'; G=$'\e[32m'; R=$'\e[31m'; Y=$'\e[33m'; N=$'\e[0m'; else B=; D=; G=; R=; Y=; N=; fi
ok()   { echo "  ${G}ok${N}    $*"; }
doing(){ echo "  ${Y}...${N}   $*"; }
warn() { echo "  ${Y}warn${N}  $*"; }
fail() { echo "  ${R}FAIL${N}  $*"; exit 1; }
T0=$(date +%s)
n=0; TOTAL=7
hdr() { n=$((n + 1)); echo "${B}[$n/$TOTAL] $*${N}"; }

echo "${B}Ride World: setup${N}  ${D}($RW)${N}"

hdr "Python"
PYSYS=$(command -v python3) || fail "python3 not found; install Python 3.10 or newer"
"$PYSYS" -c 'import sys; sys.exit(sys.version_info < (3, 10))' || fail "Python 3.10+ needed, found $("$PYSYS" --version)"
ok "$("$PYSYS" --version) at $PYSYS"

hdr "Virtual environment (.venv)"
if [ -x .venv/bin/python ]; then ok "exists: $RW/.venv"
else
  doing "creating $RW/.venv"
  "$PYSYS" -m venv .venv || fail "could not create .venv (on Ubuntu: sudo apt install python3-venv)"
  ok "created"
fi
PY=.venv/bin/python

hdr "Python packages (requirements.txt)"
SHA=$(sha256sum requirements.txt | cut -c1-16)
if [ "$(cat .venv/.req_sha 2>/dev/null)" = "$SHA" ] && $PY -c "import torch, osmnx, geopandas, h3, numba, folium" 2>/dev/null; then
  ok "already installed (requirements.txt unchanged)"
else
  doing "installing; the first time this downloads about 3 GB (PyTorch with CUDA) and takes 10-20 min"
  $PY -m pip install --upgrade pip --progress-bar on -q || fail "pip upgrade failed"
  $PY -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126 --progress-bar on \
    | { grep -v "Requirement already satisfied" || true; }
  [ "${PIPESTATUS[0]}" = 0 ] || fail "pip install failed; see the messages above"
  echo "$SHA" > .venv/.req_sha
  ok "installed"
fi

hdr "PyTorch and the GPU"
G="$G" Y="$Y" N="$N" $PY - <<'EOF' || fail "PyTorch could not be imported"
import os, torch
G, Y, N = os.environ["G"], os.environ["Y"], os.environ["N"]
print(f"  {G}ok{N}    torch {torch.__version__}", end="")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f", GPU: {p.name}, {p.total_memory / 2**30:.1f} GB")
    x = torch.randn(512, 512, device="cuda"); (x @ x).sum().item()
    print(f"  {G}ok{N}    a small computation on the GPU worked")
else:
    print(f"\n  {Y}warn{N}  no CUDA GPU found: phases 1-2 (simulation) work, model training will be very slow")
EOF

hdr "Trained models kept in the repository"
for f in data/wm_models/gru_cons_os.pt data/wm2_models/st4_skip_ros.pt data/wm3_models/wm3_exo.pt \
         data/wm3_models/rl_bc_pre.pt data/wm3_models/mbc_pre.pt data/wm3_models/drv_pre.pt; do
  if [ -f "$f" ]; then ok "$f"; else warn "missing $f (the phase that trains it will recreate it)"; fi
done

hdr "Folders and disk space"
mkdir -p logs/run && ok "logs go to $RW/logs/run/"
chmod +x run/*.sh scripts/*.sh 2>/dev/null
FREE=$(df -BG --output=avail . | tail -1 | tr -dc 0-9)
USED=$(du -sBG data/sf_hd data/pune_hd 2>/dev/null | awk '{s+=$1} END{print s+0}')
if [ "$USED" -gt 50 ]; then ok "${FREE} GB free; the high-detail data is already here (${USED} GB)"
elif [ "$FREE" -lt 150 ]; then warn "${FREE} GB free; rebuilding all data from scratch needs about 150 GB"
else ok "${FREE} GB free (a full rebuild needs about 150 GB)"; fi

hdr "What is already done"
for s in run/[1-5]_*.sh; do bash "$s" --status; done

echo
echo "${G}${B}Setup complete${N} in $(( $(date +%s) - T0 )) s. Next:"
echo "  run/1_build_world.sh --list     see the steps of a phase, their time and status"
echo "  run/1_build_world.sh            run a phase (finished steps are skipped)"
echo "  run/3_world_model3.sh --eval-only --force   rerun only the quick evaluation steps"
