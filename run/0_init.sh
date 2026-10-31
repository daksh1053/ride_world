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
if [ "$(cat .venv/.req_sha 2>/dev/null)" = "$SHA" ] && $PY -c "import torch, osmnx, geopandas, h3, numba, folium, rasterio" 2>/dev/null; then
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
    print(f"\n  {Y}warn{N}  no CUDA GPU found: the eval modes and the simulation steps work, training will be very slow")
EOF

hdr "External input data (only full-run needs it)"
GH=data/_external/ghsl
URL=https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/GHS_POP_E2025_GLOBE_R2023A_54009_100/V1-0/tiles
mkdir -p "$GH"
for tile in R5_C8 R7_C26; do      # the GHSL population tiles around San Francisco and Pune
  f="GHS_POP_E2025_GLOBE_R2023A_54009_100_V1_0_$tile"
  if [ -f "$GH/$f.tif" ]; then ok "GHSL population tile $tile"
  else
    doing "downloading GHSL population tile $tile"
    curl -fL --progress-bar -o "$GH/$f.zip" "$URL/$f.zip" || fail "download failed: $URL/$f.zip"
    (cd "$GH" && unzip -oq "$f.zip" "$f.tif") || fail "could not unzip $GH/$f.zip"
    ok "GHSL population tile $tile"
  fi
done
if [ -f data/_external/sfcta/tnc_trip_stats.json ]; then ok "SFCTA ride counts (kept in the repository)"
else warn "data/_external/sfcta is missing; restore it with: git checkout -- data/_external/sfcta"; fi

hdr "Folders and disk space"
mkdir -p runs && ok "the phase scripts write to $RW/runs/"
chmod +x run/*.sh 2>/dev/null
FREE=$(df -BG --output=avail . | tail -1 | tr -dc 0-9)
if [ "$FREE" -lt 200 ]; then warn "${FREE} GB free; a full run of every phase needs about 200 GB"
else ok "${FREE} GB free (a full run of every phase needs about 200 GB; the eval modes need under 1 GB)"; fi

hdr "Status of each phase"
for s in run/[1-5]_*.sh; do bash "$s" --status; done

echo
echo "${G}${B}Setup complete${N} in $(( $(date +%s) - T0 )) s. Next, for example:"
echo "  run/5_is_it_a_world_model.sh list           the steps of a phase, their time and status"
echo "  run/5_is_it_a_world_model.sh eval-precomputed   evaluate the results shipped in precomputed/"
echo "  run/1_build_world.sh full-run               compute phase 1 from scratch into runs/full_run/"
