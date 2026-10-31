#!/usr/bin/env bash
# Step kinds: data (makes training data), train, sim (long simulator / imagination runs), eval (scores and figures).
# Shared helpers for the run/ scripts: the step list, skipping finished steps,
# the live progress line and the logs. Phase scripts `source` this file.
#
# A phase script declares steps, then calls run_phase "$@":
#   step "<title>" <estimated minutes> <data|train|eval> "<files that mark it done>" "<command>" ["<command>" ...]
# Several commands in one step run in parallel (at most $MAXJOBS at once).
# Use "a && b" inside one command to run things one after the other.

set -u
set -m                       # each background job gets its own process group, so it can be stopped cleanly
RW="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RW" || exit 1
PY="$RW/.venv/bin/python -u"
export PYTHONUNBUFFERED=1
MAXJOBS=${MAXJOBS:-3}
HEARTBEAT=${HEARTBEAT:-60}   # seconds between progress lines when the output is not a terminal

STEP_TITLE=(); STEP_EST=(); STEP_KIND=(); STEP_DONEIF=(); STEP_CMDS=()

step() {
  local title=$1 est=$2 kind=$3 doneif=$4; shift 4
  STEP_TITLE+=("$title"); STEP_EST+=("$est"); STEP_KIND+=("$kind"); STEP_DONEIF+=("$doneif")
  local IFS=$'\x1f'; STEP_CMDS+=("$*")
}

# ---------- formatting ----------
if [ -t 1 ]; then TTY=1; B=$'\e[1m'; D=$'\e[2m'; G=$'\e[32m'; R=$'\e[31m'; Y=$'\e[33m'; C=$'\e[36m'; N=$'\e[0m'
else TTY=0; B=; D=; G=; R=; Y=; C=; N=; fi

fmt_min() {  # minutes -> "1 h 20 min" / "45 min"
  local m=${1%.*}; [ -z "$m" ] && m=0
  if [ "$m" -ge 60 ]; then printf "%d h %02d min" $((m / 60)) $((m % 60)); else printf "%d min" "$m"; fi
}
fmt_sec() {  # seconds -> "1:02:03" / "02:03"
  local s=$1
  if [ "$s" -ge 3600 ]; then printf "%d:%02d:%02d" $((s / 3600)) $((s % 3600 / 60)) $((s % 60)); else printf "%02d:%02d" $((s / 60)) $((s % 60)); fi
}
slug() { echo "$1" | tr 'A-Z' 'a-z' | tr -cs 'a-z0-9' '_' | sed 's/^_//; s/_$//' | cut -c1-40; }

# ---------- done checks ----------
stamp_file() { echo "$LOGDIR/$(printf %02d "$1").done"; }
is_done() {  # step index (1-based)
  local i=$1 f
  [ -f "$(stamp_file "$i")" ] && return 0
  local doneif=${STEP_DONEIF[$((i - 1))]}
  [ -z "$doneif" ] && return 1
  for f in $doneif; do compgen -G "$f" > /dev/null || return 1; done
  return 0
}

# ---------- live status ----------
CPU_PREV=0; CPU_T=0; CPU_PCT="-"; GPU_TXT=""; GPU_T=0
group_cpu_seconds() {  # total CPU seconds used so far by the given process groups
  local pg pat=""
  for pg in "$@"; do pat="$pat|$pg"; done
  ps -eo pgid=,times= 2>/dev/null | awk -v p="${pat#|}" 'BEGIN{n=split(p,a,"|"); for(i=1;i<=n;i++) w[a[i]]=1} ($1 in w){s+=$2} END{print s+0}'
}
update_usage() {
  local now; now=$(date +%s)
  if [ $((now - CPU_T)) -ge 2 ]; then
    local cur; cur=$(group_cpu_seconds "$@")
    if [ "$CPU_T" -gt 0 ]; then
      local d=$(( (cur - CPU_PREV) * 100 / (now - CPU_T) )); [ "$d" -lt 0 ] && d=0
      CPU_PCT="${d}%"
    fi
    CPU_PREV=$cur; CPU_T=$now
  fi
  if command -v nvidia-smi > /dev/null && [ $((now - GPU_T)) -ge 5 ]; then
    GPU_TXT=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | awk -F', ' '{printf "GPU %s%% %.1fG", $1, $2/1024}')
    GPU_T=$now
  fi
}
last_line() {  # last non-empty line of a log (tqdm's \r updates count as lines)
  tail -c 4000 "$1" 2>/dev/null | tr '\r' '\n' | sed 's/\x1b\[[0-9;]*[A-Za-z]//g' | grep -v -e '^[[:space:]]*$' -e '^### ' | tail -1 | sed 's/^[[:space:]]*//'
}
bar() {  # percent -> [██████░░░░]
  local p=$1 w=12 f; [ "$p" -gt 100 ] && p=100; f=$((p * w / 100))
  printf "["; printf "%${f}s" "" | tr ' ' '#'; printf "%$((w - f))s" "" | tr ' ' '.'; printf "]"
}

# ---------- run one step ----------
PIDS=(); JOBLOGS=()
stop_all() {
  local p
  for p in "${PIDS[@]}"; do kill -TERM -- "-$p" 2>/dev/null; done
  sleep 1
  for p in "${PIDS[@]}"; do kill -KILL -- "-$p" 2>/dev/null; wait "$p" 2>/dev/null; done
}
on_interrupt() {
  [ "$TTY" = 1 ] && printf "\r\e[K"
  echo "${Y}Stopped by you.${N} Finished steps are kept; run the same command again to continue."
  stop_all; exit 130
}

run_step() {  # index, total
  local i=$1 total=$2 idx=$(( $1 - 1 ))
  local title=${STEP_TITLE[$idx]} est=${STEP_EST[$idx]}
  local IFS=$'\x1f'; local cmds=(${STEP_CMDS[$idx]}); unset IFS
  local n=${#cmds[@]} base; base="$LOGDIR/$(printf %02d "$i")_$(slug "$title")"
  echo
  echo "${B}[$i/$total] $title${N}  ${D}(estimated ~$(fmt_min "$est"))${N}"
  PIDS=(); JOBLOGS=()
  local k
  for ((k = 0; k < n; k++)); do
    if [ "$n" = 1 ]; then JOBLOGS+=("$base.log"); else JOBLOGS+=("${base}_job$((k + 1)).log"); fi
    echo "  ${D}\$ ${cmds[$k]//"$PY"/python}${N}"
  done
  if [ "$n" = 1 ]; then echo "  ${D}log: ${JOBLOGS[0]}${N}"; else echo "  ${D}logs: ${base}_job*.log ($n jobs, up to $MAXJOBS at once)${N}"; fi
  [ "$DRY" = 1 ] && return 0

  local t0; t0=$(date +%s)
  local next=0 running=() status=() fail=0 last_hb=$t0 spin='|/-\' s=0
  for ((k = 0; k < n; k++)); do status+=("wait"); done
  CPU_T=0; CPU_PCT="-"
  while :; do
    # start jobs while there is room
    while [ "$next" -lt "$n" ] && [ "${#running[@]}" -lt "$MAXJOBS" ] && [ "$fail" = 0 ]; do
      ( echo "### $(date '+%F %T')  ${cmds[$next]}"; eval "${cmds[$next]}" ) > "${JOBLOGS[$next]}" 2>&1 &
      PIDS+=($!); running+=("$next:$!"); status[$next]="run"; next=$((next + 1))
    done
    # collect finished jobs
    local still=() r j p
    for r in "${running[@]}"; do
      j=${r%%:*}; p=${r#*:}
      if kill -0 "$p" 2>/dev/null; then still+=("$r")
      else
        wait "$p"; local rc=$?
        if [ "$rc" = 0 ]; then status[$j]="ok"; else status[$j]="fail"; fail=1; FAILED_LOG=${JOBLOGS[$j]}; FAILED_RC=$rc; fi
      fi
    done
    running=("${still[@]}")
    if [ "$fail" = 1 ] && [ "${#running[@]}" -gt 0 ]; then
      [ "$TTY" = 1 ] && printf "\r\e[K"
      echo "  ${R}A job failed; stopping the other jobs of this step.${N}"
      stop_all; running=()
    fi
    [ "${#running[@]}" = 0 ] && { [ "$next" -ge "$n" ] || [ "$fail" = 1 ]; } && break

    # progress line
    local now el pct ndone=0 pgs=() msg="" lf age
    now=$(date +%s); el=$((now - t0)); pct=$((el * 100 / (est * 60 + 1)))
    for ((k = 0; k < n; k++)); do [ "${status[$k]}" = ok ] && ndone=$((ndone + 1)); done
    for r in "${running[@]}"; do pgs+=("${r#*:}"); done
    update_usage "${pgs[@]}"
    lf=${JOBLOGS[${running[0]%%:*}]}
    age=$(( now - $(stat -c %Y "$lf" 2>/dev/null || echo "$now") ))
    msg=$(last_line "$lf")
    local jobs=""; [ "$n" -gt 1 ] && jobs=" | jobs $ndone/$n done"
    local over=""; [ "$pct" -gt 100 ] && over=" (over estimate, still running)"
    local head; head="$(fmt_sec "$el") / ~$(fmt_min "$est") $(bar "$pct") ${pct}%${over}${jobs} | CPU ${CPU_PCT}${GPU_TXT:+ $GPU_TXT}"
    if [ "$TTY" = 1 ]; then
      local cols; cols=$(tput cols 2>/dev/null || echo 120)
      local line=" ${spin:s%4:1} $head | ${msg:-starting...} (${age}s ago)"
      printf "\r\e[K%s" "${line:0:$((cols - 1))}"
      s=$((s + 1)); sleep 1
    else
      if [ $((now - last_hb)) -ge "$HEARTBEAT" ]; then
        echo "  ... $head | ${msg:0:150} (${age}s ago)"; last_hb=$now
      fi
      sleep 2
    fi
  done
  [ "$TTY" = 1 ] && printf "\r\e[K"
  local el=$(( $(date +%s) - t0 ))
  if [ "$fail" = 1 ]; then
    echo "  ${R}FAILED${N} after $(fmt_sec "$el") (exit code $FAILED_RC). Last lines of $FAILED_LOG:"
    tail -n 25 "$FAILED_LOG" | sed 's/^/    /'
    echo "  ${Y}Fix the problem and run the same command again; finished steps are skipped.${N}"
    exit 1
  fi
  echo "  ${G}done${N} in $(fmt_sec "$el")  ${D}(estimated ~$(fmt_min "$est"))${N}"
  date '+%F %T' > "$(stamp_file "$i")"
  echo "$(date '+%F %T'),$PHASE,$i,\"$title\",$el,$est" >> "$RW/logs/run/times.csv"
}

# ---------- the phase ----------
usage() {
  cat <<EOF
$PHASE_TITLE
Usage: run/$(basename "$0") [options]
  (no option)     run every step that is not done yet; finished steps are skipped
  -l, --list      list the steps, their estimated time and whether they are done
  --status        one-line summary (used by 0_init.sh)
  -f, --force     run the selected steps again even if they are done
  --from N        start at step N
  --only N[,M..]  run only these steps
  --eval-only     run only evaluation and figure steps (no data generation, long simulator runs or training)
  -n, --dry-run   show the commands without running them
  -j N            run up to N parallel jobs inside a step (default $MAXJOBS)
Logs: logs/run/$PHASE/   Timings of every finished step: logs/run/times.csv
EOF
}

run_phase() {
  local list=0 force=0 from=1 only="" evalonly=0 status=0; DRY=0
  while [ $# -gt 0 ]; do
    case $1 in
      -l|--list) list=1 ;; --status) status=1 ;; -f|--force) force=1 ;;
      --from) from=$2; shift ;; --only) only=",$2," ; shift ;;
      --eval-only) evalonly=1 ;; -n|--dry-run) DRY=1 ;; -j) MAXJOBS=$2; shift ;;
      -h|--help) usage; exit 0 ;; *) echo "Unknown option: $1"; usage; exit 2 ;;
    esac; shift
  done
  LOGDIR="$RW/logs/run/$PHASE"; mkdir -p "$LOGDIR"
  local total=${#STEP_TITLE[@]} i sel=() pend=0 ndone=0
  for ((i = 1; i <= total; i++)); do
    local want=1 k=${STEP_KIND[$((i - 1))]}
    [ "$i" -lt "$from" ] && want=0
    [ -n "$only" ] && [[ "$only" != *",$i,"* ]] && want=0
    [ "$evalonly" = 1 ] && [ "$k" != eval ] && want=0
    is_done "$i" && ndone=$((ndone + 1))
    if [ "$want" = 1 ] && { [ "$force" = 1 ] || ! is_done "$i"; }; then sel+=("$i"); pend=$((pend + ${STEP_EST[$((i - 1))]})); fi
  done

  if [ "$status" = 1 ]; then
    printf "  %-62s %2d/%-2d steps done" "$PHASE_TITLE" "$ndone" "$total"
    [ "$ndone" -lt "$total" ] && printf ", ~%s left" "$(fmt_min "$pend")"; echo; return 0
  fi
  if [ "$list" = 1 ]; then
    echo "${B}$PHASE_TITLE${N}"
    local all=0
    for ((i = 1; i <= total; i++)); do
      local e=${STEP_EST[$((i - 1))]} st="${Y}to do${N}"; all=$((all + e))
      is_done "$i" && st="${G}done${N} "
      printf "  %2d. %-72s %-6s ~%-14s %b\n" "$i" "${STEP_TITLE[$((i - 1))]}" "${STEP_KIND[$((i - 1))]}" "$(fmt_min "$e")" "$st"
    done
    echo "  Whole phase: ~$(fmt_min "$all") on a 16-core PC with a 4 GB RTX 3050."
    return 0
  fi

  if [ ! -x "$RW/.venv/bin/python" ]; then echo "${R}No .venv found.${N} Run run/0_init.sh first."; exit 1; fi
  trap on_interrupt INT TERM
  echo "${B}${C}$PHASE_TITLE${N}"
  echo "$PHASE_DESC"
  echo "Steps: $total, already done: $ndone, to run now: ${#sel[@]}"
  if [ "${#sel[@]}" = 0 ]; then
    echo "${G}Nothing to do: every step is done.${N} Use --force to run steps again, or --list to see them."; return 0
  fi
  echo "Estimated time for these steps: ~$(fmt_min "$pend") (finishes around $(date -d "+$pend min" '+%a %H:%M'))"
  local skipped=()
  for ((i = 1; i <= total; i++)); do
    [[ " ${sel[*]} " == *" $i "* ]] || skipped+=("$i")
  done
  [ "${#skipped[@]}" -gt 0 ] && echo "${D}Skipped (done or not selected): steps ${skipped[*]}${N}"
  local t0 n=0 left=$pend; t0=$(date +%s)
  for i in "${sel[@]}"; do
    n=$((n + 1))
    [ "$n" -gt 1 ] && echo "${D}  phase progress: step $n of ${#sel[@]} | elapsed $(fmt_sec $(( $(date +%s) - t0 ))) | ~$(fmt_min "$left") left${N}"
    run_step "$i" "$total"
    left=$((left - ${STEP_EST[$((i - 1))]}))
  done
  echo
  echo "${G}${B}$PHASE_TITLE: finished${N} in $(fmt_sec $(( $(date +%s) - t0 )))."
  [ -n "${PHASE_RESULTS:-}" ] && echo "Results: $PHASE_RESULTS"
  return 0
}
