#!/usr/bin/env bash
# Shared code for the run/ phase scripts. Each phase script declares its steps and the
# files its evaluation reads, then calls run_phase "$@". See README, "Running with the scripts".
#
# Three modes, given as the first word:
#   full-run          compute everything (data, training, simulator runs) into runs/full_run/
#   eval-full-run     evaluate the results of your own full run   (reads runs/full_run/,  writes runs/eval_full_run/)
#   eval-precomputed  evaluate the results shipped in the repo    (reads precomputed/,    writes runs/eval_precomputed/)
#
# A step is declared as:
#   step "<title>" <estimated minutes> <kind> "<files that show it is done>" "<command>" ["<command>" ...]
# kind: data (makes training data), train, sim (long simulator or imagination runs),
#       test (scores a model on held-out days; needs the whole dataset, so it is part of the full run),
#       page (builds browsable pages of computed runs)                               -> full-run;
#       eval (scores, figures and reports from saved results)                        -> the eval modes.
# Several commands in one step run in parallel (at most $MAXJOBS at once); "a && b" runs one after the other.
# Paths in "done" files and in EVAL_INPUTS are relative to the mode's folder (data/..., outputs/...).

set -u
set -m                       # each background job gets its own process group, so it can be stopped cleanly
set -f                       # file patterns are expanded only with compgen, inside the right folder, never against the repo
RW="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RW" || exit 1
PY="$RW/.venv/bin/python -u"
export PYTHONUNBUFFERED=1
MAXJOBS=${MAXJOBS:-3}
HEARTBEAT=${HEARTBEAT:-60}   # seconds between progress lines when the output is not a terminal
FULL_RUN_DIR="$RW/runs/full_run"
PRECOMPUTED_DIR="$RW/precomputed"

STEP_TITLE=(); STEP_EST=(); STEP_KIND=(); STEP_DONEIF=(); STEP_CMDS=()
EVAL_INPUTS=""               # set by the phase script: files the eval steps read (globs), all required
EVAL_OPTIONAL=""             # files the eval steps use when present (copied if they exist)
EVAL_NEEDS=""                # results of other phases' eval steps this eval reads: "<file>=<phase script>" ...

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
is_compute() { [ "$1" != eval ]; }

# ---------- done checks (full-run only) ----------
stamp_file() { echo "$LOGDIR/$(printf %02d "$1").done"; }
is_done() {  # step index (1-based), checked inside $WORK
  local i=$1 f
  [ -f "$(stamp_file "$i")" ] && return 0
  local doneif=${STEP_DONEIF[$((i - 1))]}
  [ -z "$doneif" ] && return 1
  for f in $doneif; do compgen -G "$WORK/$f" > /dev/null || return 1; done
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
bar() {  # percent -> [####........]
  local p=$1 w=12 f; [ "$p" -gt 100 ] && p=100; f=$((p * w / 100))
  printf "["; printf "%${f}s" "" | tr ' ' '#'; printf "%$((w - f))s" "" | tr ' ' '.'; printf "]"
}

# ---------- run one step ----------
PIDS=()
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
  local n=${#cmds[@]} base k; base="$LOGDIR/$(printf %02d "$i")_$(slug "$title")"
  local joblogs=()
  echo
  echo "${B}[$i/$total] $title${N}  ${D}(estimated ~$(fmt_min "$est"))${N}"
  for ((k = 0; k < n; k++)); do
    if [ "$n" = 1 ]; then joblogs+=("$base.log"); else joblogs+=("${base}_job$((k + 1)).log"); fi
    echo "  ${D}\$ ${cmds[$k]//"$PY"/python}${N}"
  done
  if [ "$n" = 1 ]; then echo "  ${D}log: ${joblogs[0]}${N}"; else echo "  ${D}logs: ${base}_job*.log ($n jobs, up to $MAXJOBS at once)${N}"; fi
  [ "$DRY" = 1 ] && return 0

  local t0; t0=$(date +%s)
  local next=0 running=() status=() fail=0 last_hb=$t0 spin='|/-\' s=0 failed_log="" failed_rc=0
  for ((k = 0; k < n; k++)); do status+=("wait"); done
  PIDS=(); CPU_T=0; CPU_PCT="-"
  while :; do
    while [ "$next" -lt "$n" ] && [ "${#running[@]}" -lt "$MAXJOBS" ] && [ "$fail" = 0 ]; do
      ( set +f; echo "### $(date '+%F %T')  ${cmds[$next]}"; eval "${cmds[$next]}" ) > "${joblogs[$next]}" 2>&1 &
      PIDS+=($!); running+=("$next:$!"); status[$next]="run"; next=$((next + 1))
    done
    local still=() r j p
    for r in "${running[@]}"; do
      j=${r%%:*}; p=${r#*:}
      if kill -0 "$p" 2>/dev/null; then still+=("$r")
      else
        wait "$p"; local rc=$?
        if [ "$rc" = 0 ]; then status[$j]="ok"; else status[$j]="fail"; fail=1; failed_log=${joblogs[$j]}; failed_rc=$rc; fi
      fi
    done
    running=("${still[@]}")
    if [ "$fail" = 1 ] && [ "${#running[@]}" -gt 0 ]; then
      [ "$TTY" = 1 ] && printf "\r\e[K"
      echo "  ${R}A job failed; stopping the other jobs of this step.${N}"
      stop_all; running=()
    fi
    [ "${#running[@]}" = 0 ] && { [ "$next" -ge "$n" ] || [ "$fail" = 1 ]; } && break

    local now el pct ndone=0 pgs=() msg lf age
    now=$(date +%s); el=$((now - t0)); pct=$((el * 100 / (est * 60 + 1)))
    for ((k = 0; k < n; k++)); do [ "${status[$k]}" = ok ] && ndone=$((ndone + 1)); done
    for r in "${running[@]}"; do pgs+=("${r#*:}"); done
    update_usage "${pgs[@]}"
    lf=${joblogs[${running[0]%%:*}]}
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
    echo "  ${R}FAILED${N} after $(fmt_sec "$el") (exit code $failed_rc). Last lines of $failed_log:"
    tail -n 25 "$failed_log" | sed 's/^/    /'
    if [ "$MODE" = full-run ]; then echo "  ${Y}Fix the problem and run the same command again; finished steps are skipped.${N}"
    else echo "  ${Y}Fix the problem and run the same command again.${N}"; fi
    exit 1
  fi
  echo "  ${G}done${N} in $(fmt_sec "$el")  ${D}(estimated ~$(fmt_min "$est"))${N}"
  date '+%F %T' > "$(stamp_file "$i")"
  echo "$(date '+%F %T'),$MODE,$PHASE,$i,\"$title\",$el,$est" >> "$WORK/logs/times.csv"
}

# ---------- preparing the eval folder ----------
expand_inputs() {  # print the EVAL_INPUTS files that exist in folder $1, one per line, relative to it
  local src=$1 g
  ( cd "$src" 2>/dev/null || exit 0
    for g in $EVAL_INPUTS $EVAL_OPTIONAL; do compgen -G "$g"; done )
}
missing_inputs() {  # print the EVAL_INPUTS patterns with no match in folder $1
  local src=$1 g
  for g in $EVAL_INPUTS; do compgen -G "$src/$g" > /dev/null || echo "$g"; done
}
prepare_eval() {
  local miss; miss=$(missing_inputs "$SRC")
  if [ -n "$miss" ]; then
    echo "${R}The evaluation inputs are not all in $SRC.${N} Missing:"
    echo "$miss" | sed 's/^/    /' | head -20
    if [ "$MODE" = eval-full-run ]; then echo "Run ${B}run/$(basename "$0") full-run${N} first (and the earlier phases' full runs)."
    else echo "precomputed/ is incomplete; restore it with: git checkout -- precomputed"; fi
    exit 1
  fi
  local need lack=""
  for need in $EVAL_NEEDS; do
    compgen -G "$WORK/${need%%=*}" > /dev/null || lack="$lack ${need#*=}"
  done
  if [ -n "$lack" ]; then
    echo "${R}This evaluation also uses results of other phases' evaluations, which are not in $WORK yet.${N}"
    echo "Run first:"; for need in $(echo $lack | tr ' ' '\n' | sort -u); do echo "    run/$need $MODE"; done
    exit 1
  fi
  # remove this phase's old eval results so nothing stale can be mistaken for a new result
  local i f
  for ((i = 1; i <= ${#STEP_TITLE[@]}; i++)); do
    [ "${STEP_KIND[$((i - 1))]}" = eval ] || continue
    for f in ${STEP_DONEIF[$((i - 1))]}; do compgen -G "$WORK/$f" | while read -r x; do rm -rf "$x"; done; done
    rm -f "$(stamp_file "$i")"
  done
  local files n sz
  files=$(expand_inputs "$SRC"); n=$(echo "$files" | grep -c .)
  sz=$(cd "$SRC" && echo "$files" | tr '\n' '\0' | du -sch --files0-from=- 2>/dev/null | tail -1 | cut -f1)
  echo "Copying the $n input files ($sz) from $SRC"
  echo "                                   to $WORK"
  ( cd "$SRC" && echo "$files" | tr '\n' '\0' | xargs -0 cp -a --parents -t "$WORK" ) || { echo "${R}copy failed${N}"; exit 1; }
}

# ---------- the phase ----------
usage() {
  cat <<EOF
${B}$PHASE_TITLE${N}
Usage: run/$(basename "$0") <mode> [options]

Modes:
  full-run          compute everything this phase needs (data, training, simulator runs)
                    into runs/full_run/; finished steps are skipped, so it resumes after a stop
  eval-full-run     evaluate the results of your full run: copies the files the evaluation
                    reads from runs/full_run/ to runs/eval_full_run/ and makes the scores,
                    figures and reports there
  eval-precomputed  the same evaluation on the results shipped in the repository:
                    copies them from precomputed/ to runs/eval_precomputed/ and evaluates there
  list              list the steps, their estimated time, and their status in your full run

Options:
  --from N        start at step N            --only N[,M..]  run only these steps
  -f, --force     full-run: run steps again even if they are done
  -n, --dry-run   show the commands without running them
  -j N            parallel jobs inside a step (default $MAXJOBS)
EOF
}

setup_work() {  # $1 mode -> sets SRC, WORK, LOGDIR, exports the folder variables
  case $1 in
    full-run)         SRC="";                 WORK="$FULL_RUN_DIR" ;;
    eval-full-run)    SRC="$FULL_RUN_DIR";    WORK="$RW/runs/eval_full_run" ;;
    eval-precomputed) SRC="$PRECOMPUTED_DIR"; WORK="$RW/runs/eval_precomputed" ;;
  esac
  LOGDIR="$WORK/logs/$PHASE"
  export RIDE_WORLD_DATA="$WORK/data" RIDE_WORLD_OUTPUTS="$WORK/outputs"
}

run_phase() {
  MODE=""; local force=0 from=1 only="" status=0; DRY=0
  while [ $# -gt 0 ]; do
    case $1 in
      full-run|eval-full-run|eval-precomputed|list) MODE=$1 ;;
      -l|--list) MODE=list ;; --status) status=1 ;; -f|--force) force=1 ;;
      --from) from=$2; shift ;; --only) only=",$2," ; shift ;;
      -n|--dry-run) DRY=1 ;; -j) MAXJOBS=$2; shift ;;
      --print-inputs) echo "$EVAL_INPUTS" | tr ' ' '\n' | grep .; exit 0 ;;     # used by run/make_precomputed.sh
      --print-optional) echo "$EVAL_OPTIONAL" | tr ' ' '\n' | grep .; exit 0 ;;
      -h|--help|help) usage; exit 0 ;;
      *) echo "Unknown argument: $1"; echo; usage; exit 2 ;;
    esac; shift
  done
  [ -z "$MODE" ] && [ "$status" = 0 ] && { usage; exit 2; }
  local total=${#STEP_TITLE[@]} i

  if [ "$status" = 1 ] || [ "$MODE" = list ]; then
    setup_work full-run
    local ndone=0 ncomp=0 pend=0 all_c=0 all_e=0
    for ((i = 1; i <= total; i++)); do
      local e=${STEP_EST[$((i - 1))]}
      if is_compute "${STEP_KIND[$((i - 1))]}"; then
        ncomp=$((ncomp + 1)); all_c=$((all_c + e))
        if is_done "$i"; then ndone=$((ndone + 1)); else pend=$((pend + e)); fi
      else all_e=$((all_e + e)); fi
    done
    if [ "$status" = 1 ]; then
      local pre="${G}complete${N}"; [ -n "$(missing_inputs "$PRECOMPUTED_DIR")" ] && pre="${R}incomplete${N}"
      printf "  %-58s precomputed: %b | full run: %2d/%-2d steps done" "$PHASE_TITLE" "$pre" "$ndone" "$ncomp"
      [ "$ndone" -lt "$ncomp" ] && printf ", ~%s left" "$(fmt_min "$pend")"; echo; return 0
    fi
    echo "${B}$PHASE_TITLE${N}"
    printf "  %-3s %-72s %-6s %-14s %s\n" "" "step" "kind" "estimate" "in runs/full_run"
    for ((i = 1; i <= total; i++)); do
      local k=${STEP_KIND[$((i - 1))]} st="${Y}to do${N}"
      if is_compute "$k"; then is_done "$i" && st="${G}done${N}"; else st="${D}(eval step)${N}"; fi
      printf "  %2d. %-72s %-6s ~%-13s %b\n" "$i" "${STEP_TITLE[$((i - 1))]}" "$k" "$(fmt_min "${STEP_EST[$((i - 1))]}")" "$st"
    done
    echo "  full-run: ~$(fmt_min "$all_c")   each eval mode: ~$(fmt_min "$all_e")   (16-core PC, 4 GB RTX 3050)"
    return 0
  fi

  if [ ! -x "$RW/.venv/bin/python" ]; then echo "${R}No .venv found.${N} Run run/0_init.sh first."; exit 1; fi
  setup_work "$MODE"
  mkdir -p "$WORK/data" "$WORK/outputs" "$LOGDIR"
  trap on_interrupt INT TERM

  local sel=() pend=0 ndone=0
  for ((i = 1; i <= total; i++)); do
    local k=${STEP_KIND[$((i - 1))]} want=1
    if [ "$MODE" = full-run ]; then is_compute "$k" || want=0; else [ "$k" = eval ] || want=0; fi
    [ "$i" -lt "$from" ] && want=0
    [ -n "$only" ] && [[ "$only" != *",$i,"* ]] && want=0
    [ "$want" = 0 ] && continue
    if [ "$MODE" = full-run ] && [ "$force" = 0 ] && is_done "$i"; then ndone=$((ndone + 1)); continue; fi
    sel+=("$i"); pend=$((pend + ${STEP_EST[$((i - 1))]}))
  done

  echo "${B}${C}$PHASE_TITLE${N}  ${B}[$MODE]${N}"
  echo "$PHASE_DESC"
  case $MODE in
    full-run)         echo "Computes into:   $WORK" ;;
    eval-full-run)    echo "Reads results of your full run: $SRC"; echo "Writes the evaluation to:       $WORK" ;;
    eval-precomputed) echo "Reads the results shipped in the repo: $SRC"; echo "Writes the evaluation to:              $WORK" ;;
  esac
  if [ "${#sel[@]}" = 0 ]; then
    if [ "$MODE" = full-run ] && [ "$ndone" -gt 0 ]; then
      echo "${G}Nothing to do: all $ndone computation steps are done.${N} Next: run/$(basename "$0") eval-full-run"
    else echo "No steps selected."; fi
    return 0
  fi
  [ "$MODE" = full-run ] && [ "$ndone" -gt 0 ] && echo "${D}Already done in this full run (skipped): $ndone steps${N}"
  if [ "$MODE" = full-run ] && [ "$DRY" = 0 ]; then
    # inputs from outside the project (population grid, SFCTA counts): shared, read only
    [ -d "$RW/data/_external" ] && ln -sfn "$RW/data/_external" "$WORK/data/_external"
  fi
  if [ "$MODE" != full-run ] && [ "$DRY" = 0 ]; then prepare_eval; fi
  echo "Steps to run: ${#sel[@]}, estimated ~$(fmt_min "$pend") (finishes around $(date -d "+$pend min" '+%a %H:%M'))"

  local t0 n=0 left=$pend; t0=$(date +%s)
  for i in "${sel[@]}"; do
    n=$((n + 1))
    [ "$n" -gt 1 ] && echo "${D}  phase progress: step $n of ${#sel[@]} | elapsed $(fmt_sec $(( $(date +%s) - t0 ))) | ~$(fmt_min "$left") left${N}"
    run_step "$i" "$total"
    left=$((left - ${STEP_EST[$((i - 1))]}))
  done
  echo
  echo "${G}${B}$PHASE_TITLE [$MODE]: finished${N} in $(fmt_sec $(( $(date +%s) - t0 )))."
  if [ "$MODE" = full-run ]; then echo "Next: run/$(basename "$0") eval-full-run"
  else echo "Results in $WORK/outputs/: ${PHASE_RESULTS:-}"; fi
  return 0
}
