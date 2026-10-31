#!/usr/bin/env bash
# Fill precomputed/ with exactly the files the evaluation of each phase reads.
# Only needed when the shipped results should change; the eval-precomputed mode reads this folder.
#
#   run/make_precomputed.sh from-original   copy them from the original development folders data/ and outputs/
#                                           (the run behind the slides and STEPS.md)
#   run/make_precomputed.sh from-full-run   copy them from your own full run in runs/full_run/
set -u
set -f    # patterns are expanded with compgen inside the source folder only
RW="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RW" || exit 1
case ${1:-} in
  from-original) SRC="$RW" ;;
  from-full-run) SRC="$RW/runs/full_run" ;;
  *) sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
DST="$RW/precomputed"
LIST=$(mktemp)
for s in $(compgen -G "run/[1-5]_*.sh"); do
  for g in $(bash "$s" --print-inputs); do
    m=$(cd "$SRC" && compgen -G "$g")
    if [ -z "$m" ]; then echo "missing in $SRC: $g  (needed by $s)"; MISSING=1; fi
    echo "$m" >> "$LIST"
  done
  for g in $(bash "$s" --print-optional); do (cd "$SRC" && compgen -G "$g") >> "$LIST"; done
done
[ "${MISSING:-0}" = 1 ] && { echo "Not all inputs exist; precomputed/ was not changed."; rm -f "$LIST"; exit 1; }
sort -u "$LIST" | grep . > "$LIST.u" || { echo "No input files found; precomputed/ was not changed."; exit 1; }
rm -rf "$DST"; mkdir -p "$DST"
(cd "$SRC" && tr '\n' '\0' < "$LIST.u" | xargs -0 cp -a --parents -t "$DST")
cat > "$DST/README.md" <<EOF
# Precomputed results

The files the evaluation steps of every phase read, copied from a finished full run
(\`run/make_precomputed.sh ${1}\`, $(date +%F)). The eval-precomputed mode of the phase scripts
copies them to runs/eval_precomputed/ and evaluates there, so this folder never changes.
The layout mirrors a run folder: data/ (models, simulator records) and outputs/ (saved scores and training logs).
EOF
echo "precomputed/: $(wc -l < "$LIST.u") files, $(du -sh "$DST" | cut -f1)"
echo "largest files:"; find "$DST" -type f -printf '%s %P\n' | sort -rn | head -5 | awk '{printf "  %6.1f MB  %s\n", $1/1e6, $2}'
find "$DST" -type f -size +95M -printf '  WARNING: over the 100 MB GitHub limit: %P\n'
rm -f "$LIST" "$LIST.u"
