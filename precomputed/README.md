# Precomputed results

The files the evaluation steps of every phase read, copied from a finished full run
(`run/make_precomputed.sh from-original`, 2026-10-31). The eval-precomputed mode of the phase scripts
copies them to runs/eval_precomputed/ and evaluates there, so this folder never changes.
The layout mirrors a run folder: data/ (models, simulator records) and outputs/ (saved scores and training logs).
