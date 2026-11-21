#!/bin/bash
# Train the 30M and then the 60M driver policy (scripts/42), restarting from the last saved
# round after a crash (the 4 GB GPU sometimes runs out of memory).
cd /home/daksh/Desktop/btp/ride_world
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ITERS=30
run() {  # tag chunk workers... args...
  tag=$1; chunk=$2; shift 2
  for try in $(seq 1 12); do
    done_it=$(tail -n +2 outputs/wm3/stage7b/train_$tag.csv 2>/dev/null | wc -l)
    [ "$done_it" -ge $ITERS ] && return 0
    res=""; [ -f data/wm3_models/$tag.pt ] && [ "$done_it" -ge 1 ] && res="--resume"
    echo "== $(date) $tag attempt $try $res (rounds done $done_it)" >> outputs/wm3/stage7b/logs/42_$tag.log
    DRV2_CHUNK=$chunk .venv/bin/python scripts/42_wm3_driver2_grpo.py --tag $tag --iters $ITERS --policy-device cuda $res "$@" \
      >> outputs/wm3/stage7b/logs/42_$tag.log 2>&1
    sleep 30
  done
}
run drv2_30m 3,3 --hidden 768 --layers 6 --heads 12 --lr 3e-4 --workers 1 1
run drv2_60m 3,3 --hidden 768 --layers 13 --heads 12 --lr 2e-4 --workers 1 1
