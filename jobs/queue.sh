#!/usr/bin/env bash
# Queue worker: one per GPU. Jobs are claimed atomically (mkdir), so several workers share a job list.
#   bash jobs/queue.sh <gpu> [jobs file]      (default jobs file: jobs/all_jobs.txt, lines "<set> <model> <fold>")
set -uo pipefail
gpu=$1
here=$(cd "$(dirname "$0")" && pwd)
list=${2:-$here/all_jobs.txt}
claims=${CLAIMS_DIR:-$here/.claims}
mkdir -p "$claims"
while read -r set model fold; do
  [[ -z ${set:-} || $set == \#* ]] && continue
  mkdir "$claims/$set-$model-fold$fold" 2>/dev/null || continue
  echo "[$(date -Is)] GPU $gpu -> $set $model fold$fold"
  if python "$here/../scripts/run_job.py" --set "$set" --model "$model" --fold "$fold" --gpu "$gpu"; then
    echo "[$(date -Is)] GPU $gpu OK $set $model fold$fold"
  else
    echo "[$(date -Is)] GPU $gpu FAILED $set $model fold$fold"; touch "$claims/$set-$model-fold$fold/FAILED"
  fi
done < "$list"
echo "[$(date -Is)] GPU $gpu queue empty"
