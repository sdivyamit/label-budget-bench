#!/usr/bin/env bash
# Run on your Mac. Pulls every result off the GPU box so the instance can be deleted safely.
#   bash queue/end_of_day.sh <ssh-host>            # sync only (fleet keeps running, Weeks 3-5)
#   bash queue/end_of_day.sh <ssh-host> --stop     # also stop workers after their current job,
#                                                  # then print the snapshot + delete steps (Week 2)
set -euo pipefail
HOST=${1:?usage: end_of_day.sh <ssh-host> [--stop]}
REMOTE=${REMOTE_DIR:-label-budget-bench}
cd "$(dirname "$0")/.."

if [ "${2:-}" = "--stop" ]; then
  ssh "$HOST" "touch $REMOTE/queue/STOP"
  echo "STOP requested; waiting for running jobs to finish or checkpoint..."
  while ssh "$HOST" "pgrep -f train/run.py >/dev/null"; do sleep 30; done
fi

for d in preds results logs queue/failed; do
  rsync -az --info=stats0 "$HOST:$REMOTE/$d/" "./$d/"
done
rsync -az "$HOST:$REMOTE/checkpoints/" ./checkpoints/ 2>/dev/null || true
echo "synced: $(ls preds/runs | wc -l | tr -d ' ') run directories, $(ls preds/runs/*/DONE 2>/dev/null | wc -l | tr -d ' ') finished"

if [ "${2:-}" = "--stop" ]; then
  ssh "$HOST" "rm -f $REMOTE/queue/STOP"
  cat <<EOF

Workers stopped and results synced. To stop paying for the GPU:
  1. Thunder console -> instance -> Snapshot (keeps the env + HF cache; ~USD 0.05/GB/month).
  2. Wait for the snapshot to show Ready, then Delete the instance.
  3. Next morning: create an instance from the snapshot, then on it:
       cd $REMOTE && nohup bash -c 'CUDA_VISIBLE_DEVICES=0 .venv/bin/python queue/worker.py' > worker0.log 2>&1 &
EOF
fi
