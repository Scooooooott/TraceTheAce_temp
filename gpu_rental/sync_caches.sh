#!/usr/bin/env bash
# Run on the LOCAL machine (not the rented box) for the duration of a long
# precompute run. Periodically pulls the cache .pkl files back so an
# instance failure/preemption partway through a multi-hour run doesn't cost
# more than one sync interval's worth of progress -- the precompute
# scripts' own chunk-checkpointing only protects against the *process*
# dying, not the *instance*'s disk disappearing with it.
#
# Usage: bash gpu_rental/sync_caches.sh user@host [remote_repo_path] [interval_seconds]
set -euo pipefail

REMOTE="${1:?usage: sync_caches.sh user@host [remote_repo_path] [interval_seconds]}"
REMOTE_PATH="${2:-~/trace-the-ace}"
INTERVAL="${3:-600}"

mkdir -p ./gpu_rental/backup
echo "syncing data/processed/*.pkl from $REMOTE:$REMOTE_PATH every ${INTERVAL}s (Ctrl-C to stop)"

while true; do
  ts=$(date +%H:%M:%S)
  if rsync -avz "$REMOTE:$REMOTE_PATH/data/processed/"*.pkl ./gpu_rental/backup/ 2>&1 | grep -v '^$'; then
    echo "[$ts] synced"
  else
    echo "[$ts] rsync failed (instance down? network blip?) -- will retry"
  fi
  sleep "$INTERVAL"
done
