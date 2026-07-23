#!/usr/bin/env bash
#
# Run a training notebook headlessly (no browser, survives disconnect).
#
# Executes the notebook top-to-bottom with jupyter nbconvert, streaming output
# to a logfile. Because it runs at the OS level via nohup, you can close the
# browser / drop the SSH session and training keeps going. The notebook's own
# resume logic (latest.pt + meta.json) means re-running picks up where it left.
#
# Usage:
#   scripts/run_notebook.sh                                  # defaults: N=2
#   NOTEBOOK=notebooks/train_9x9_n4.ipynb scripts/run_notebook.sh
#   MODE=tmux scripts/run_notebook.sh                        # inside tmux instead
#
# Monitor:
#   tail -f runs/<run_dir>/notebook.log
#
# Stop a nohup job:  kill $(cat runs/<run_dir>/notebook.pid)

set -euo pipefail

# --- resolve working directory ---
# Run from the caller's directory (the workspace root), NOT from the project dir.
# The project (dl-quoridor/) is expected to be cloned alongside the notebook.
RUN_DIR="${RUN_DIR:-$(pwd)}"
cd "$RUN_DIR"

# --- config (env-overridable) ---
PYTHON="${PYTHON:-python3}"
NOTEBOOK="${NOTEBOOK:-train_9x9_n2.ipynb}"
MODE="${MODE:-nohup}"
# Where logs/executed copy go. Derive a run tag from the notebook name.
TAG="$(basename "$NOTEBOOK" .ipynb)"
LOG_DIR="${LOG_DIR:-runs/${TAG}}"
SESSION="${SESSION:-quoridor_${TAG}}"

mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/notebook.log"
OUT_NB="$LOG_DIR/${TAG}.executed.ipynb"

if [ ! -f "$NOTEBOOK" ]; then
  echo "ERROR: notebook not found: $NOTEBOOK" >&2
  exit 1
fi

# nbconvert execution:
#   --to notebook --execute : run every cell in order
#   --ExecutePreprocessor.timeout=-1 : no per-cell timeout (training is long)
#   --output : write the executed copy (with outputs) next to the logs
CMD="$PYTHON -m jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=-1 \
  --output '$OUT_NB' \
  '$NOTEBOOK'"

# Keep macOS awake while training (no-op on Linux).
CAFFEINATE=""
if command -v caffeinate >/dev/null 2>&1; then
  CAFFEINATE="caffeinate -i"
fi
CMD="$CAFFEINATE $CMD"

echo "=== launching notebook execution ==="
echo "  mode      : $MODE"
echo "  notebook  : $NOTEBOOK"
echo "  logfile   : $LOG"
echo "  executed  : $OUT_NB"
echo "===================================="

case "$MODE" in
  nohup)
    setsid bash -c "$CMD" >>"$LOG" 2>&1 &
    PID=$!
    echo "$PID" > "$LOG_DIR/notebook.pid"
    echo "Started (pid $PID). Tail with:  tail -f $LOG"
    echo "Stop with:  kill $PID   (or kill \$(cat $LOG_DIR/notebook.pid))"
    ;;
  tmux)
    if ! command -v tmux >/dev/null 2>&1; then
      echo "ERROR: tmux not installed. Use MODE=nohup instead." >&2
      exit 1
    fi
    tmux new-session -d -s "$SESSION" "$CMD 2>&1 | tee -a '$LOG'"
    echo "Started tmux session '$SESSION'."
    echo "Attach with:  tmux attach -t $SESSION      (detach: Ctrl-b then d)"
    echo "Tail log:     tail -f $LOG"
    ;;
  *)
    echo "ERROR: unknown MODE='$MODE' (use nohup or tmux)." >&2
    exit 1
    ;;
esac
