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
#   cd dl-quoridor && git pull && git checkout dev
#   scripts/run_notebook.sh n2        # N=2 9x9 training (run dir from notebook)
#   scripts/run_notebook.sh n4        # N=4 9x9 training (run dir from notebook)
#   MODE=tmux scripts/run_notebook.sh n2      # inside tmux instead of nohup
#
# There is NO default variant — you must pass n2 or n4 explicitly.
#
# Monitor:
#   tail -f runs/<run_dir>/notebook.log
#
# Stop a nohup job:  kill $(cat runs/<run_dir>/notebook.pid)

set -euo pipefail

# --- resolve repo root from THIS script's location, then run from there ---
# Everything (notebooks/, runs/, configs/) is repo-relative — no sibling-clone model.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# --- required variant selector: n2 | n4 (NO default) ---
VARIANT="${1:-}"
case "$VARIANT" in
  n2) NOTEBOOK="notebooks/train_9x9_n2.ipynb" ;;
  n4) NOTEBOOK="notebooks/train_9x9_n4.ipynb" ;;
  *)
    echo "Usage: scripts/run_notebook.sh {n2|n4}   (no default)" >&2
    echo "  n2  ->  notebooks/train_9x9_n2.ipynb" >&2
    echo "  n4  ->  notebooks/train_9x9_n4.ipynb" >&2
    echo "  (run dir is read from RUN_DIR in the notebook)" >&2
    exit 1
    ;;
esac

# --- extract RUN_DIR name from the notebook itself ---
# The notebook defines: RUN_DIR = os.path.join(REPO_DIR, "runs", "<tag>")
# We parse that tag so the script always stays in sync with what the notebook uses.
TAG="$(sed -n 's/.*\\\"runs\\\", \\\"\([^\\]*\)\\\".*/\1/p' "$NOTEBOOK")"
if [ -z "$TAG" ]; then
  echo "ERROR: could not extract RUN_DIR tag from $NOTEBOOK" >&2
  exit 1
fi

# --- config (env-overridable) ---
PYTHON="${PYTHON:-python3}"
MODE="${MODE:-nohup}"
LOG_DIR="${LOG_DIR:-runs/${TAG}}"
SESSION="${SESSION:-quoridor_${TAG}}"

mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/notebook.log"
OUT_NB="$LOG_DIR/${TAG}.executed.ipynb"

if [ ! -f "$NOTEBOOK" ]; then
  echo "ERROR: notebook not found: $NOTEBOOK" >&2
  exit 1
fi

# --- GPU / orphan pre-flight -------------------------------------------------
# The parallel self-play stack spawns many worker processes and one GPU-batcher
# thread. If a previous run was HARD-killed (kernel kill / SIGKILL) mid-self-play
# its finally: cleanup never ran, so worker processes are orphaned and leak CUDA
# state. Enough orphans wedge the GPU, and then the very first CUDA call of the
# next run hangs forever with no exception — which silently burns a full 5-hour
# training slot. Catch that here, before we launch.
#
#   PREFLIGHT=on   (default) abort if a wedged GPU / leftover procs are detected
#   PREFLIGHT=warn           print the warning but launch anyway
#   PREFLIGHT=off            skip the check entirely
#   PREFLIGHT_KILL=1         actively kill leftover GPU python procs, then continue
PREFLIGHT="${PREFLIGHT:-on}"
PREFLIGHT_KILL="${PREFLIGHT_KILL:-0}"

preflight() {
  [ "$PREFLIGHT" = "off" ] && return 0
  command -v nvidia-smi >/dev/null 2>&1 || { echo "  [preflight] no nvidia-smi (CPU host?) — skipping GPU check"; return 0; }

  echo "=== GPU pre-flight ==="
  # Memory summary (used/total across all GPUs).
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
             --format=csv,noheader 2>/dev/null | sed 's/^/  gpu /' || true

  # Leftover compute processes. On a box that runs one training at a time, ANY
  # python compute process here before we launch is an orphan from a prior run.
  local apps
  apps="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory \
                     --format=csv,noheader 2>/dev/null || true)"
  local py_pids
  py_pids="$(echo "$apps" | grep -iE 'python' | awk -F',' '{gsub(/ /,"",$1); print $1}' | grep -E '^[0-9]+$' || true)"

  if [ -z "$py_pids" ]; then
    echo "  [preflight] no leftover GPU compute processes — clean. OK to launch."
    echo "======================"
    return 0
  fi

  echo "  [preflight] WARNING: leftover GPU compute process(es) detected:"
  echo "$apps" | grep -iE 'python' | sed 's/^/    /' || true

  if [ "$PREFLIGHT_KILL" = "1" ]; then
    echo "  [preflight] PREFLIGHT_KILL=1 — terminating leftover PIDs: $py_pids"
    # shellcheck disable=SC2086
    kill $py_pids 2>/dev/null || true
    sleep 5
    # shellcheck disable=SC2086
    kill -9 $py_pids 2>/dev/null || true
    sleep 2
    echo "  [preflight] done. Continuing launch."
    echo "======================"
    return 0
  fi

  echo "  These are almost certainly orphans from a hard-killed run and will"
  echo "  wedge the GPU (the next run's first CUDA call hangs with no error)."
  echo "  Fix, then re-launch:"
  echo "      kill $py_pids           # or: PREFLIGHT_KILL=1 $0"
  echo "      # if memory is still held with no owner:  sudo nvidia-smi --gpu-reset -i 0"
  echo "  Override:  PREFLIGHT=warn $0   (launch anyway)   |   PREFLIGHT=off $0 (skip check)"
  echo "======================"

  if [ "$PREFLIGHT" = "warn" ]; then
    echo "  [preflight] PREFLIGHT=warn — launching despite the warning."
    return 0
  fi
  echo "ERROR: aborting launch to avoid burning a training slot on a wedged GPU." >&2
  exit 1
}

# Don't double-launch the same notebook: if a previous nohup pid is still alive.
PID_FILE="$LOG_DIR/notebook.pid"
if [ -f "$PID_FILE" ]; then
  OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [ -n "${OLD_PID:-}" ] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "ERROR: a run for '$TAG' is already active (pid $OLD_PID). " >&2
    echo "       Stop it first:  kill $OLD_PID" >&2
    exit 1
  fi
fi

preflight

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
