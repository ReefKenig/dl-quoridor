#!/bin/bash
# Deploy Quoridor to production server
# Usage: cd dl-quoridor && ./deploy/deploy.sh
# Requires: VPN connection to Colman network + SSH key installed on server

set -e

SERVER="cs501@10.10.248.141"
REMOTE_DIR="~/dl-quoridor"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Get run directories for served variants from MODELS.json
registered_runs=$(python3 -c "
import json
with open('$PROJECT_DIR/runs/MODELS.json') as f:
    registry = json.load(f)
dirs = set()
for v in registry['variants'].values():
    if isinstance(v, str): continue
    entry = registry['models'][v['model']]
    path = entry.get('path') or entry.get('run_dir', '')
    if path.startswith('runs/'):
        dirs.add(path.split('/')[1])
for d in sorted(dirs):
    print(d)
")

# Build dynamic excludes for runs/ dirs not in the registry
run_excludes=()
for dir in "$PROJECT_DIR"/runs/*/; do
    dirname=$(basename "$dir")
    if ! echo "$registered_runs" | grep -qx "$dirname"; then
        run_excludes+=(--exclude="runs/$dirname")
    fi
done

echo "==> Syncing project files..."
rsync -avz --progress \
  --exclude-from="$SCRIPT_DIR/excludes.txt" \
  "${run_excludes[@]}" \
  ./ ${SERVER}:${REMOTE_DIR}/

echo "==> Restarting service..."
ssh ${SERVER} "sudo /usr/bin/systemctl restart quoridor"

echo "==> Done! Checking status..."
ssh ${SERVER} "systemctl status quoridor --no-pager | head -5"
