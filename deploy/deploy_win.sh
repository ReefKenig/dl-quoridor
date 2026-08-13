#!/bin/bash
# Deploy Quoridor to production server (Windows Friendly Version)
# Usage: cd dl-quoridor && ./deploy/deploy_win.sh

set -e

SERVER="cs501@10.10.248.141"
REMOTE_DIR="~/quoridor"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ARCHIVE_PATH="../deploy_package.tar.gz"
ARCHIVE_NAME="deploy_package.tar.gz"

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

echo "==> Packaging project files (excluding heavy directories)..."
tar --exclude-from="$SCRIPT_DIR/excludes.txt" \
    "${run_excludes[@]}" \
    -czf ${ARCHIVE_PATH} .

echo "==> Transferring archive to server..."
scp ${ARCHIVE_PATH} ${SERVER}:${REMOTE_DIR}/

echo "==> Extracting archive on server..."
ssh ${SERVER} "cd ${REMOTE_DIR} && tar -xzf ${ARCHIVE_NAME} && rm ${ARCHIVE_NAME}"

echo "==> Cleaning up local archive..."
rm ${ARCHIVE_PATH}

echo "==> Restarting service..."
ssh ${SERVER} "sudo /usr/bin/systemctl restart quoridor"

echo "==> Done! Checking status..."
ssh ${SERVER} "systemctl status quoridor --no-pager | head -5"
