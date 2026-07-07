#!/bin/bash
# Deploy Quoridor to production server (Windows Friendly Version)
# Usage: cd dl-quoridor && ./deploy/deploy_win.sh

set -e

SERVER="cs501@10.10.248.141"
REMOTE_DIR="~/quoridor"
ARCHIVE_NAME="deploy_package.tar.gz"

echo "==> Packaging project files (excluding heavy directories)..."
tar --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='notebooks' \
    --exclude='outputs' \
    --exclude='docs' \
    --exclude='.git' \
    --exclude='runs/legacy_2p' \
    --exclude='runs/n2_5x5_buf10k_v1' \
    --exclude='runs/n2_5x5_cv2_v1' \
    --exclude='runs/n2_5x5_g097_v1' \
    --exclude='runs/n4_5x5_v1' \
    --exclude='runs/n4_5x5_v2_killed' \
    --exclude='*.ipynb' \
    -czf ${ARCHIVE_NAME} .

echo "==> Transferring archive to server..."
scp ${ARCHIVE_NAME} ${SERVER}:${REMOTE_DIR}/

echo "==> Extracting archive on server..."
ssh ${SERVER} "cd ${REMOTE_DIR} && tar -xzf ${ARCHIVE_NAME} && rm ${ARCHIVE_NAME}"

echo "==> Cleaning up local archive..."
rm ${ARCHIVE_NAME}

echo "==> Restarting service..."
ssh ${SERVER} "sudo /usr/bin/systemctl restart quoridor"

echo "==> Done! Checking status..."
ssh ${SERVER} "systemctl status quoridor --no-pager | head -5"
