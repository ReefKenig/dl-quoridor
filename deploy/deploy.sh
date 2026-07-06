#!/bin/bash
# Deploy Quoridor to production server
# Usage: cd dl-quoridor && ./deploy/deploy.sh
# Requires: VPN connection to Colman network + SSH key installed on server

set -e

SERVER="cs501@10.10.248.141"
REMOTE_DIR="~/quoridor"

echo "==> Syncing project files..."
rsync -avz --progress \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='notebooks/' \
  --exclude='outputs/' \
  --exclude='docs/' \
  --exclude='.git' \
  --exclude='runs/legacy_2p/' \
  --exclude='runs/n2_5x5_buf10k_v1/' \
  --exclude='runs/n2_5x5_cv2_v1/' \
  --exclude='runs/n2_5x5_g097_v1/' \
  --exclude='runs/n4_5x5_v1/' \
  --exclude='runs/n4_5x5_v2_killed/' \
  --exclude='*.ipynb' \
  ./ ${SERVER}:${REMOTE_DIR}/

echo "==> Restarting service..."
ssh ${SERVER} "sudo /usr/bin/systemctl restart quoridor"

echo "==> Done! Checking status..."
ssh ${SERVER} "systemctl status quoridor --no-pager | head -5"
