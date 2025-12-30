#!/bin/bash
# RLVE Training Full Restart Script
# Usage: ./restart_rlve.sh [MODEL_SIZE] [N_BATCHES]
#   MODEL_SIZE: 0.5B, 7B (default: 7B)
#   N_BATCHES: number of batches (default: 1000)

set -e

MODEL_SIZE=${1:-"7B"}
N_BATCHES=${2:-1000}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="/data/logs"

echo "=========================================="
echo "RLVE Training Full Restart"
echo "=========================================="
echo "Model: $MODEL_SIZE"
echo "Batches: $N_BATCHES"
echo "=========================================="

# Run cleanup (which includes killing processes, restarting Ray, starting server)
echo ""
echo "Running cleanup..."
"${SCRIPT_DIR}/cleanup_rlve.sh"

# Start RLVE training
echo ""
echo "[Step 5/5] Starting RLVE training..."
echo "  Log file: ${LOG_DIR}/tinker-rlve.log"
echo ""

cd "$SCRIPT_DIR"
./run_rlve_gb200.sh "$MODEL_SIZE" "$N_BATCHES"
