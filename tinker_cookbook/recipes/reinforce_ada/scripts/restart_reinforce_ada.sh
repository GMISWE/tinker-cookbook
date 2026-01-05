#!/bin/bash
# Reinforce-Ada Full Restart Script
# Usage: ./restart_reinforce_ada.sh [MODEL_SIZE] [N_BATCHES]
#   MODEL_SIZE: 4B, 7B, 0.5B (default: 4B)
#   N_BATCHES: number of batches (default: 10)

set -e

MODEL_SIZE=${1:-"4B"}
N_BATCHES=${2:-10}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RLVE_SCRIPTS="/root/gavin/tinker-cookbook/tinker_cookbook/recipes/rlve/scripts"

echo "=========================================="
echo "Reinforce-Ada Training Full Restart"
echo "=========================================="
echo "Model: $MODEL_SIZE"
echo "Batches: $N_BATCHES"
echo "=========================================="

# Run cleanup (reuse RLVE cleanup - kills processes, restarts Ray, starts server)
echo ""
echo "Running cleanup..."
"${RLVE_SCRIPTS}/cleanup_rlve.sh"

# Start Reinforce-Ada training
echo ""
echo "[Step 5/5] Starting Reinforce-Ada training..."
echo "  Log file: /data/logs/reinforce-ada.log"
echo ""

"${SCRIPT_DIR}/run_reinforce_ada.sh" "$MODEL_SIZE" "$N_BATCHES"
