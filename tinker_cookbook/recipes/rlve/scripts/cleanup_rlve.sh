#!/bin/bash
# RLVE Environment Cleanup Script
# Kills processes, restarts Ray, starts server - but does NOT start training
# Usage: ./cleanup_rlve.sh [--no-server]
#   --no-server: Skip starting the opentinker-miles server

set -e

START_SERVER=true
if [[ "$1" == "--no-server" ]]; then
    START_SERVER=false
fi

LOG_DIR="/data/logs"

echo "=========================================="
echo "RLVE Environment Cleanup"
echo "=========================================="
echo "Start server: $START_SERVER"
echo "=========================================="

# Step 1: Kill existing processes
echo ""
echo "[Step 1/4] Killing existing processes..."
pkill -9 -f "tinker_cookbook.recipes.rlve.train" 2>/dev/null || true
pkill -9 -f "uvicorn.*training" 2>/dev/null || true
sleep 2
echo "  ✓ Processes killed"

# Step 2: Clean logs and Python cache
echo ""
echo "[Step 2/4] Cleaning logs and Python cache..."
rm -rf ${LOG_DIR}/rlve-tinker 2>/dev/null || true
rm -f ${LOG_DIR}/tinker-rlve.log 2>/dev/null || true
rm -f ${LOG_DIR}/opentinker-miles.log 2>/dev/null || true
mkdir -p ${LOG_DIR}
find /root/gavin/opentinker-miles -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find /root/gavin/tinker-cookbook -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find /root/gavin/miles -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
echo "  ✓ Logs and cache cleaned"

# Step 3: Restart Ray
echo ""
echo "[Step 3/4] Restarting Ray cluster..."
ray stop --force 2>/dev/null || true
sleep 3
ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265 > /dev/null 2>&1
sleep 3

# Verify Ray is up
if ray status > /dev/null 2>&1; then
    GPU_COUNT=$(ray status 2>&1 | grep -oP '\d+\.\d+/\d+\.\d+ GPU' | head -1 | grep -oP '/\K\d+' || echo "4")
    echo "  ✓ Ray cluster started with ${GPU_COUNT} GPUs"
else
    echo "  ✗ Ray failed to start"
    exit 1
fi

# Step 4: Start opentinker-miles server (optional)
if [ "$START_SERVER" = true ]; then
    echo ""
    echo "[Step 4/4] Starting opentinker-miles server..."
    cd /root/gavin/opentinker-miles

    ALLOW_PARTIAL_BATCHES=true \
    PYTHONPATH=/root/gavin/opentinker-miles:/root/Megatron-LM:/root/miles:$PYTHONPATH \
    nohup python3 -m uvicorn training.api:app --host 0.0.0.0 --port 8000 \
        >> ${LOG_DIR}/opentinker-miles.log 2>&1 &

    SERVER_PID=$!
    echo "  Server PID: $SERVER_PID"

    # Wait for server to be ready (max 30 seconds)
    echo "  Waiting for server to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:8000/health -H "X-API-Key: slime-dev-key" > /dev/null 2>&1; then
            echo "  ✓ Server ready after ${i}s"
            break
        fi
        if [ $i -eq 30 ]; then
            echo "  ✗ Server failed to start after 30s"
            tail -20 ${LOG_DIR}/opentinker-miles.log
            exit 1
        fi
        sleep 1
    done

    # Verify server health
    echo ""
    echo "Server health:"
    HEALTH=$(curl -s http://localhost:8000/health -H "X-API-Key: slime-dev-key")
    echo "  $HEALTH" | python3 -m json.tool 2>/dev/null || echo "  $HEALTH"
else
    echo ""
    echo "[Step 4/4] Skipping server start (--no-server)"
fi

echo ""
echo "=========================================="
echo "Cleanup complete!"
echo "=========================================="
if [ "$START_SERVER" = true ]; then
    echo "Server is running. You can now:"
    echo "  - Run RLVE training manually"
    echo "  - Run tests against the server"
    echo "  - Check server logs: tail -f ${LOG_DIR}/opentinker-miles.log"
else
    echo "Ray is ready. Server was not started."
    echo "To start server manually, use /restart-server"
fi
echo "=========================================="
