#!/bin/bash
# Test Text Arena (TicTacToe) on miles-gmi-tinker namespace
#
# This tests the existing TicTacToe 2-player game from the multiplayer_rl
# recipes as a baseline for multi-agent coordination.
#
# Prerequisites:
# - miles-training-0 pod running with Ray cluster (DP=2, PP=2)
# - kgateway-training deployment up and running
# - tinker_gmi and tinker-cookbook installed in pod
# - textarena package installed

echo "=== Text Arena (TicTacToe) Test ==="
echo "Testing 2-player TicTacToe baseline..."
echo ""

kubectl exec -n miles-gmi-tinker miles-training-0 -- bash -c '
export TINKER_API_KEY=slime-dev-key
export TINKER_BASE_URL=http://kgateway-training.miles-gmi-tinker:8000
export PYTHONPATH=/tmp/tinker-cookbook:/tmp/tinker_gmi/src:$PYTHONPATH
python -u -m tinker_cookbook.recipes.multiplayer_rl.text_arena.train \
    model_name=/data/models/Qwen2.5-0.5B-Instruct \
    renderer_name=qwen3 \
    game_name=TicTacToe-v0 \
    batch_size=8 \
    learning_rate=3e-5 \
    max_tokens=32 \
    num_train_datapoints=50 \
    num_test_datapoints=10 \
    eval_every=5 \
    save_every=20 \
    log_path=/tmp/hackathon-tictactoe-test
' 2>&1

EXIT_CODE=$?
echo ""
echo "=== Test Complete ==="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Exit Code: 0 ✅ SUCCESS"
else
    echo "Exit Code: $EXIT_CODE ❌ FAILED"
fi
