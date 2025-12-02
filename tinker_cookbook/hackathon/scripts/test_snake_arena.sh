#!/bin/bash
# Test Snake Arena on miles-gmi-tinker namespace
#
# This tests the multi-agent Snake Battle Royale game where 4 snakes
# compete on a shared 20x20 grid with simultaneous moves.
#
# Prerequisites:
# - miles-training-0 pod running with Ray cluster (DP=2, PP=2)
# - kgateway-training deployment up and running
# - tinker_gmi and tinker-cookbook installed in pod

echo "=== Snake Arena Test ==="
echo "Testing 4-player Snake Battle Royale..."
echo ""

kubectl exec -n miles-gmi-tinker miles-training-0 -- bash -c '
export TINKER_API_KEY=slime-dev-key
export PYTHONPATH=/tmp/tinker-cookbook:/tmp/tinker_gmi/src:$PYTHONPATH
python -u -m tinker_cookbook.hackathon.games.snake_arena.train \
    model_name=/data/models/Qwen2.5-0.5B-Instruct \
    renderer_name=qwen3 \
    base_url=http://kgateway-training.miles-gmi-tinker:8000 \
    num_players=4 \
    grid_size=20 \
    num_food=3 \
    max_steps=100 \
    batch_size=8 \
    learning_rate=3e-5 \
    max_tokens=64 \
    num_train_games=50 \
    num_test_games=10 \
    eval_every=5 \
    save_every=20 \
    log_path=/tmp/hackathon-snake-test \
    behavior_if_log_dir_exists=delete
' 2>&1

EXIT_CODE=$?
echo ""
echo "=== Test Complete ==="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Exit Code: 0 ✅ SUCCESS"
else
    echo "Exit Code: $EXIT_CODE ❌ FAILED"
fi
