#!/bin/bash
# Test Poker Arena on miles-gmi-tinker namespace
#
# This tests the multi-agent Texas Hold'em game where 6 LLM agents
# play poker with imperfect information (hidden cards).
#
# Prerequisites:
# - miles-training-0 pod running with Ray cluster (DP=2, PP=2)
# - kgateway-training deployment up and running
# - tinker_gmi and tinker-cookbook installed in pod

echo "=== Poker Arena Test ==="
echo "Testing 6-player Texas Hold'em..."
echo ""

kubectl exec -n miles-gmi-tinker miles-training-0 -- bash -c '
export TINKER_API_KEY=slime-dev-key
export PYTHONPATH=/tmp/tinker-cookbook:/tmp/tinker_gmi/src:$PYTHONPATH
python -u -m tinker_cookbook.hackathon.games.poker_arena.train \
    model_name=/data/models/Qwen2.5-0.5B-Instruct \
    renderer_name=qwen3 \
    base_url=http://kgateway-training.miles-gmi-tinker:8000 \
    num_players=6 \
    starting_stack=200 \
    small_blind=1 \
    big_blind=2 \
    batch_size=12 \
    learning_rate=3e-5 \
    max_tokens=128 \
    num_train_hands=50 \
    num_test_hands=10 \
    eval_every=5 \
    save_every=20 \
    log_path=/tmp/hackathon-poker-test \
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
