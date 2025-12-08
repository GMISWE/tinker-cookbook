#!/bin/bash
# Test 5: Multi-Agent RL - Guess the Number
#
# This tests multi-turn RL where the policy learns to guess a target number
# with feedback on whether the guess is too high or low.
#
# Expected: The policy should learn to efficiently narrow down the target number
#
# Prerequisites:
# - miles-training-0 pod running with Ray cluster (DP=2, PP=2)
# - kgateway-training deployment up and running
# - tinker_gmi and tinker-cookbook installed in miles-training-0 pod

kubectl exec -n miles-gmi-tinker miles-training-0 -- bash -c 'export TINKER_API_KEY=slime-dev-key && export TINKER_BASE_URL=http://kgateway-training.miles-gmi-tinker:8000 && python -m tinker_cookbook.recipes.multiplayer_rl.guess_number.train \
    model_name=/data/models/Qwen2.5-0.5B-Instruct \
    renderer_name=qwen3 \
    group_size=4 \
    batch_size=8 \
    learning_rate=3e-5 \
    max_tokens=64 \
    eval_every=5 \
    save_every=20 \
    log_path=/tmp/cookbook-guess-number-test 2>&1'
