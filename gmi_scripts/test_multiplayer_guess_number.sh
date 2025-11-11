#!/bin/bash
# Test 5: Multi-Agent RL - Guess the Number
#
# This tests multi-turn RL where the policy learns to guess a target number
# with feedback on whether the guess is too high or low.
#
# Expected: The policy should learn to efficiently narrow down the target number
#
# Prerequisites:
# - slime-training-0 pod running with Ray cluster (DP=2, PP=2)
# - kgateway-training deployment up and running
# - tinker_gmi and tinker-cookbook installed in slime-training-0 pod

kubectl exec -n slime-gmi slime-training-0 -- bash -c 'export TINKER_API_KEY=slime-dev-key && python -m tinker_cookbook.recipes.multiplayer_rl.guess_number.train \
    model_name=/data/models/Qwen2.5-0.5B-Instruct \
    renderer_name=qwen3 \
    base_url=http://kgateway-training.slime-gmi:8000 \
    group_size=4 \
    batch_size=8 \
    learning_rate=3e-5 \
    max_tokens=64 \
    eval_every=5 \
    save_every=20 \
    log_path=/tmp/cookbook-guess-number-test \
    behavior_if_log_dir_exists=delete 2>&1'
