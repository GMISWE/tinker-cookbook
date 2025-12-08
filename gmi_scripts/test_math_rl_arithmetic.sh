#!/bin/bash
# Successful math RL test with arithmetic environment
#
# Context: This script tests the Tinker API integration with GMI Wrapper (kgateway)
# deployed on a Miles RL training cluster.
#
# Key Fix: num_substeps=1
# - Ensures all 8 samples (group_size=4 × groups_per_batch=2) are sent together
# - Matches Miles's global_batch_size=8 expectation (hardcoded in kgateway)
# - Without this fix, samples get split into chunks causing batch size mismatch
#
# Prerequisites:
# - miles-training-0 pod running with Ray cluster (DP=2, PP=2)
# - kgateway-training deployment up and running
# - tinker_gmi and tinker-cookbook installed in miles-training-0 pod
#
# Usage:
#   ./test_math_rl_arithmetic.sh

kubectl exec -n miles-gmi-tinker miles-training-0 -- bash -c 'export TINKER_API_KEY=slime-dev-key && python -m tinker_cookbook.recipes.math_rl.train \
    model_name=/data/models/Qwen2.5-0.5B-Instruct \
    renderer_name=qwen3 \
    env=arithmetic \
    base_url=http://kgateway-training.miles-gmi-tinker:8000 \
    group_size=4 \
    groups_per_batch=2 \
    learning_rate=2e-4 \
    max_tokens=64 \
    eval_every=5 \
    save_every=10 \
    num_substeps=1 \
    behavior_if_log_dir_exists=delete 2>&1'
