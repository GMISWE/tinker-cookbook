#!/bin/bash
# Test 7: DPO - Direct Preference Optimization
#
# This tests DPO training on the HHH (Helpful, Harmless, Honest) preference dataset.
# DPO uses a custom loss function and is similar to SFT (debug_train_only=True).
#
# Prerequisites:
# - slime-training-0 pod running with Ray cluster (DP=2, PP=2)
# - kgateway-training deployment up and running
# - tinker_gmi and tinker-cookbook installed in slime-training-0 pod

kubectl exec -n slime-gmi slime-training-0 -- bash -c 'export TINKER_API_KEY=slime-dev-key && python -m tinker_cookbook.recipes.preference.dpo.train \
    model_name=/data/models/Qwen2.5-0.5B-Instruct \
    renderer_name=qwen3 \
    base_url=http://kgateway-training.slime-gmi:8000 \
    dataset=hhh \
    batch_size=8 \
    learning_rate=1e-5 \
    dpo_beta=0.1 \
    max_length=8192 \
    log_path=/tmp/cookbook-dpo-test \
    behavior_if_log_dir_exists=delete 2>&1'
