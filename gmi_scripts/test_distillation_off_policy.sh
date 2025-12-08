#!/bin/bash
# Test 6: Model Distillation - Off-policy SFT on OpenThoughts3
#
# This tests standard supervised learning on the OpenThoughts3 dataset,
# which contains reasoning traces with chain-of-thought style responses.
#
# This is an SFT task (debug_train_only=True)
#
# NOTE: Parameters reduced for H200 GPU memory constraints:
# - batch_size: 8→4 (reduced for memory)
# - max_length: 8192→512 (reduced to prevent OOM)
# - buffer_size: 1024→256 (adjusted proportionally)
# - max_prompts: 100→50 (reduced dataset size for faster testing)
#
# Prerequisites:
# - miles-training-0 pod running with Ray cluster (DP=2, PP=2)
# - kgateway-training deployment up and running
# - tinker_gmi and tinker-cookbook installed in miles-training-0 pod

kubectl exec -n miles-gmi-tinker miles-training-0 -- bash -c 'export TINKER_API_KEY=slime-dev-key && python -m tinker_cookbook.recipes.distillation.off_policy_reasoning \
    model_name=/data/models/Qwen2.5-0.5B-Instruct \
    renderer_name=qwen3 \
    base_url=http://kgateway-training.miles-gmi-tinker:8000 \
    batch_size=4 \
    learning_rate=1e-4 \
    lora_rank=32 \
    max_length=512 \
    buffer_size=256 \
    max_prompts=50 \
    eval_every=5 \
    save_every=20 \
    num_epochs=1 \
    log_path=/tmp/cookbook-distillation-test \
    behavior_if_log_dir_exists=delete 2>&1'
