#!/bin/bash
# Test 4: Prompt Distillation - Language Classification
#
# This tests prompt distillation (SFT) on a minimal language classification dataset.
# The goal is to internalize a language classification prompt into the model parameters.
#
# This is an SFT task (debug_train_only=True)
#
# Prerequisites:
# - slime-training-0 pod running with Ray cluster (DP=2, PP=2)
# - kgateway-training deployment up and running
# - tinker_gmi and tinker-cookbook installed in slime-training-0 pod

# Copy the minimal dataset to the pod
echo "Copying minimal language classification dataset to pod..."
kubectl cp /tmp/tmux-tmp/tinker-cookbook/gmi_scripts/minimal_lang_data.jsonl slime-gmi/slime-training-0:/tmp/minimal_lang_data.jsonl

# Run the training
echo "Starting prompt distillation training..."
kubectl exec -n slime-gmi slime-training-0 -- bash -c 'export TINKER_API_KEY=slime-dev-key && python -m tinker_cookbook.recipes.prompt_distillation.train \
    model_name=/data/models/Qwen2.5-0.5B-Instruct \
    renderer_name=qwen3 \
    base_url=http://kgateway-training.slime-gmi:8000 \
    file_path=/tmp/minimal_lang_data.jsonl \
    batch_size=4 \
    learning_rate=1e-4 \
    lora_rank=32 \
    max_length=2048 \
    num_epochs=2 \
    eval_every=5 \
    save_every=10 \
    log_path=/tmp/cookbook-prompt-distillation-test \
    behavior_if_log_dir_exists=delete 2>&1'
