#!/bin/bash
# Test DPO with REDUCED max_length to avoid OOM
# Based on working GSM8K config which uses max_tokens=128

kubectl exec -n miles-gmi-tinker miles-training-0 -- bash -c 'export TINKER_API_KEY=slime-dev-key && export HF_DATASETS_OFFLINE=0 && export HF_HUB_OFFLINE=0 && export HF_DATASETS_CACHE=/data/datasets && export HF_HOME=/data && python -m tinker_cookbook.recipes.preference.dpo.train \
    model_name=/data/models/Qwen2.5-0.5B-Instruct \
    renderer_name=qwen3 \
    base_url=http://kgateway-training.miles-gmi-tinker:8000 \
    dataset=hhh \
    batch_size=4 \
    learning_rate=1e-5 \
    dpo_beta=0.1 \
    max_length=128 \
    log_path=/tmp/cookbook-dpo-test-reduced \
    behavior_if_log_dir_exists=delete 2>&1'
