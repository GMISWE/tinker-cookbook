#!/usr/bin/env python3
"""
Test 3: Preference Learning - Shorter Responses

This tests the PairwisePreferenceRLDatasetBuilder with a simple preference model
that always prefers shorter responses. This is an RL task (needs debug_train_only=False).

Expected: ac_tokens_per_turn should drop significantly after 40 steps
"""

import asyncio
from time import time

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.preference.shorter.env import (
    ShorterComparisonBuilder,
    ShorterPreferenceModelBuilder,
)
from tinker_cookbook.rl import train
from tinker_cookbook.rl.preference_envs import PairwisePreferenceRLDatasetBuilder


def build_config() -> train.Config:
    # Use local model path (kgateway auto-adds _torch_dist if needed)
    model_name = "/data/models/Qwen2.5-0.5B-Instruct"
    renderer_name = "qwen3"  # Explicit renderer for local path

    comparison_builder = ShorterComparisonBuilder()
    dataset_builder = PairwisePreferenceRLDatasetBuilder(
        comparison_builder=comparison_builder,
        batch_size=8,  # Reduced for testing
        policy_renderer_name=renderer_name,
        policy_model_name=model_name,
        group_size=4,  # Reduced for testing
        preference_model_builder=ShorterPreferenceModelBuilder(),
    )

    return train.Config(
        model_name=model_name,
        log_path=f"/tmp/cookbook-shorter-test",
        dataset_builder=dataset_builder,
        learning_rate=3e-5,
        max_tokens=64,
        eval_every=5,
        compute_post_kl=True,  # Re-enabled to test SGLang prompt_logprobs with logging
        base_url="http://kgateway-training.slime-gmi:8000",
        debug_train_only=False,  # RL needs SGLang for sampling
    )


def main():
    config = build_config()
    # Auto-delete old log dir for automated testing
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="delete")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    main()
