"""
Minimal supervised learning test using tinker-cookbook pattern.
Tests compatibility with kgateway-training + Slime backend.
"""
import asyncio
import chz
import sys
from tinker_cookbook import model_info
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig


def build_config_blueprint() -> chz.Blueprint[train.Config]:
    # Use local Qwen2.5-0.5B-Instruct model for fast testing
    model_name = "/data/models/Qwen2.5-0.5B-Instruct_torch_dist"
    # Use HF model name for tokenizer/renderer lookup
    hf_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    renderer_name = model_info.get_recommended_renderer_name(hf_model_name)

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=hf_model_name,  # Use HF name for tokenizer
        renderer_name=renderer_name,
        max_length=2048,  # Shorter for testing
        batch_size=2,     # Small batch size
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )

    # Use our minimal test dataset
    dataset = FromConversationFileBuilder(
        common_config=common_config,
        file_path="/tmp/tinker-cookbook/test_data_mini.jsonl"
    )

    return chz.Blueprint(train.Config).apply({
        "log_path": "/tmp/tinker-examples/test_sl_minimal",
        "model_name": model_name,
        "dataset_builder": dataset,
        "learning_rate": 2e-4,
        "lr_schedule": "constant",  # Keep it simple
        "num_epochs": 1,
        "eval_every": 0,  # No evaluation for now
        "save_every": 0,  # No checkpointing for now
        # Point to kgateway-training service
        "base_url": "http://kgateway-training.slime-gmi:8000",
    })


def main(config: train.Config):
    print(f"Testing with model: {config.model_name}")
    print(f"Base URL: {config.base_url}")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    blueprint = build_config_blueprint()
    # Override with command-line args if provided
    blueprint.make_from_argv(sys.argv[1:])
    config = blueprint.make()
    main(config)
