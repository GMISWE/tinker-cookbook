"""
CLI for Poker Arena training.

Usage:
    python -m tinker_cookbook.hackathon.games.poker_arena.train \
        model_name=/data/models/Qwen2.5-0.5B-Instruct \
        renderer_name=qwen3 \
        base_url=http://kgateway-training.miles-gmi-tinker:8000 \
        num_players=6 \
        batch_size=12 \
        learning_rate=3e-5
"""

import asyncio
from datetime import datetime

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.hackathon.games.poker_arena.env import PokerArenaDatasetBuilder
from tinker_cookbook.rl import train


@chz.chz
class CLIConfig:
    # Model settings
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    renderer_name: str | None = None
    base_url: str | None = None
    lora_rank: int = 32

    # Game settings
    num_players: int = 6
    starting_stack: int = 200
    small_blind: int = 1
    big_blind: int = 2

    # Training settings
    batch_size: int = 12  # Should be divisible by num_players
    learning_rate: float = 3e-5
    max_tokens: int = 128  # Poker needs more tokens for reasoning
    num_train_hands: int = 1000
    num_test_hands: int = 100

    # Logging settings
    eval_every: int = 5
    save_every: int = 20
    wandb_project: str | None = None
    wandb_name: str | None = None
    log_path: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


def build_config(cli_config: CLIConfig) -> train.Config:
    """Build the training configuration from CLI config."""
    model_name = cli_config.model_name
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )

    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = f"poker-arena-{cli_config.num_players}p-{cli_config.batch_size}bs-{cli_config.learning_rate}lr-{date_and_time}"

    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        log_path = f"/tmp/tinker-examples/poker-arena/{run_name}"

    if cli_config.wandb_name is not None:
        wandb_name = cli_config.wandb_name
    else:
        wandb_name = run_name

    dataset_builder = PokerArenaDatasetBuilder(
        num_players=cli_config.num_players,
        starting_stack=cli_config.starting_stack,
        small_blind=cli_config.small_blind,
        big_blind=cli_config.big_blind,
        batch_size=cli_config.batch_size,
        num_train_hands=cli_config.num_train_hands,
        num_test_hands=cli_config.num_test_hands,
        model_name=model_name,
        renderer_name=renderer_name,
        base_url=cli_config.base_url,
    )

    return train.Config(
        model_name=model_name,
        base_url=cli_config.base_url,
        log_path=log_path,
        dataset_builder=dataset_builder,
        learning_rate=cli_config.learning_rate,
        max_tokens=cli_config.max_tokens,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        lora_rank=cli_config.lora_rank,
    )


def main():
    cli_config = chz.entrypoint(CLIConfig)
    config = build_config(cli_config)
    cli_utils.check_log_dir(config.log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)
    asyncio.run(train.main(config))


if __name__ == "__main__":
    main()
