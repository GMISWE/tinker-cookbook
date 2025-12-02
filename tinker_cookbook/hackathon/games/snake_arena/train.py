"""
CLI for Snake Arena training.

Usage:
    python -m tinker_cookbook.hackathon.games.snake_arena.train \
        model_name=/data/models/Qwen2.5-0.5B-Instruct \
        renderer_name=qwen3 \
        base_url=http://kgateway-training.miles-gmi-tinker:8000 \
        num_players=4 \
        batch_size=8 \
        learning_rate=3e-5
"""

import asyncio
from datetime import datetime

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.hackathon.games.snake_arena.env import SnakeArenaDatasetBuilder
from tinker_cookbook.rl import train


@chz.chz
class CLIConfig:
    # Model settings
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    renderer_name: str | None = None
    base_url: str | None = None
    lora_rank: int = 32

    # Game settings
    num_players: int = 4
    grid_size: int = 20
    num_food: int = 3
    max_steps: int = 200

    # Training settings
    batch_size: int = 8  # Should be divisible by num_players
    learning_rate: float = 3e-5
    max_tokens: int = 64
    num_train_games: int = 1000
    num_test_games: int = 100

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
    run_name = f"snake-arena-{cli_config.num_players}p-{cli_config.batch_size}bs-{cli_config.learning_rate}lr-{date_and_time}"

    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        log_path = f"/tmp/tinker-examples/snake-arena/{run_name}"

    if cli_config.wandb_name is not None:
        wandb_name = cli_config.wandb_name
    else:
        wandb_name = run_name

    dataset_builder = SnakeArenaDatasetBuilder(
        num_players=cli_config.num_players,
        grid_size=cli_config.grid_size,
        num_food=cli_config.num_food,
        max_steps=cli_config.max_steps,
        batch_size=cli_config.batch_size,
        num_train_games=cli_config.num_train_games,
        num_test_games=cli_config.num_test_games,
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
