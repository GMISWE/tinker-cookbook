"""
Snake Arena: Multi-agent Snake Battle Royale

A competitive multi-agent environment where 4-8 snake agents compete
on a shared grid. Agents must survive, eat food to grow, and avoid
collisions with walls and other snakes.

Key Features:
- Simultaneous turns (all snakes move at once)
- Full visibility (see all snake positions)
- JSON-based observations and actions
- Configurable grid size and number of players

Example:
    python -m tinker_cookbook.hackathon.games.snake_arena.train \\
        model_name=/data/models/Qwen2.5-0.5B-Instruct \\
        num_players=4 \\
        grid_size=20
"""

from tinker_cookbook.hackathon.games.snake_arena.env import (
    SnakeArenaDatasetBuilder,
    SnakeEnv,
    SnakeEnvGroupBuilder,
    SnakeGameState,
    MultiPlayerCoordinator,
)

__all__ = [
    "SnakeArenaDatasetBuilder",
    "SnakeEnv",
    "SnakeEnvGroupBuilder",
    "SnakeGameState",
    "MultiPlayerCoordinator",
]
