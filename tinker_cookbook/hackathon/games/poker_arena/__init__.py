"""
Poker Arena: LLM Texas Hold'em

A multi-agent poker environment where 6 LLM agents play No-Limit
Texas Hold'em. Agents must make strategic betting decisions with
imperfect information (hidden hole cards).

Key Features:
- Sequential turns (one player at a time)
- Imperfect information (hidden opponent cards)
- JSON-based observations and actions
- Standard poker hand evaluation

Example:
    python -m tinker_cookbook.hackathon.games.poker_arena.train \\
        model_name=/data/models/Qwen2.5-0.5B-Instruct \\
        num_players=6 \\
        starting_stack=200
"""

from tinker_cookbook.hackathon.games.poker_arena.env import (
    PokerArenaDatasetBuilder,
    PokerEnv,
    PokerEnvGroupBuilder,
    PokerGameState,
    PokerRoundCoordinator,
    Card,
    evaluate_hand,
    best_hand,
)

__all__ = [
    "PokerArenaDatasetBuilder",
    "PokerEnv",
    "PokerEnvGroupBuilder",
    "PokerGameState",
    "PokerRoundCoordinator",
    "Card",
    "evaluate_hand",
    "best_hand",
]
