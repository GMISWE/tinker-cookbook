# Game Development Guide

This guide explains how to create new games for the hackathon.

## Overview

Creating a new game involves:
1. **Design the game rules** - Actions, observations, rewards
2. **Implement the environment** - `env.py` with Env class
3. **Create the training script** - `train.py` with CLI
4. **Write tests** - Verify it works

## Quick Template

### Minimal Game Structure

```
my_game/
├── __init__.py
├── env.py      # Environment implementation
├── train.py    # Training CLI
└── README.md   # Documentation
```

### env.py Template

```python
"""My Game Environment"""

import asyncio
import json
from dataclasses import dataclass
from typing import Sequence

import chz
from tinker import types
from tinker_cookbook.renderers import Renderer, get_renderer
from tinker_cookbook.rl.types import (
    Action, Env, EnvGroupBuilder, Observation,
    RLDataset, RLDatasetBuilder, StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

STOP_CONDITION = ["}\n"]


class MyGameState:
    """Shared game state."""

    def __init__(self):
        self.done = False
        # Initialize your game state

    def get_observation(self, player_id: int) -> dict:
        """Get observation for a player."""
        return {
            "player_id": player_id,
            # Add game-specific observation fields
        }

    def apply_action(self, player_id: int, action: str) -> float:
        """Apply action and return reward."""
        # Process the action
        # Update game state
        # Return reward
        return 0.0


class MyGameCoordinator:
    """Coordinates multi-player actions."""

    def __init__(self, game_state: MyGameState):
        self.game_state = game_state
        self.condition = asyncio.Condition()

    @property
    def done(self) -> bool:
        return self.game_state.done

    async def take_action(self, player_id: int, action: str) -> float:
        """Process action and return reward."""
        async with self.condition:
            reward = self.game_state.apply_action(player_id, action)
            self.condition.notify_all()
            return reward


@dataclass
class MyGameEnv(Env):
    """Environment for a single player."""

    player_id: int
    coordinator: MyGameCoordinator
    renderer: Renderer

    @property
    def stop_condition(self):
        return STOP_CONDITION

    def _format_observation(self) -> str:
        """Format game state as prompt."""
        obs = self.coordinator.game_state.get_observation(self.player_id)
        return f"""Game observation:
{json.dumps(obs, indent=2)}

Respond with JSON: {{"action": "your_action"}}
"""

    async def initial_observation(self):
        prompt = self._format_observation()
        return self.renderer.build_generation_prompt(
            [{"role": "user", "content": prompt}]
        ), self.stop_condition

    async def step(self, action: Action) -> StepResult:
        if self.coordinator.done:
            return StepResult(
                reward=0.0,
                episode_done=True,
                next_observation=types.ModelInput.empty(),
                next_stop_condition=self.stop_condition,
                metrics={},
            )

        # Parse action from LLM output
        action_str = self._parse_action(action)

        # Apply action
        reward = await self.coordinator.take_action(self.player_id, action_str)

        if self.coordinator.done:
            return StepResult(
                reward=reward,
                episode_done=True,
                next_observation=types.ModelInput.empty(),
                next_stop_condition=self.stop_condition,
                metrics={},
            )

        # Get next observation
        prompt = self._format_observation()
        next_obs = self.renderer.build_generation_prompt(
            [{"role": "user", "content": prompt}]
        )

        return StepResult(
            reward=reward,
            episode_done=False,
            next_observation=next_obs,
            next_stop_condition=self.stop_condition,
            metrics={},
        )

    def _parse_action(self, action: Action) -> str:
        """Parse LLM output to action string."""
        try:
            message = self.renderer.parse_response(action)[0]
            text = message["content"]
            data = json.loads(text[text.find("{"):text.rfind("}")+1])
            return data.get("action", "default")
        except:
            return "default"


@dataclass
class MyGameEnvGroupBuilder(EnvGroupBuilder):
    """Builds player environments for one game."""

    renderer: Renderer
    num_players: int = 2

    async def make_envs(self) -> Sequence[Env]:
        game_state = MyGameState()
        coordinator = MyGameCoordinator(game_state)
        return [
            MyGameEnv(
                player_id=i,
                coordinator=coordinator,
                renderer=self.renderer,
            )
            for i in range(self.num_players)
        ]


class MyGameDataset(RLDataset):
    def __init__(self, batch_size: int, builder: MyGameEnvGroupBuilder, num_games: int):
        self.batch_size = batch_size
        self.builder = builder
        self.num_games = num_games

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        games_per_batch = self.batch_size // self.builder.num_players
        return [self.builder for _ in range(games_per_batch)]

    def __len__(self) -> int:
        return self.num_games


@chz.chz
class MyGameDatasetBuilder(RLDatasetBuilder):
    num_players: int = 2
    batch_size: int = 4
    num_train_games: int = 100
    num_test_games: int = 10
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    renderer_name: str = "qwen3"
    base_url: str | None = None

    async def __call__(self):
        renderer = get_renderer(self.renderer_name, get_tokenizer(self.model_name))

        train_builder = MyGameEnvGroupBuilder(
            renderer=renderer,
            num_players=self.num_players,
        )
        train_dataset = MyGameDataset(
            batch_size=self.batch_size,
            builder=train_builder,
            num_games=self.num_train_games,
        )

        test_builder = MyGameEnvGroupBuilder(
            renderer=renderer,
            num_players=self.num_players,
        )
        test_dataset = MyGameDataset(
            batch_size=self.batch_size,
            builder=test_builder,
            num_games=self.num_test_games,
        )

        return train_dataset, test_dataset
```

### train.py Template

```python
"""Training CLI for My Game."""

import asyncio
from datetime import datetime

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.hackathon.games.my_game.env import MyGameDatasetBuilder
from tinker_cookbook.rl import train


@chz.chz
class CLIConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    renderer_name: str | None = None
    base_url: str | None = None
    lora_rank: int = 32

    num_players: int = 2
    batch_size: int = 4
    learning_rate: float = 3e-5
    max_tokens: int = 64

    eval_every: int = 5
    save_every: int = 20
    log_path: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


def build_config(cli_config: CLIConfig) -> train.Config:
    model_name = cli_config.model_name
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(model_name)

    log_path = cli_config.log_path or f"/tmp/my-game/{datetime.now().strftime('%Y-%m-%d-%H-%M')}"

    dataset_builder = MyGameDatasetBuilder(
        num_players=cli_config.num_players,
        batch_size=cli_config.batch_size,
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
        lora_rank=cli_config.lora_rank,
    )


def main():
    cli_config = chz.entrypoint(CLIConfig)
    config = build_config(cli_config)
    cli_utils.check_log_dir(config.log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)
    asyncio.run(train.main(config))


if __name__ == "__main__":
    main()
```

## Design Principles

### 1. Observation Design

**Good observations are:**
- Complete: All info needed to make a decision
- Structured: Easy to parse (JSON)
- Concise: Don't overwhelm the LLM

```python
# Good
{
    "your_position": [5, 3],
    "enemy_positions": [[2, 1], [8, 7]],
    "valid_moves": ["up", "down", "left"]
}

# Bad (too verbose)
"You are at position (5, 3). The first enemy is at (2, 1).
The second enemy is at (8, 7). You can move up, down, or left."
```

### 2. Action Design

**Good actions are:**
- Discrete: Finite set of choices
- Unambiguous: Clear meaning
- JSON-parseable: Easy to extract

```python
# Good
{"action": "move", "direction": "up"}
{"action": "attack", "target": 2}

# Bad
"I'll move up and then maybe attack if I can"
```

### 3. Reward Design

**Good rewards are:**
- Immediate: Provide feedback each step
- Sparse but meaningful: Big rewards for goals
- Normalized: Similar scale across games

```python
# Good reward structure
STEP_SURVIVAL = 0.1      # Small positive for staying alive
GOAL_ACHIEVED = 1.0      # Big positive for winning
FAILURE = -1.0           # Big negative for losing
INVALID_ACTION = -0.5    # Penalty for bad actions
```

### 4. Coordination Patterns

**Simultaneous (like Snake):**
```python
async def submit_move(player_id, move):
    pending_moves[player_id] = move
    if len(pending_moves) == num_players:
        # Process all moves together
        rewards = process_all_moves()
        notify_all()
    else:
        await wait_for_processing()
```

**Sequential (like Poker):**
```python
async def take_turn(player_id, action):
    if current_player != player_id:
        await wait_for_turn()
    process_action(action)
    advance_to_next_player()
    notify_all()
```

## Testing Your Game

### Local Test

```python
# test_my_game.py
import asyncio
from my_game.env import MyGameState, MyGameCoordinator

async def test_game():
    state = MyGameState()
    coord = MyGameCoordinator(state)

    # Simulate a game
    reward1 = await coord.take_action(0, "action_a")
    reward2 = await coord.take_action(1, "action_b")

    print(f"Rewards: {reward1}, {reward2}")
    print(f"Game done: {coord.done}")

asyncio.run(test_game())
```

### Integration Test

```bash
#!/bin/bash
kubectl exec -n miles-gmi-tinker miles-training-0 -- bash -c '
export TINKER_API_KEY=slime-dev-key
python -m tinker_cookbook.hackathon.games.my_game.train \
    model_name=/data/models/Qwen2.5-0.5B-Instruct \
    batch_size=4 \
    num_train_games=10 \
    log_path=/tmp/my-game-test \
    behavior_if_log_dir_exists=delete
'
```

## Common Pitfalls

### 1. Async Deadlocks

**Problem:** Players waiting for each other forever
```python
# Bad - deadlock
async def step(self):
    await other_player.step()  # Other player is waiting for us!
```

**Solution:** Use conditions and notify
```python
# Good
async def step(self):
    async with condition:
        do_action()
        condition.notify_all()  # Wake up waiting players
```

### 2. Missing Stop Condition

**Problem:** LLM generates forever
```python
# Bad
STOP_CONDITION = []  # Never stops!
```

**Solution:** Define clear stop sequences
```python
# Good
STOP_CONDITION = ["}\n", "}\n\n"]  # Stop after JSON
```

### 3. Reward Scale Issues

**Problem:** Rewards too large or too small
```python
# Bad - rewards in different scales
survival_reward = 0.001
win_reward = 1000
```

**Solution:** Keep rewards in similar range
```python
# Good - normalized
survival_reward = 0.1
win_reward = 2.0
```

### 4. Invalid Action Handling

**Problem:** Game crashes on bad LLM output
```python
# Bad
action = json.loads(llm_output)["action"]  # Crashes if invalid
```

**Solution:** Handle gracefully
```python
# Good
try:
    action = json.loads(llm_output)["action"]
except:
    action = "default_action"  # Fallback
```

## Advanced Topics

### Self-Play vs Fixed Opponent

```python
# Self-play: All players learn
train_builder = EnvGroupBuilder(self_play=True)

# Fixed opponent: One player learns
test_builder = EnvGroupBuilder(
    self_play=False,
    opponent_policy=fixed_policy,
)
```

### Multi-Turn Games

For games with multiple interactions per episode:
```python
async def step(self, action):
    reward = process_action(action)

    if episode_done:
        return StepResult(reward=reward, episode_done=True, ...)

    # Continue to next turn
    next_obs = get_next_observation()
    return StepResult(reward=reward, episode_done=False, next_observation=next_obs, ...)
```

### Metrics and Logging

```python
def _get_metrics(self) -> dict[str, float]:
    return {
        "game/score": self.score,
        "game/turns": self.turn_count,
        "player/win_rate": self.wins / max(1, self.games),
    }
```

## Examples

See existing games for reference:
- [Snake Arena](games/snake_arena/) - Simultaneous, 4-8 players
- [Poker Arena](games/poker_arena/) - Sequential, 6 players
- [TicTacToe](../tinker_cookbook/recipes/multiplayer_rl/text_arena/) - Turn-based, 2 players
- [Guess Number](../tinker_cookbook/recipes/multiplayer_rl/guess_number/) - Single agent, multi-turn

## Summary

1. **Design first:** Define rules, observations, actions, rewards
2. **Implement env.py:** GameState, Coordinator, Env, DatasetBuilder
3. **Create train.py:** CLI config and training setup
4. **Test locally:** Verify game logic works
5. **Test integration:** Run on cluster with model
6. **Iterate:** Tune rewards and training parameters
