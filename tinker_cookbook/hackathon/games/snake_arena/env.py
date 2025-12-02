"""
Snake Arena: Multi-agent Snake Battle Royale Environment

A competitive multi-agent environment where 4-8 snake agents compete on a shared
grid. All snakes move simultaneously each turn, creating interesting dynamics
around collision prediction and strategic positioning.

Game Rules:
- Snakes move in one of four directions: up, down, left, right
- Eating food makes the snake grow by 1 segment
- Colliding with walls, own body, or other snakes causes death
- Head-to-head collisions kill both snakes
- Last snake standing wins (or highest score if time limit reached)

Rewards:
- +0.1 per step survived
- +1.0 for eating food
- -1.0 for dying
- +2.0 bonus for being last snake alive
"""

import asyncio
import json
import random
from dataclasses import dataclass, field
from typing import Sequence

import chz
import numpy as np
from tinker import types
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.renderers import Renderer, get_renderer
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    Observation,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

# Action parsing stop condition - look for JSON closing brace
STOP_CONDITION = ["}\n", "}\n\n"]

# Rewards
SURVIVAL_REWARD = 0.1
FOOD_REWARD = 1.0
DEATH_REWARD = -1.0
WINNER_BONUS = 2.0
INVALID_ACTION_REWARD = -0.5

# Directions
DIRECTIONS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

# Opposite directions (can't reverse)
OPPOSITE = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}


class SnakeGameState:
    """Manages the shared game state for all snakes on the grid."""

    def __init__(
        self,
        grid_size: int = 20,
        num_players: int = 4,
        num_food: int = 3,
        max_steps: int = 200,
        seed: int | None = None,
    ):
        self.grid_size = grid_size
        self.num_players = num_players
        self.num_food = num_food
        self.max_steps = max_steps
        self.rng = random.Random(seed)

        # Initialize snakes at different positions
        self.snakes: dict[int, list[tuple[int, int]]] = {}
        self.directions: dict[int, str] = {}
        self.alive: dict[int, bool] = {}
        self.scores: dict[int, int] = {}

        self._initialize_snakes()
        self.food_positions: list[tuple[int, int]] = []
        self._spawn_food(self.num_food)

        self.current_step = 0
        self.done = False
        self.pending_moves: dict[int, str] = {}

    def _initialize_snakes(self) -> None:
        """Place snakes at starting positions around the grid."""
        # Starting positions in corners and edges
        start_positions = [
            (2, 2, "right"),
            (2, self.grid_size - 3, "left"),
            (self.grid_size - 3, 2, "right"),
            (self.grid_size - 3, self.grid_size - 3, "left"),
            (self.grid_size // 2, 2, "right"),
            (self.grid_size // 2, self.grid_size - 3, "left"),
            (2, self.grid_size // 2, "down"),
            (self.grid_size - 3, self.grid_size // 2, "up"),
        ]

        for player_id in range(self.num_players):
            row, col, direction = start_positions[player_id % len(start_positions)]
            # Create snake with 3 segments (head + 2 body)
            dr, dc = DIRECTIONS[OPPOSITE[direction]]
            snake = [
                (row, col),
                (row + dr, col + dc),
                (row + 2 * dr, col + 2 * dc),
            ]
            self.snakes[player_id] = snake
            self.directions[player_id] = direction
            self.alive[player_id] = True
            self.scores[player_id] = 0

    def _spawn_food(self, count: int) -> None:
        """Spawn food at random empty positions."""
        occupied = set()
        for snake in self.snakes.values():
            occupied.update(snake)
        occupied.update(self.food_positions)

        empty_positions = [
            (r, c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
            if (r, c) not in occupied
        ]

        if empty_positions:
            new_food = self.rng.sample(empty_positions, min(count, len(empty_positions)))
            self.food_positions.extend(new_food)

    def get_valid_moves(self, player_id: int) -> list[str]:
        """Get valid moves for a player (can't reverse direction)."""
        if not self.alive[player_id]:
            return []
        current_dir = self.directions[player_id]
        return [d for d in DIRECTIONS.keys() if d != OPPOSITE[current_dir]]

    def get_observation(self, player_id: int) -> dict:
        """Get the observation for a specific player."""
        snake = self.snakes[player_id]
        other_snakes = []
        for pid, s in self.snakes.items():
            if pid != player_id:
                other_snakes.append({
                    "id": pid,
                    "body": s,
                    "alive": self.alive[pid],
                })

        return {
            "grid_size": self.grid_size,
            "your_id": player_id,
            "your_snake": snake,
            "your_direction": self.directions[player_id],
            "your_length": len(snake),
            "your_score": self.scores[player_id],
            "other_snakes": other_snakes,
            "food": self.food_positions,
            "turn": self.current_step,
            "max_turns": self.max_steps,
            "valid_moves": self.get_valid_moves(player_id),
            "alive_count": sum(1 for a in self.alive.values() if a),
        }

    def submit_move(self, player_id: int, direction: str) -> None:
        """Submit a move for a player. Moves are processed simultaneously."""
        if self.alive[player_id] and direction in DIRECTIONS:
            # Validate move (can't reverse)
            if direction != OPPOSITE[self.directions[player_id]]:
                self.pending_moves[player_id] = direction
            else:
                # Invalid move - keep current direction
                self.pending_moves[player_id] = self.directions[player_id]

    def process_moves(self) -> dict[int, float]:
        """Process all pending moves simultaneously and return rewards."""
        if self.done:
            return {pid: 0.0 for pid in range(self.num_players)}

        rewards: dict[int, float] = {pid: 0.0 for pid in range(self.num_players)}

        # Calculate new head positions
        new_heads: dict[int, tuple[int, int]] = {}
        for player_id in range(self.num_players):
            if not self.alive[player_id]:
                continue

            direction = self.pending_moves.get(player_id, self.directions[player_id])
            self.directions[player_id] = direction

            head = self.snakes[player_id][0]
            dr, dc = DIRECTIONS[direction]
            new_head = (head[0] + dr, head[1] + dc)
            new_heads[player_id] = new_head

        # Check for wall collisions
        dead_this_turn: set[int] = set()
        for player_id, new_head in new_heads.items():
            r, c = new_head
            if r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size:
                dead_this_turn.add(player_id)

        # Check for head-to-head collisions
        head_positions: dict[tuple[int, int], list[int]] = {}
        for player_id, new_head in new_heads.items():
            if player_id not in dead_this_turn:
                if new_head not in head_positions:
                    head_positions[new_head] = []
                head_positions[new_head].append(player_id)

        for pos, players in head_positions.items():
            if len(players) > 1:
                dead_this_turn.update(players)

        # Check for body collisions (with self or others)
        all_body_positions: set[tuple[int, int]] = set()
        for player_id, snake in self.snakes.items():
            if self.alive[player_id]:
                # Include body (not head, as head will move)
                all_body_positions.update(snake[:-1])  # Exclude tail (will move)

        for player_id, new_head in new_heads.items():
            if player_id not in dead_this_turn:
                if new_head in all_body_positions:
                    dead_this_turn.add(player_id)

        # Process deaths
        for player_id in dead_this_turn:
            self.alive[player_id] = False
            rewards[player_id] = DEATH_REWARD

        # Move surviving snakes
        food_eaten: set[tuple[int, int]] = set()
        for player_id, new_head in new_heads.items():
            if player_id not in dead_this_turn:
                snake = self.snakes[player_id]

                # Check if eating food
                ate_food = new_head in self.food_positions
                if ate_food:
                    food_eaten.add(new_head)
                    rewards[player_id] += FOOD_REWARD
                    self.scores[player_id] += 1
                    # Don't remove tail (snake grows)
                    snake.insert(0, new_head)
                else:
                    # Normal move - add head, remove tail
                    snake.insert(0, new_head)
                    snake.pop()
                    rewards[player_id] += SURVIVAL_REWARD

        # Remove eaten food and spawn new
        for food in food_eaten:
            self.food_positions.remove(food)
        self._spawn_food(len(food_eaten))

        # Clear pending moves
        self.pending_moves.clear()
        self.current_step += 1

        # Check for game end
        alive_players = [pid for pid in range(self.num_players) if self.alive[pid]]
        if len(alive_players) <= 1 or self.current_step >= self.max_steps:
            self.done = True
            if len(alive_players) == 1:
                rewards[alive_players[0]] += WINNER_BONUS

        return rewards


class MultiPlayerCoordinator:
    """Coordinates simultaneous moves across multiple players.

    Unlike TwoPlayerCoordinator which handles sequential turns, this coordinator
    collects moves from all players before processing them simultaneously.
    """

    def __init__(self, game_state: SnakeGameState):
        self.game_state = game_state
        self.condition = asyncio.Condition()
        self.moves_submitted: set[int] = set()
        self.last_rewards: dict[int, float] = {}
        self.processing = False

    @property
    def done(self) -> bool:
        return self.game_state.done

    @property
    def num_players(self) -> int:
        return self.game_state.num_players

    def get_alive_players(self) -> list[int]:
        return [pid for pid in range(self.num_players) if self.game_state.alive[pid]]

    async def submit_move(self, player_id: int, direction: str) -> float:
        """Submit a move and wait for all players to move. Returns reward."""
        async with self.condition:
            if not self.game_state.alive[player_id]:
                return 0.0

            # Submit this player's move
            self.game_state.submit_move(player_id, direction)
            self.moves_submitted.add(player_id)

            # Check if all alive players have submitted
            alive_players = set(self.get_alive_players())
            all_submitted = self.moves_submitted >= alive_players

            if all_submitted and not self.processing:
                # This player triggers the processing
                self.processing = True
                self.last_rewards = self.game_state.process_moves()
                self.moves_submitted.clear()
                self.processing = False
                self.condition.notify_all()
            else:
                # Wait for processing to complete
                await self.condition.wait_for(
                    lambda: player_id in self.last_rewards or self.done
                )

            return self.last_rewards.get(player_id, 0.0)

    async def wait_for_game_ready(self, player_id: int) -> None:
        """Wait until it's time for this player to submit a move."""
        async with self.condition:
            # If game is done or player is dead, return immediately
            if self.done or not self.game_state.alive[player_id]:
                return
            # Otherwise, player can submit move immediately (simultaneous turns)


@dataclass
class SnakeEnv(Env):
    """Environment for a single snake player in the arena."""

    player_id: int
    coordinator: MultiPlayerCoordinator
    renderer: Renderer

    @property
    def stop_condition(self) -> StopCondition:
        return STOP_CONDITION

    def _format_observation(self) -> str:
        """Format the game state as a prompt for the LLM."""
        obs = self.coordinator.game_state.get_observation(self.player_id)

        prompt = f"""You are Snake #{obs['your_id']} in a {obs['grid_size']}x{obs['grid_size']} Snake Battle Arena.

CURRENT STATE (Turn {obs['turn']}/{obs['max_turns']}):
- Your position (head first): {obs['your_snake']}
- Your length: {obs['your_length']}, Score: {obs['your_score']}
- Current direction: {obs['your_direction']}
- Food locations: {obs['food']}
- Snakes alive: {obs['alive_count']}/{self.coordinator.num_players}

OTHER SNAKES:
"""
        for other in obs['other_snakes']:
            status = "ALIVE" if other['alive'] else "DEAD"
            prompt += f"- Snake #{other['id']}: {status}, body: {other['body']}\n"

        prompt += f"""
VALID MOVES: {obs['valid_moves']}

Respond with a JSON object containing your move:
{{"direction": "up"}}  or  {{"direction": "down"}}  or  {{"direction": "left"}}  or  {{"direction": "right"}}

Choose wisely - eat food to grow, avoid walls and other snakes!
"""
        return prompt

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        """Get the initial observation for this player."""
        await self.coordinator.wait_for_game_ready(self.player_id)
        prompt = self._format_observation()
        return self.renderer.build_generation_prompt([{"role": "user", "content": prompt}]), self.stop_condition

    async def step(self, action: Action) -> StepResult:
        """Process the player's action and return the result."""
        if self.coordinator.done or not self.coordinator.game_state.alive[self.player_id]:
            return self._get_done_step()

        # Parse the action from LLM output
        direction = self._parse_action(action)

        # Submit move and wait for all players
        reward = await self.coordinator.submit_move(self.player_id, direction)

        # Check if game is done or player is dead
        is_done = self.coordinator.done or not self.coordinator.game_state.alive[self.player_id]

        if is_done:
            return StepResult(
                reward=reward,
                episode_done=True,
                next_observation=types.ModelInput.empty(),
                next_stop_condition=self.stop_condition,
                metrics=self._get_metrics(),
            )

        # Get next observation
        prompt = self._format_observation()
        next_obs = self.renderer.build_generation_prompt([{"role": "user", "content": prompt}])

        return StepResult(
            reward=reward,
            episode_done=False,
            next_observation=next_obs,
            next_stop_condition=self.stop_condition,
            metrics=self._get_metrics(),
        )

    def _parse_action(self, action: Action) -> str:
        """Parse the LLM output to extract the direction."""
        try:
            # Get text from action
            message = self.renderer.parse_response(action)[0]
            text = message["content"].strip()

            # Try to parse as JSON
            # Find JSON object in text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = text[start:end]
                data = json.loads(json_str)
                direction = data.get("direction", "").lower()
                if direction in DIRECTIONS:
                    return direction
        except (json.JSONDecodeError, KeyError, IndexError):
            pass

        # Default to current direction if parsing fails
        return self.coordinator.game_state.directions[self.player_id]

    def _get_done_step(self) -> StepResult:
        """Return a done step result."""
        return StepResult(
            reward=0.0,
            episode_done=True,
            next_observation=types.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics=self._get_metrics(),
        )

    def _get_metrics(self) -> dict[str, float]:
        """Get metrics for this player."""
        gs = self.coordinator.game_state
        return {
            "snake/length": float(len(gs.snakes[self.player_id])),
            "snake/score": float(gs.scores[self.player_id]),
            "snake/alive": float(gs.alive[self.player_id]),
            "game/turn": float(gs.current_step),
            "game/alive_count": float(sum(1 for a in gs.alive.values() if a)),
        }


@dataclass
class SnakeEnvGroupBuilder(EnvGroupBuilder):
    """Builder for a group of snake environments sharing one game."""

    renderer: Renderer
    num_players: int = 4
    grid_size: int = 20
    num_food: int = 3
    max_steps: int = 200
    seed: int | None = None

    async def make_envs(self) -> Sequence[Env]:
        """Create all player environments for one game."""
        game_state = SnakeGameState(
            grid_size=self.grid_size,
            num_players=self.num_players,
            num_food=self.num_food,
            max_steps=self.max_steps,
            seed=self.seed,
        )
        coordinator = MultiPlayerCoordinator(game_state)

        return [
            SnakeEnv(
                player_id=player_id,
                coordinator=coordinator,
                renderer=self.renderer,
            )
            for player_id in range(self.num_players)
        ]


class SnakeArenaDataset(RLDataset):
    """Dataset for Snake Arena games."""

    def __init__(
        self,
        batch_size: int,
        builder: SnakeEnvGroupBuilder,
        num_games: int,
    ):
        self.batch_size = batch_size
        self.builder = builder
        self.num_games = num_games
        self.num_players = builder.num_players

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        """Get a batch of game builders."""
        # Each game produces num_players environments
        games_per_batch = self.batch_size // self.num_players
        return [self.builder for _ in range(games_per_batch)]

    def __len__(self) -> int:
        return self.num_games


@chz.chz
class SnakeArenaDatasetBuilder(RLDatasetBuilder):
    """Dataset builder for Snake Arena with CLI configuration."""

    # Game settings
    num_players: int = 4
    grid_size: int = 20
    num_food: int = 3
    max_steps: int = 200

    # Training settings
    batch_size: int = 8  # Should be divisible by num_players
    num_train_games: int = 1000
    num_test_games: int = 100

    # Model settings
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    renderer_name: str = "qwen3"
    base_url: str | None = None

    async def __call__(self) -> tuple[SnakeArenaDataset, SnakeArenaDataset | None]:
        """Build training and test datasets."""
        renderer = get_renderer(self.renderer_name, get_tokenizer(self.model_name))

        train_builder = SnakeEnvGroupBuilder(
            renderer=renderer,
            num_players=self.num_players,
            grid_size=self.grid_size,
            num_food=self.num_food,
            max_steps=self.max_steps,
        )

        train_dataset = SnakeArenaDataset(
            batch_size=self.batch_size,
            builder=train_builder,
            num_games=self.num_train_games,
        )

        test_builder = SnakeEnvGroupBuilder(
            renderer=renderer,
            num_players=self.num_players,
            grid_size=self.grid_size,
            num_food=self.num_food,
            max_steps=self.max_steps,
            seed=42,  # Fixed seed for reproducible testing
        )

        test_dataset = SnakeArenaDataset(
            batch_size=self.batch_size,
            builder=test_builder,
            num_games=self.num_test_games,
        )

        return train_dataset, test_dataset
