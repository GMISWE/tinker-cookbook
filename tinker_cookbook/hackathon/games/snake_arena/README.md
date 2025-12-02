# Snake Arena: Multi-Agent Snake Battle Royale

A competitive multi-agent environment where 4-8 snake agents compete on a shared 2D grid. All snakes move simultaneously each turn, creating dynamic gameplay around collision prediction and strategic positioning.

## Game Overview

### Objective
Survive longer than other snakes while growing by eating food. The last snake standing wins.

### Rules
- **Movement:** Snakes move in one of four directions (up, down, left, right)
- **Growth:** Eating food makes the snake grow by 1 segment
- **Death:** Colliding with walls, own body, or other snakes causes death
- **Head-to-head:** If two snake heads collide, both die
- **Turns:** All snakes move simultaneously (not turn-based)

### Rewards
| Event | Reward |
|-------|--------|
| Survive one step | +0.1 |
| Eat food | +1.0 |
| Die (collision) | -1.0 |
| Last snake alive | +2.0 bonus |

## Architecture

### Miles Role (The Brain)
The LLM uses Miles to define **survival logic**:
- "If head position near wall, turn away"
- "If food is closer than enemy, go for food"
- "If enemy is approaching, predict their move and evade"

### Tinker Role (The Muscle)
Tinker provides **fast inference** for:
- Real-time decision making (all players move simultaneously)
- Lookahead simulation (predict 5-10 moves ahead)
- Trap prediction (identify dangerous positions)

## Observation Format

Each turn, the agent receives a JSON observation:

```json
{
  "grid_size": 20,
  "your_id": 0,
  "your_snake": [[10, 10], [10, 9], [10, 8]],
  "your_direction": "right",
  "your_length": 3,
  "your_score": 2,
  "other_snakes": [
    {"id": 1, "body": [[5, 5], [5, 4]], "alive": true},
    {"id": 2, "body": [[15, 15]], "alive": false}
  ],
  "food": [[3, 7], [12, 18]],
  "turn": 45,
  "max_turns": 200,
  "valid_moves": ["up", "down", "right"],
  "alive_count": 2
}
```

### Key Fields
- `your_snake`: List of (row, col) positions, head first
- `your_direction`: Current direction (can't reverse)
- `other_snakes`: All other snakes with positions and alive status
- `food`: List of food positions
- `valid_moves`: Moves that won't reverse direction

## Action Format

Respond with a JSON object:

```json
{"direction": "up"}
```

Valid directions: `"up"`, `"down"`, `"left"`, `"right"`

**Note:** You cannot reverse direction (e.g., can't go "left" if currently going "right").

## Training

### Basic Training

```bash
python -m tinker_cookbook.hackathon.games.snake_arena.train \
    model_name=/data/models/Qwen2.5-0.5B-Instruct \
    renderer_name=qwen3 \
    base_url=http://kgateway-training.miles-gmi-tinker:8000 \
    num_players=4 \
    batch_size=8 \
    learning_rate=3e-5 \
    max_tokens=64 \
    log_path=/tmp/snake-arena-test
```

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_players` | 4 | Number of snakes (4-8) |
| `grid_size` | 20 | Grid dimensions (20x20) |
| `num_food` | 3 | Food items on grid |
| `max_steps` | 200 | Maximum turns per game |
| `batch_size` | 8 | Training batch size |
| `learning_rate` | 3e-5 | Adam learning rate |
| `max_tokens` | 64 | Max tokens per response |

## Implementation Details

### Key Classes

```python
class SnakeGameState:
    """Manages shared game state for all snakes."""
    # Grid, snake positions, food, scores

class MultiPlayerCoordinator:
    """Coordinates simultaneous moves across players."""
    # Collects all moves, processes together

class SnakeEnv(Env):
    """Single player's view of the game."""
    # Observation formatting, action parsing

class SnakeArenaDatasetBuilder:
    """CLI-configurable dataset builder."""
    # Creates training/test datasets
```

### Coordination Pattern

Unlike sequential turn games, Snake Arena uses **simultaneous coordination**:

1. All alive players receive observations
2. All players submit moves in parallel
3. Coordinator collects all moves
4. Moves are processed together (collisions resolved)
5. Rewards distributed, next turn begins

```python
# Pseudocode
async def submit_move(player_id, direction):
    game_state.submit_move(player_id, direction)
    if all_players_submitted():
        rewards = game_state.process_moves()
        notify_all_players()
    else:
        await wait_for_processing()
    return rewards[player_id]
```

## Expected Training Results

### Metrics to Monitor
- `snake/length`: Average snake length (should increase)
- `snake/score`: Food eaten (should increase)
- `game/alive_count`: Snakes surviving (varies with skill)
- `reward/mean`: Average reward per episode

### Learning Progression
1. **Early training:** Random movements, quick deaths
2. **Mid training:** Basic wall avoidance, some food seeking
3. **Late training:** Strategic positioning, opponent prediction

## Tips for Participants

### Strategy Ideas
1. **Greedy:** Always go for nearest food
2. **Defensive:** Prioritize survival over food
3. **Aggressive:** Try to trap other snakes
4. **Predictive:** Simulate opponent moves

### Common Mistakes
- Reversing direction (invalid move)
- Ignoring other snake positions
- Not accounting for simultaneous moves
- Tunnel vision on food (ignoring threats)

## Files

```
snake_arena/
├── __init__.py      # Package exports
├── env.py           # Environment implementation
├── train.py         # Training CLI
└── README.md        # This file
```
