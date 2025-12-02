# Poker Arena: LLM Texas Hold'em

A multi-agent poker environment where 6 LLM agents play No-Limit Texas Hold'em. Agents must make strategic betting decisions with imperfect information (hidden opponent cards).

## Game Overview

### Objective
Win chips by either having the best hand at showdown or making all opponents fold.

### Rules
- **Variant:** No-Limit Texas Hold'em
- **Players:** 6 agents
- **Starting Stack:** 100 big blinds (200 chips)
- **Blinds:** Small blind 1, Big blind 2
- **Betting Rounds:** Preflop, Flop, Turn, River
- **Hand Rankings:** Standard poker (High Card to Royal Flush)

### Rewards
Rewards are chips won/lost normalized by the big blind:
- Win 50 chips → Reward = +25.0
- Lose 30 chips → Reward = -15.0

## Architecture

### Miles Role (The Brain)
The LLM uses Miles to define **betting strategy**:
- "If hand strength > 80%, raise aggressively"
- "If pot odds > implied odds, fold"
- "If opponent shows weakness (check), bluff"

### Tinker Role (The Muscle)
Tinker provides **fast inference** for:
- Pot odds calculation
- Decision making within 500ms turn timer
- Multi-round betting sequences

## Observation Format

Each turn, the agent receives a JSON observation:

```json
{
  "your_id": 2,
  "your_hole_cards": ["Ah", "Ks"],
  "community_cards": ["Qd", "Jc", "Th"],
  "pot": 45,
  "current_bet": 12,
  "your_stack": 88,
  "your_bet_this_round": 6,
  "to_call": 6,
  "players": [
    {"id": 0, "stack": 94, "bet": 12, "folded": false},
    {"id": 1, "stack": 0, "bet": 2, "folded": true},
    {"id": 2, "stack": 88, "bet": 6, "folded": false, "is_you": true, "to_act": true}
  ],
  "betting_round": "flop",
  "dealer_position": 0,
  "min_raise": 18,
  "max_raise": 88,
  "valid_actions": ["fold", "call", "raise"],
  "active_players": 4
}
```

### Key Fields
- `your_hole_cards`: Your two private cards (e.g., "Ah" = Ace of hearts)
- `community_cards`: Shared cards (0-5 depending on round)
- `to_call`: Chips needed to stay in hand
- `min_raise` / `max_raise`: Valid raise range
- `betting_round`: Current phase (preflop/flop/turn/river)

## Action Format

Respond with a JSON object:

```json
{"action": "fold"}
{"action": "check"}
{"action": "call"}
{"action": "raise", "amount": 50}
```

### Valid Actions
- **fold:** Give up your hand
- **check:** Pass (only if no bet to call)
- **call:** Match the current bet
- **raise:** Increase the bet (specify amount)

## Card Notation

Cards are represented as rank + suit:
- **Ranks:** 2-9, T (10), J, Q, K, A
- **Suits:** h (hearts), d (diamonds), c (clubs), s (spades)

Examples: `Ah` (Ace of hearts), `Tc` (Ten of clubs), `2d` (Two of diamonds)

## Hand Rankings

From lowest to highest:
1. High Card
2. Pair
3. Two Pair
4. Three of a Kind
5. Straight
6. Flush
7. Full House
8. Four of a Kind
9. Straight Flush
10. Royal Flush

## Training

### Basic Training

```bash
python -m tinker_cookbook.hackathon.games.poker_arena.train \
    model_name=/data/models/Qwen2.5-0.5B-Instruct \
    renderer_name=qwen3 \
    base_url=http://kgateway-training.miles-gmi-tinker:8000 \
    num_players=6 \
    batch_size=12 \
    learning_rate=3e-5 \
    max_tokens=128 \
    log_path=/tmp/poker-arena-test
```

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_players` | 6 | Players per table |
| `starting_stack` | 200 | Starting chips (100 BB) |
| `small_blind` | 1 | Small blind amount |
| `big_blind` | 2 | Big blind amount |
| `batch_size` | 12 | Training batch size |
| `learning_rate` | 3e-5 | Adam learning rate |
| `max_tokens` | 128 | Max tokens per response |

## Implementation Details

### Key Classes

```python
class Card:
    """Playing card with rank and suit."""

class PokerGameState:
    """Manages one poker hand."""
    # Deck, hole cards, community, stacks, pot

class PokerRoundCoordinator:
    """Coordinates sequential betting."""
    # Players take turns, one at a time

class PokerEnv(Env):
    """Single player's view."""
    # Observation formatting, action parsing

class PokerArenaDatasetBuilder:
    """CLI-configurable dataset builder."""
```

### Coordination Pattern

Poker uses **sequential coordination** (unlike Snake's simultaneous moves):

1. Player receives observation (their turn)
2. Player submits action (fold/check/call/raise)
3. Coordinator processes action, updates state
4. Next player's turn (or next betting round)
5. Repeat until hand ends

```python
async def wait_for_turn(player_id):
    async with condition:
        await condition.wait_for(
            lambda: current_player == player_id or done
        )

async def take_action(player_id, action, amount):
    async with condition:
        game_state.apply_action(player_id, action, amount)
        condition.notify_all()
```

## Expected Training Results

### Metrics to Monitor
- `poker/stack`: Ending stack (chips)
- `poker/won`: Win rate
- `poker/folded`: Fold rate
- `reward/mean`: Average chips won/lost

### Learning Progression
1. **Early:** Random play, loses to blinds
2. **Mid:** Basic hand strength awareness
3. **Late:** Position play, pot odds, bluffing

## Tips for Participants

### Strategy Ideas
1. **Tight-Aggressive:** Play few hands, bet big
2. **Loose-Passive:** Play many hands, check/call
3. **Position Aware:** Play more hands in late position
4. **Pot Odds:** Calculate if call is profitable

### Poker Concepts
- **Position:** Later = better (more info)
- **Pot Odds:** Chips to call vs. pot size
- **Hand Ranges:** Estimate opponent's possible hands
- **Bluffing:** Bet with weak hand to win pot

### Common Mistakes
- Playing too many hands preflop
- Ignoring position
- Not adjusting to opponents
- Betting without a plan

## Files

```
poker_arena/
├── __init__.py      # Package exports
├── env.py           # Environment implementation
├── train.py         # Training CLI
└── README.md        # This file
```
