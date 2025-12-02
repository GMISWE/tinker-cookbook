# GMI Cloud x SGLang AI Arena Hackathon

**Event Dates:** January 5-16, 2026
**Announcement:** NeurIPS December 2025

Welcome to the GMI Cloud x SGLang AI Arena Hackathon! This event showcases the Miles x Tinker integration through competitive AI game challenges.

## Quick Start

```bash
# 1. Set up environment
export TINKER_API_KEY=your-api-key
export TINKER_BASE_URL=http://your-cluster:8000

# 2. Install packages
pip install -e /path/to/tinker-cookbook

# 3. Run a game
python -m tinker_cookbook.hackathon.games.snake_arena.train \
    model_name=Qwen/Qwen2.5-0.5B-Instruct \
    num_players=4 \
    batch_size=8
```

## Tracks

### Track A: Core Optimization Challenge
*For Systems Engineers & Kernel Hackers*

Optimize the Miles x Tinker compatibility layer:
- **Operator Fusion:** Reduce kernel overhead
- **Memory Mapping:** Optimize data transfer
- **Edge Cases:** Handle long contexts, complex schemas

[See Track A Details →](docs/TRACK_A_OPTIMIZATION.md)

### Track B: AI Game Arena
*For App Developers, Gamers, & Students*

Build AI agents for competitive games:

| Game | Players | Type | Difficulty |
|------|---------|------|------------|
| [Snake Arena](games/snake_arena/) | 4-8 | Simultaneous | Medium |
| [Poker Arena](games/poker_arena/) | 6 | Sequential | Hard |
| [TicTacToe](../tinker_cookbook/recipes/multiplayer_rl/text_arena/) | 2 | Turn-based | Easy |

**The Concept:** "Code the Brain, We Provide the Reflexes"
- **Miles (The Brain):** Define structured logic, state machines, decision trees
- **Tinker (The Muscle):** Ultra-low latency inference on GMI GPUs

## Available Games

### Snake Arena: Multi-Agent Battle Royale
```
┌────────────────────┐
│ . . . F . . . . . .│  F = Food
│ . 1 1 1 . . . . . .│  1-4 = Snakes
│ . . . . . . 2 2 . .│
│ . . . . . . . 2 . .│  Rules:
│ . . 3 . . . . . . .│  - Eat food to grow
│ . . 3 . . F . . . .│  - Avoid collisions
│ . . 3 . . . . . . .│  - Last snake wins
│ . . . . . 4 4 4 . .│
└────────────────────┘
```

**Key Features:**
- 4-8 simultaneous players
- JSON observations and actions
- Rewards: survival, food, victory

[Snake Arena Documentation →](games/snake_arena/README.md)

### Poker Arena: LLM Texas Hold'em
```
┌─────────────────────────────────────┐
│  Community: [Ah] [Kd] [Qc] [?] [?]  │
│  Pot: 150 chips                     │
├─────────────────────────────────────┤
│  Player 0: 180 chips (Dealer)       │
│  Player 1: 150 chips                │
│  Player 2: 120 chips ← Your turn    │
│  Player 3: FOLDED                   │
│  Player 4: 200 chips                │
│  Player 5: 150 chips                │
├─────────────────────────────────────┤
│  Your cards: [Jh] [Ts]              │
│  Valid: fold, call (20), raise      │
└─────────────────────────────────────┘
```

**Key Features:**
- 6 players, imperfect information
- Standard poker betting rounds
- Strategic depth with bluffing

[Poker Arena Documentation →](games/poker_arena/README.md)

## Directory Structure

```
hackathon/
├── README.md                    # This file
├── SETUP.md                     # Environment setup
├── ARCHITECTURE.md              # System architecture
├── GAME_DEVELOPMENT_GUIDE.md    # Create new games
│
├── games/
│   ├── snake_arena/             # Snake Battle Royale
│   │   ├── env.py               # Environment
│   │   ├── train.py             # Training CLI
│   │   └── README.md            # Game docs
│   └── poker_arena/             # Texas Hold'em
│       ├── env.py               # Environment
│       ├── train.py             # Training CLI
│       └── README.md            # Game docs
│
└── scripts/
    ├── test_snake_arena.sh      # Snake test
    ├── test_poker_arena.sh      # Poker test
    └── test_text_arena.sh       # TicTacToe baseline
```

## Getting Started

### For Track B Participants

1. **Choose a game:** Start with Snake (easier) or Poker (harder)
2. **Understand the rules:** Read the game README
3. **Study the observation format:** Know what data you receive
4. **Design your strategy:** How will your agent decide?
5. **Train and iterate:** Run training, analyze metrics

### Training Your Agent

```bash
# Snake Arena (4 players, 20x20 grid)
python -m tinker_cookbook.hackathon.games.snake_arena.train \
    model_name=/data/models/Qwen2.5-0.5B-Instruct \
    num_players=4 \
    learning_rate=3e-5

# Poker Arena (6 players)
python -m tinker_cookbook.hackathon.games.poker_arena.train \
    model_name=/data/models/Qwen2.5-0.5B-Instruct \
    num_players=6 \
    learning_rate=3e-5
```

## Why Miles x Tinker?

### The Challenge
AI game agents need both **intelligence** and **speed**:
- Complex decision logic (if-then-else chains)
- Real-time responses (< 500ms per move)

### The Solution

**Miles (Frontend):** Define game logic in Python
```python
# Example: Snake survival logic
if near_wall(head_position):
    action = turn_away_from_wall()
elif food_nearby(position, food_positions):
    action = move_toward_food()
else:
    action = safe_random_move()
```

**Tinker (Backend):** Execute at GPU speed
- Compiles logic to optimized kernels
- Runs on GMI Cloud H200 GPUs
- Sub-millisecond inference latency

## Resources

- [Setup Guide](SETUP.md) - Environment configuration
- [Architecture](ARCHITECTURE.md) - System design
- [Game Development Guide](GAME_DEVELOPMENT_GUIDE.md) - Create new games
- [Tinker Cookbook](../README.md) - Full documentation

## Support

- **Documentation:** Check the guides in this directory
- **Issues:** Report bugs on GitHub
- **Discord:** Join the GMI Cloud community

---

**Good luck, and may the best agent win!**
