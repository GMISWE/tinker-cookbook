# Miles x Tinker Architecture

This document explains the system architecture powering the hackathon games.

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Hackathon Participant                        │
│                    (Your Agent Code)                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Tinker Cookbook                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  Snake Arena    │  │  Poker Arena    │  │  Other Games    │  │
│  │  (env.py)       │  │  (env.py)       │  │                 │  │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │
│           └────────────────────┼────────────────────┘           │
│                               ▼                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              RL Training Loop (rl/train.py)                 ││
│  │  - Rollouts: Sample from policy                             ││
│  │  - Rewards: Compute advantages                              ││
│  │  - Training: Update model weights                           ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Tinker API Client                            │
│  - ServiceClient: Connection management                         │
│  - TrainingClient: forward_backward(), optim_step()             │
│  - SamplingClient: sample() for inference                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    kgateway (API Gateway)                        │
│  - Translates Tinker API to Slime/Miles format                  │
│  - Manages model lifecycle                                       │
│  - Routes requests to Ray actors                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Miles/Slime Backend                          │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                 Ray Cluster                                  ││
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       ││
│  │  │MegatronActor │  │MegatronActor │  │MegatronActor │  ...  ││
│  │  │   (DP=0)     │  │   (DP=1)     │  │   (PP=1)     │       ││
│  │  └──────────────┘  └──────────────┘  └──────────────┘       ││
│  │           │                │                │                ││
│  │  ┌────────┴────────────────┴────────────────┴─────────┐     ││
│  │  │              Megatron-LM (Distributed Training)     │     ││
│  │  └────────────────────────────────────────────────────┘     ││
│  │                                                              ││
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       ││
│  │  │ SGLangEngine │  │ SGLangEngine │  │ SGLangEngine │  ...  ││
│  │  │  (Sampling)  │  │  (Sampling)  │  │  (Sampling)  │       ││
│  │  └──────────────┘  └──────────────┘  └──────────────┘       ││
│  └─────────────────────────────────────────────────────────────┘│
│                            │                                     │
│  ┌─────────────────────────┴───────────────────────────────────┐│
│  │                  H200 GPUs (8x 143GB)                       ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### Tinker Cookbook (Your Code)

The hackathon games are built on Tinker Cookbook abstractions:

```python
# Environment Interface
class Env(ABC):
    async def initial_observation(self) -> tuple[Observation, StopCondition]
    async def step(self, action: Action) -> StepResult

# Dataset Builder
class RLDatasetBuilder(ABC):
    async def __call__(self) -> tuple[RLDataset, RLDataset | None]

# Training Loop
async def train.main(config: Config):
    # 1. Create training/sampling clients
    # 2. For each iteration:
    #    - Sample from policy (rollouts)
    #    - Compute rewards and advantages
    #    - Update model weights
```

### Tinker API

The Tinker API provides three main primitives:

```python
# 1. Create clients
service_client = tinker.ServiceClient(base_url=base_url)
training_client = service_client.create_lora_training_client(
    base_model="Qwen/Qwen2.5-0.5B-Instruct",
    rank=32,
)

# 2. Training step
fwd_bwd = training_client.forward_backward(batch, loss_fn="importance_sampling")
training_client.optim_step(adam_params)

# 3. Sampling
sampling_client = training_client.save_weights_and_get_sampling_client("policy")
response = sampling_client.sample(prompt, sampling_params)
```

### kgateway (API Gateway)

The kgateway translates Tinker API calls to the Slime/Miles backend:

| Tinker API | kgateway Route | Backend Action |
|------------|----------------|----------------|
| `create_training_client()` | `POST /api/v1/models` | Create Ray actors |
| `forward_backward()` | `POST /api/v1/train` | Distributed forward/backward |
| `optim_step()` | `POST /api/v1/optim` | Apply gradients |
| `sample()` | `POST /api/v1/sample` | SGLang inference |

### Miles/Slime Backend

The backend uses Ray for distributed training:

- **MegatronTrainRayActor:** Handles training (DP=2, PP=2)
- **SGLangEngine:** Handles fast sampling
- **RolloutManager:** Coordinates sampling actors

## Multi-Agent Coordination

### Simultaneous Moves (Snake Arena)

```
Turn N:
  ┌─────────────────────────────────────────────────┐
  │  All 4 snakes receive observations              │
  │  All 4 snakes submit moves (in parallel)        │
  │  Coordinator waits for all moves                │
  │  Coordinator processes moves together           │
  │  Collisions resolved, rewards computed          │
  │  All 4 snakes receive new observations          │
  └─────────────────────────────────────────────────┘
Turn N+1:
  ...
```

**Implementation:**
```python
class MultiPlayerCoordinator:
    async def submit_move(self, player_id, direction):
        game_state.submit_move(player_id, direction)
        if all_players_submitted():
            rewards = game_state.process_moves()
            notify_all_players()
        else:
            await wait_for_processing()
        return rewards[player_id]
```

### Sequential Turns (Poker Arena)

```
Betting Round:
  ┌─────────────────────────────────────────────────┐
  │  Player 0 receives observation (their turn)     │
  │  Player 0 submits action (fold/call/raise)      │
  │  Coordinator processes action                    │
  │  Player 1 receives observation (their turn)     │
  │  Player 1 submits action                        │
  │  ...                                            │
  │  Round ends when bets are matched               │
  └─────────────────────────────────────────────────┘
```

**Implementation:**
```python
class PokerRoundCoordinator:
    async def wait_for_turn(self, player_id):
        async with condition:
            await condition.wait_for(
                lambda: current_player == player_id or done
            )

    async def take_action(self, player_id, action, amount):
        game_state.apply_action(player_id, action, amount)
        condition.notify_all()
```

## Data Flow

### Training Step

```
1. Environment generates observations
   └─> JSON state (grid, cards, etc.)

2. Renderer converts to tokens
   └─> [token_1, token_2, ..., token_n]

3. Sampling client generates response
   └─> Model input → SGLang → tokens + logprobs

4. Environment parses action
   └─> JSON {"direction": "up"} or {"action": "raise", "amount": 50}

5. Environment computes reward
   └─> +1.0 for food, -1.0 for death, etc.

6. Training loop computes advantages
   └─> advantage = reward - baseline

7. Training client updates weights
   └─> forward_backward() → optim_step()
```

### Memory Layout (H200 GPUs)

```
┌────────────────────────────────────────┐
│  GPU Memory (143 GB per GPU)           │
├────────────────────────────────────────┤
│  Model Weights (LoRA)     ~2 GB        │
│  Megatron Overhead        ~10 GB       │
│  SGLang KV Cache          ~50-100 GB   │
│  Activations              ~20-40 GB    │
│  Available                ~10-50 GB    │
└────────────────────────────────────────┘

Key constraints:
- max_tokens affects KV cache size
- batch_size affects activation memory
- Keep max_tokens ≤ 128 for safety
```

## Latency Breakdown

| Operation | Typical Latency |
|-----------|-----------------|
| Observation formatting | < 1ms |
| Token encoding | < 5ms |
| SGLang inference | 50-200ms |
| Action parsing | < 1ms |
| Game logic | < 1ms |
| Reward computation | < 1ms |
| **Total per turn** | **~100-250ms** |

For real-time games with 500ms turn timers, this leaves margin for:
- Network latency
- Multiple inference calls (lookahead)
- Complex game logic

## Optimization Opportunities (Track A)

### Operator Fusion
```python
# Before: Multiple small kernels
attention_output = attention(q, k, v)
normalized = layer_norm(attention_output)
output = feed_forward(normalized)

# After: Fused kernel
output = fused_transformer_block(q, k, v)
```

### Memory Mapping
```python
# Before: Copy between CPU and GPU
data = data.to('cuda')
result = model(data)
result = result.to('cpu')

# After: Direct memory access
with torch.cuda.stream(stream):
    result = model(data)  # Async, no copy
```

### Edge Cases
- Long context windows (> 4K tokens)
- Nested JSON schemas
- Batch size mismatches
- Gradient accumulation

## Summary

The Miles x Tinker architecture enables:

1. **Structured Logic:** Define game strategies in Python
2. **Fast Inference:** Execute on distributed GPUs
3. **Multi-Agent:** Coordinate simultaneous or sequential play
4. **RL Training:** Learn from self-play and rewards

For Track A, focus on the kgateway → Slime translation layer.
For Track B, focus on the Env implementation and game logic.
