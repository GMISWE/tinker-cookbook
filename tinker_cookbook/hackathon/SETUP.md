# Hackathon Environment Setup

This guide walks you through setting up your environment for the GMI Cloud x SGLang AI Arena Hackathon.

## Prerequisites

- Python 3.10+
- kubectl access to the cluster
- Git

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/thinking-machines/tinker-cookbook.git
cd tinker-cookbook
```

### 2. Install Dependencies

```bash
pip install -e .
```

### 3. Set Environment Variables

```bash
export TINKER_API_KEY=slime-dev-key
export TINKER_BASE_URL=http://kgateway-training.miles-gmi-tinker:8000
```

## Cluster Setup (GMI Cloud)

### Check Pod Status

```bash
kubectl get pods -n miles-gmi-tinker
```

Expected output:
```
NAME                                READY   STATUS    RESTARTS   AGE
kgateway-training-xxx               1/1     Running   0          1d
miles-training-0                    1/1     Running   0          1d
```

### Install Packages on Pod

```bash
# Copy packages to pod
kubectl cp /path/to/tinker_gmi miles-gmi-tinker/miles-training-0:/tmp/tinker_gmi
kubectl cp /path/to/tinker-cookbook miles-gmi-tinker/miles-training-0:/tmp/tinker-cookbook

# Install packages
kubectl exec -n miles-gmi-tinker miles-training-0 -- bash -c "
  cd /tmp/tinker_gmi && pip install -e .
  cd /tmp/tinker-cookbook && pip install -e .
"
```

### Verify Installation

```bash
kubectl exec -n miles-gmi-tinker miles-training-0 -- python -c "
from tinker_cookbook.hackathon.games import snake_arena, poker_arena
print('Snake Arena:', snake_arena)
print('Poker Arena:', poker_arena)
print('Installation verified!')
"
```

## Running Games

### Snake Arena

```bash
# Using test script
./hackathon/scripts/test_snake_arena.sh

# Or directly
kubectl exec -n miles-gmi-tinker miles-training-0 -- bash -c '
export TINKER_API_KEY=slime-dev-key
python -m tinker_cookbook.hackathon.games.snake_arena.train \
    model_name=/data/models/Qwen2.5-0.5B-Instruct \
    renderer_name=qwen3 \
    base_url=http://kgateway-training.miles-gmi-tinker:8000 \
    num_players=4 \
    batch_size=8 \
    log_path=/tmp/snake-test \
    behavior_if_log_dir_exists=delete
'
```

### Poker Arena

```bash
# Using test script
./hackathon/scripts/test_poker_arena.sh

# Or directly
kubectl exec -n miles-gmi-tinker miles-training-0 -- bash -c '
export TINKER_API_KEY=slime-dev-key
python -m tinker_cookbook.hackathon.games.poker_arena.train \
    model_name=/data/models/Qwen2.5-0.5B-Instruct \
    renderer_name=qwen3 \
    base_url=http://kgateway-training.miles-gmi-tinker:8000 \
    num_players=6 \
    batch_size=12 \
    log_path=/tmp/poker-test \
    behavior_if_log_dir_exists=delete
'
```

## Configuration Options

### Common Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | - | Path to model (required) |
| `renderer_name` | `qwen3` | Chat format renderer |
| `base_url` | - | Tinker API endpoint |
| `batch_size` | varies | Training batch size |
| `learning_rate` | `3e-5` | Adam learning rate |
| `max_tokens` | `64` | Max tokens per response |
| `log_path` | `/tmp/...` | Output directory |

### Snake Arena Specific

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_players` | `4` | Number of snakes (4-8) |
| `grid_size` | `20` | Grid dimensions |
| `num_food` | `3` | Food items on grid |
| `max_steps` | `200` | Max turns per game |

### Poker Arena Specific

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_players` | `6` | Players per table |
| `starting_stack` | `200` | Starting chips (100 BB) |
| `small_blind` | `1` | Small blind amount |
| `big_blind` | `2` | Big blind amount |

## Troubleshooting

### Connection Refused

```
Error: Connection refused to kgateway-training
```

**Solution:** Check kgateway pod is running:
```bash
kubectl get pods -n miles-gmi-tinker -l app=kgateway-training
kubectl logs -n miles-gmi-tinker deployment/kgateway-training --tail=50
```

### Module Not Found

```
ModuleNotFoundError: No module named 'tinker_cookbook.hackathon'
```

**Solution:** Reinstall packages:
```bash
kubectl exec -n miles-gmi-tinker miles-training-0 -- bash -c "
  cd /tmp/tinker-cookbook && pip install -e .
"
```

### Ray Actors Not Running

```
Error: No actors found
```

**Solution:** Check Ray cluster status:
```bash
kubectl exec -n miles-gmi-tinker miles-training-0 -- ray list actors
```

If no actors, restart pods:
```bash
kubectl delete pod -n miles-gmi-tinker miles-training-0
kubectl wait --for=condition=Ready pod/miles-training-0 -n miles-gmi-tinker --timeout=300s
```

### Out of Memory

```
CUDA out of memory
```

**Solution:** Reduce batch size or max_tokens:
```bash
batch_size=4  # Reduce from 8
max_tokens=32  # Reduce from 64
```

### Cleanup Between Tests

Always clean up before running new tests:
```bash
kubectl exec -n miles-gmi-tinker miles-training-0 -- python3 /tmp/tinker_gmi/tests_integration/cleanup_test_env.py
```

## Viewing Logs

### Training Logs

```bash
# View live logs
kubectl exec -n miles-gmi-tinker miles-training-0 -- tail -f /tmp/snake-test/logs.log

# View metrics
kubectl exec -n miles-gmi-tinker miles-training-0 -- cat /tmp/snake-test/metrics.jsonl
```

### kgateway Logs

```bash
# Filter out noise
kubectl logs -n miles-gmi-tinker -l app=kgateway-training --tail=-1 | grep -Ev '(408|500|200|polling)'
```

## Next Steps

1. [Read the Architecture Guide](ARCHITECTURE.md) - Understand the system
2. [Create Your Own Game](GAME_DEVELOPMENT_GUIDE.md) - Build something new
3. Start training and iterating!
