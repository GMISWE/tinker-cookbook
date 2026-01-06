# RLVE Parameter Alignment: GB200 Script vs Tinker-Cookbook

This document captures the parameter alignment between the Miles GB200 RLVE training script and the tinker-cookbook RLVE recipe, including reasoning for each parameter choice.

## Reference Scripts

- **GB200 Script**: `/root/gavin/kgateway/python/ai_extension/training/examples/RLVE/scripts/training/Qwen2.5-7B/rlve-4xGB200.sh`
- **Tinker Script**: `/root/gavin/tinker-cookbook/tests/run_rlve_gb200.sh`
- **Training Module**: `tinker_cookbook.recipes.rlve.train`

## Parameter Comparison

### Batch Size Configuration

| Parameter | GB200 Script | Tinker-Cookbook | Reasoning |
|-----------|--------------|-----------------|-----------|
| `rollout_batch_size` | 128 | N/A (derived) | Number of prompts per rollout batch |
| `n_samples_per_prompt` | 16 | `group_size=16` | Samples generated per prompt for GRPO |
| `global_batch_size` | 128 × 16 = 2048 | `groups_per_batch × group_size = 8 × 16 = 128` | Total samples per training step |

**Reasoning**: Tinker uses smaller batches (128 vs 2048) due to HTTP API overhead. The key ratio `n_samples_per_prompt=16` is preserved for proper GRPO advantage estimation within groups.

### RL Algorithm Parameters

| Parameter | GB200 Script | Tinker-Cookbook | Reasoning |
|-----------|--------------|-----------------|-----------|
| `advantage_estimator` | `grpo` | `grpo` (implicit) | Group Relative Policy Optimization - normalizes advantages within each group |
| `loss_fn` | N/A (uses PPO internally) | `ppo` | PPO loss with clipping for stable updates |
| `eps_clip` | 0.2 | 0.2 (tinker default) | Clips policy ratio to [1-ε, 1+ε] |
| `eps_clip_high` | 0.28 | N/A | Asymmetric clipping (higher bound) - not exposed in tinker |
| `use_tis` | True | N/A | Truncated Importance Sampling - clips π_train/π_rollout before loss |
| `kl_penalty_coef` | 0.0 | 0.02 | KL penalty as soft regularization (see below) |
| `entropy_coef` | 0.0 | 0.0 (default) | No entropy bonus |

#### Why `loss_fn=ppo` + `kl_penalty_coef=0.02`?

The GB200 script uses three regularization mechanisms:
1. **PPO Clipping** (`eps_clip=0.2`): Hard constraint on policy ratio
2. **Asymmetric Clipping** (`eps_clip_high=0.28`): Allows slightly larger positive updates
3. **TIS** (`use_tis=True`): Clips importance ratio before applying to loss

Tinker-cookbook's `importance_sampling` loss has **no clipping** - it's vanilla policy gradient:
```
loss = -ratio * advantage
```

Without regularization, this leads to:
- Unbounded policy updates
- Entropy increasing (policy becoming uniform)
- Reward not improving

**Solution**: Use `loss_fn=ppo` which applies PPO clipping:
```
loss = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
```

Additionally, `kl_penalty_coef=0.02` adds soft KL regularization:
```
loss = policy_loss - β * KL(π_new || π_ref)
```

This serves as a complement to hard clipping:
- **Hard clipping (PPO)**: Prevents large single-step updates
- **Soft KL penalty**: Prevents cumulative drift from reference policy

**Typical values for `kl_penalty_coef`**:
- 0.01 - 0.05: Common range for RLHF/GRPO
- InstructGPT used 0.02
- Too high (>0.1): Policy stuck near reference
- Too low (<0.01): Insufficient regularization

### Generation Parameters

| Parameter | GB200 Script | Tinker-Cookbook | Reasoning |
|-----------|--------------|-----------------|-----------|
| `rollout_max_response_len` | 4096 | `max_tokens=4096` | Max tokens per response |
| `rollout_temperature` | 1.0 | 1.0 (default) | Sampling temperature for exploration |
| `rollout_top_p` | N/A | 0.9 (default) | Nucleus sampling threshold |

### Optimizer Parameters

| Parameter | GB200 Script | Tinker-Cookbook | Reasoning |
|-----------|--------------|-----------------|-----------|
| `lr` | 1e-6 | `learning_rate=1e-6` | Conservative LR for RL fine-tuning |
| `lr_decay_style` | constant | constant (default) | No LR decay during RL |
| `weight_decay` | 0.01 | 0.01 (default) | L2 regularization |
| `optimizer` | adam | adam (default) | Adam optimizer |

### RLVE Environment Configuration

| Parameter | GB200 Script | Tinker-Cookbook | Reasoning |
|-----------|--------------|-----------------|-----------|
| `environment_list` | 16 environments | 16 environments | Default RLVE paper environments |
| `custom_prompt_preprocessor` | TinyZero | TinyZero | Prompt formatting for math problems |
| `answer_marker_type` | `<answer></answer>` | `<answer></answer>` | Answer extraction markers |
| `initial_difficulty` | 0 | 0 (default) | Start with easiest problems |
| `difficulty_sliding_window_size` | 4 | 4 (default) | Window for difficulty adjustment |
| `min_metric_to_increase_difficulty` | 0.9 | 0.9 (default) | 90% accuracy to increase difficulty |

### Parallelism (Server-Side)

| Parameter | GB200 Script | Tinker-Cookbook | Notes |
|-----------|--------------|-----------------|-------|
| `tensor_model_parallel_size` | 2 | 1 (opentinker default) | TP for model sharding |
| `context_parallel_size` | 2 | 1 (opentinker default) | CP for long sequences |
| `pipeline_model_parallel_size` | 1 | 1 | PP disabled |
| `sequence_parallel` | True | False | SP for memory efficiency |

**Note**: Parallelism is configured in tinkercloud `slime_builder.py`, not in tinker-cookbook.

## 16 Default RLVE Environments

```
Division
EuclidGame
GCDOne_Counting
HamiltonianPath
LampChanging
LargestConvexPolygon
Multiplication
PCPPermutation
Path_NoGoingBack_Counting
SAT
ShortestPath
Sorting
SpiralMatrix
SubsequenceReversalLNDS
UndamagedSubmatrixCounting
WYRLevelingGround
```

## Current Tinker Script Configuration

```bash
python -m tinker_cookbook.recipes.rlve.train \
    model_name="Qwen/Qwen2.5-7B-Instruct" \
    base_url="http://localhost:8000" \
    groups_per_batch=8 \
    group_size=16 \
    max_tokens=4096 \
    learning_rate=1e-6 \
    loss_fn=ppo \
    kl_penalty_coef=0.02 \
    log_path="/data/logs/rlve-tinker" \
    wandb_project=rlve-qwen \
    wandb_name="tinker-7B-4xH200" \
    behavior_if_log_dir_exists=delete \
    eval_every=0 \
    save_every=0 \
    n_batches=1000
```

## Key Differences and Trade-offs

### What Tinker-Cookbook Cannot Control

1. **TIS (Truncated Importance Sampling)**: Implemented in Miles, not exposed via API
2. **Asymmetric Clipping** (`eps_clip_high`): Not available in tinker
3. **Parallelism Settings**: Configured server-side in tinkercloud
4. **Over-sampling** (`over_sampling_batch_size`): Miles-internal optimization

### Compensating Mechanisms

| Missing Feature | Compensation |
|-----------------|--------------|
| TIS | `loss_fn=ppo` provides similar ratio clipping |
| Asymmetric clipping | `kl_penalty_coef=0.02` provides soft regularization |
| Large batch size | More training iterations with smaller batches |

## Monitoring Metrics

Key metrics to watch in wandb:

| Metric | Expected Behavior | Problem Indicator |
|--------|-------------------|-------------------|
| `reward/mean` | Increasing | Flat or decreasing |
| `entropy` | Stable or slightly decreasing | Rapidly increasing (policy collapse) |
| `kl/approx` | Low (<0.1) | High (>0.5) indicates drift |
| `accuracy/{env}` | Increasing per environment | Flat indicates learning failure |
| `difficulty/{env}` | Increasing as accuracy improves | Stuck at 0 |

## References

- [GRPO Paper](https://arxiv.org/abs/2402.03300): Group Relative Policy Optimization
- [PPO Paper](https://arxiv.org/abs/1707.06347): Proximal Policy Optimization
- [InstructGPT](https://arxiv.org/abs/2203.02155): KL penalty coefficient reference
- [RLVE Paper](https://arxiv.org/abs/2505.00000): Reinforcement Learning with Verifiable Environments
