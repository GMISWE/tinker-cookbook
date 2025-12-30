# Reinforce-Ada (Tinker Recipe)

This recipe ports the Reinforce-Ada adaptive sampling flow into the upstream
tinker-cookbook RL pipeline. It adds multi-round adaptive sampling and optional
global GRPO statistics to the standard RL training loop.

## Entry Point
Run:

```bash
python -m tinker_cookbook.recipes.reinforce_ada.train
```

## Key Features
- Multi-round adaptive sampling to reduce signal collapse.
- Balanced or positive-focused sampling strategies.
- Optional global GRPO statistics across prompt history.

## Core CLI Flags
- `multiround_adaptive_downsampling=True|False`
- `reinforce_ada_choice=balanced|positive-focused`
- `max_rounds=<int>`
- `round_repeat=<int>`
- `positive_threshold=<float>`
- `global_stat_est=True|False`

## Dataset Expectations
The dataset must have:
- `problem`: the prompt text
- `gt`: the ground-truth answer

Default dataset is `RLHFlow/reinforce_ada_hard_prompt`.

## How It Works (High Level)
1) Build math environments from dataset rows.
2) Sample trajectories per prompt, optionally using adaptive multi-round sampling.
3) Compute rewards and GRPO advantages (local or global mean).
4) Train via the standard RL loop in `tinker_cookbook/rl/train.py`.

## Notes
- `n_batches` controls the total number of training steps when set.
- `group_size` is the final number of samples kept per prompt when adaptive
  sampling is enabled.
