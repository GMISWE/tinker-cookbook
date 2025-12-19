# Tinker for RLVE
This implementation of RLVE is built on Tinker, which decouples environments from the training loop to enable flexible adaptation.
This separation allows a clean and modular design that makes environments dynamically adaptive to the policy.

<div align="center">
  <img src="../../../assets/rlve_showcase.png" width="60%" />
</div>


## Installation

Install tinker-cookbook following the official guidance [here](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main?tab=readme-ov-file#installation):

```
1. Sign up for Tinker through the [waitlist](https://thinkingmachines.ai/tinker).
2. Once you have access, create an API key from the [console](https://tinker-console.thinkingmachines.ai) and export it as environment variable `TINKER_API_KEY`.
3. Install tinker python client via `pip install tinker`
4. We recommend installing `tinker-cookbook` in a virtual env either with `conda` or `uv`. For running most examples, you can install via `pip install -e .`.
```

Current implementation is compatible with `tinker==0.3.0`.


## Usage

### Default Setup in Paper
```bash
python -m tinker_cookbook.recipes.rlve.train model_name="Qwen/Qwen3-4B-Instruct-2507"
```

### Advanced Usage
```bash
# Speeds up difficulty adaptation by adjusting the hyperparameters of adaptive environments,
# but risks pushing the model into overly hard problems too soon.
python -m tinker_cookbook.recipes.rlve.train model_name="Qwen/Qwen3-4B-Instruct-2507" \
    initial_difficulty=3 \
    difficulty_sliding_window_size=2 \
    min_prompts_before_difficulty_check=1
```

### Key Parameters

**RLVE Configuration:**
- `reward_key`: Reward metric to use (default: "accuracy")
- `initial_difficulty`: Initial difficulty level (default: 0)
- `difficulty_sliding_window_size`: Sliding window size for difficulty (default: 4)
- `min_prompts_before_difficulty_check`: Minimum prompts before checking difficulty (default: 8)
- `n_samples_per_prompt`: Number of samples per prompt (default: group_size)
- `min_metric_to_increase_difficulty`: Minimum metric to increase difficulty (default: 0.9)

These arguments are aligned with those defined [here](https://github.com/Zhiyuan-Zeng/RLVE/blob/main/slime/utils/arguments.py#L845-#L899).

### Difference from the Implementation Using Slime

In this RLVE implementation based on Tinker, we made a minor modification compared to the main implementation using slime (used in our paper's original experiments) to avoid altering the primary training loop.

Specifically, instead of checking performance statistics and updating difficulty at the end of each training step, we integrate these operations directly within the environment implementation.
Whenever an environment is queried to provide a reward, it simultaneously updates its internal statistics and performs the difficulty adjustment.

We expect this difference to have a negligible effect on the overall dynamics of difficulty adaptation, particularly when training jointly on a sufficiently large number of environments.
