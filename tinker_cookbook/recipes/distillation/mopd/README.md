# M-OPD: multi-teacher on-policy distillation with a top-K reward

This recipe runs the M-OPD stage of [Open-MOPD](https://github.com/BytedTsinghua-SIA/Open-MOPD)
(arXiv 2608.19098) through the Tinker API. A student that was fine-tuned on mixed data learns from three
teachers, one per domain (math, code, instruction following), on its own samples.

```bash
python -m tinker_cookbook.recipes.distillation.mopd.train n_steps=60
```

## One step

1. **Rollouts.** 128 prompts at math : code : IF = 2 : 2 : 1 (51 / 51 / 26). The student samples one
   response to each at temperature 1, up to 16,384 tokens.
2. **Scores.** The sampler returns its 20 most likely ids at every sampled position. One student `forward`
   scores them, and the 16 with the highest logprobs stay: S. The teacher of the prompt's domain scores
   those 16 ids with a `forward`: T.
3. **Reward.** For each id, `softmax over the 16 of S, times (T - S)`. There is no baseline and no critic.
4. **Domain weight.** Each sequence's reward is multiplied by its domain's weight. The weight brings each
   domain's share of the loss toward one third and follows the size of the domain's reward.
5. **Four updates.** The 128 sequences are ordered into 4 length-balanced shards of 32. Update j takes rows
   `[8j, 8j + 8)` of every shard. Each update is one `forward_backward_custom` with a clipped
   policy-gradient loss over the 16 ids (ratio range 0.8 to 1.28, loss capped at `-3A` for a negative
   advantage), then one `optim_step`. With `refresh_advantage` the update rebuilds the reward from its
   own current logprobs.

The loss of an update is the mean over its 32 sequences of each sequence's own token mean. Short
sequences therefore weigh as much as long ones, on top of the domain weight.

## Settings

| setting | default | meaning |
|---|---|---|
| `n_steps` | 60 | rollout steps; each makes four optimizer steps |
| `batch_size` | 128 | prompts per step, a multiple of 16 |
| `max_tokens` | 16384 | response limit of a rollout |
| `refresh_advantage` | True | rebuild the reward in every update; False keeps the rollout-time reward |
| `seed` | 1 | the prompt sampler's seed |
| `replay_batch` | None | a rollout batch recorded from the reference; every step trains on it in place of own rollouts, and `batch_size`, `max_tokens` and `seed` play no part |
| `log_path` | `~/tinker-runs/mopd` | metrics and logs |
| `behavior_if_log_dir_exists` | ask | what to do when `log_path` exists: ask, delete, resume or raise |
| `wandb_project`, `wandb_name` | None | log to Weights & Biases as well |
| `base_url` | None | the Tinker service to use |

Fixed by the method: the models (`BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-MixSFT` and its three `RL-*`
teachers, all with full weights), prompts of at most 2,048 tokens, AdamW at learning rate 1.5e-6 with
betas (0.9, 0.999), weight decay 0.01 and a gradient clip at norm 1.0.

## Requirements

The service must hold four full-weight models at once, return top-20 logprobs from the sampler, and accept
`[N, K]` targets in `forward` and `forward_backward_custom`. A step at the default sizes needs a context
of 18,432 tokens. For a quick check, run `n_steps=1 batch_size=16 max_tokens=512`.

## Metrics

`metrics.jsonl` uses the reference's names, so the two runs can be read side by side:

- `mt_opd/rm_scores_mean`: the mean reward per response token.
- `mt_opd/domain/<domain>/reward_abs_mean`, `loss_weight`, `prompt_count`, `token_share`,
  `mean_response_length`.
- `mt_opd/inner_update_<i>/grad_norm` (as the optimizer step reports it), `pg_loss` (the update's loss
  times 4 over its sequence count, an eighth at the default sizes, as the reference reports it) and `ppo_kl` (the mean over sequences of `rollout logprob - current logprob`,
  summed over the 16 ids).
- `mt_opd/m4_advantage_refreshed`: 1.0 when the refresh ran.

## Tests

`step_test.py` and `prompts_test.py` compare the arithmetic and the prompt sampler with values recorded
from the reference (`testdata/reference_step.pt`): one recorded step's reward, domain weights and update
membership, and the reference's own loss and sampler on seeded inputs.
