"""The arithmetic of one M-OPD step: reward, domain weight, update membership and the update's loss.

M-OPD is the multi-teacher on-policy distillation step of Open-MOPD (arXiv 2608.19098). "The reference" below
is its implementation, commit 4809a96 of github.com/BytedTsinghua-SIA/Open-MOPD, paths under
`training/verl/verl/`. Every function here reproduces one of its bodies; `step_test.py` holds them to values
recorded from it.

Tensors of one sequence have shape [r, K]: r response positions, K candidate ids per position.
"""

import heapq
from collections.abc import Sequence
from dataclasses import dataclass

import torch

# The reference's `clip_ratio_low`, `clip_ratio_high` and `clip_ratio_c` (`trainer/ppo/core_algos.py`,
# `compute_policy_loss_vanilla`): the ratio's range, and the cap on the loss of an id with a negative advantage.
CLIP_LOW, CLIP_HIGH, CLIP_NEGATIVE = 0.2, 0.28, 3.0
# The reference bounds the gap-following factor of a domain's weight (`workers/actor/mt_opd.py`).
MAGNITUDE_FACTOR_RANGE = (0.05, 20.0)
# The reference bounds `current logprob - rollout logprob` before the exponential (`compute_policy_loss_vanilla`).
LOG_RATIO_RANGE = (-20.0, 20.0)


def topk_reward(student_logprobs: torch.Tensor, teacher_logprobs: torch.Tensor) -> torch.Tensor:
    """The reward of each candidate id: the student's probability, renormalised over the K ids, times the
    teacher's logprob minus the student's."""
    return torch.softmax(student_logprobs, dim=-1) * (teacher_logprobs - student_logprobs)


def best_candidates(
    ids: torch.Tensor, logprobs: torch.Tensor, keep: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Of each position's candidate ids, the `keep` with the highest logprobs, and those logprobs."""
    top = torch.topk(logprobs, keep, dim=-1)
    return ids.gather(-1, top.indices), top.values


def rows_by_domain(domains: Sequence[str]) -> dict[str, list[int]]:
    """The rows of each domain present, the domains in sorted order."""
    return {d: [row for row, name in enumerate(domains) if name == d] for d in sorted(set(domains))}


@dataclass(frozen=True)
class DomainStatistics:
    """What a rollout batch holds of one domain."""

    sequences: int
    response_tokens: int
    # the domain's part of the batch's response tokens
    token_share: float
    # the mean over the domain's response positions of |the reward summed over the K ids|
    reward_abs_mean: float


def domain_statistics(
    domains: Sequence[str], response_tokens: Sequence[int], abs_reward: Sequence[float]
) -> dict[str, DomainStatistics]:
    """`abs_reward[i]` is sequence i's sum over its response positions of |the reward summed over the K ids|."""
    total = sum(response_tokens)
    statistics = {}
    for domain, rows in rows_by_domain(domains).items():
        tokens = sum(response_tokens[i] for i in rows)
        statistics[domain] = DomainStatistics(
            len(rows), tokens, tokens / total, sum(abs_reward[i] for i in rows) / tokens
        )
    return statistics


def domain_weights(
    statistics: dict[str, DomainStatistics], target_share: dict[str, float]
) -> dict[str, float]:
    """The loss weight of each domain's sequences (the reference's `compute_domain_loss_weights` with
    `normalize_reward_scale=1.0` and `reward_scale_direction="multiply"`).

    A domain's weight is its target share over its share of the batch's tokens, times how far its mean
    absolute reward lies from the batch's, and all weights are scaled to a token-weighted mean of 1.
    """
    scale = {d: target_share[d] / s.token_share for d, s in statistics.items()}
    batch_magnitude = sum(scale[d] * s.reward_abs_mean for d, s in statistics.items()) / sum(
        scale.values()
    )
    low, high = MAGNITUDE_FACTOR_RANGE
    scale = {
        d: scale[d] * min(max(s.reward_abs_mean / batch_magnitude, low), high)
        for d, s in statistics.items()
    }
    weighted_share = sum(scale[d] * s.token_share for d, s in statistics.items())
    return {d: scale[d] / weighted_share for d in scale}


def rollout_metrics(
    statistics: dict[str, DomainStatistics], reward_sum: Sequence[float], weights: dict[str, float]
) -> dict[str, float]:
    """The reference's metrics of a rollout batch, under its names. `reward_sum[i]` is sequence i's reward
    summed over its positions and ids."""
    total = sum(s.response_tokens for s in statistics.values())
    metrics = {"mt_opd/rm_scores_mean": sum(reward_sum) / total}
    for domain, s in statistics.items():
        prefix = f"mt_opd/domain/{domain}"
        metrics[f"{prefix}/prompt_count"] = float(s.sequences)
        metrics[f"{prefix}/token_share"] = s.token_share
        metrics[f"{prefix}/mean_response_length"] = s.response_tokens / s.sequences
        metrics[f"{prefix}/reward_abs_mean"] = s.reward_abs_mean
        metrics[f"{prefix}/loss_weight"] = weights[domain]
    return metrics


class _Subset:
    """Rows on their way into one shard, and the sum of their workloads."""

    def __init__(self) -> None:
        self.total = 0
        self.items: list[tuple[int, int]] = []

    def add(self, row: int, workload: int) -> None:
        self.items.append((row, workload))
        self.total += workload

    def merge(self, other: "_Subset") -> None:
        self.items.extend(other.items)
        self.total += other.total

    def __lt__(self, other: "_Subset") -> bool:
        if self.total != other.total:
            return self.total < other.total
        if len(self.items) != len(other.items):
            return len(self.items) < len(other.items)
        return self.items < other.items


class _Partition:
    """One subset per shard, the heaviest first. Two partitions merge heaviest onto lightest."""

    def __init__(self, items: list[tuple[int, int]], shards: int) -> None:
        self.subsets = [_Subset() for _ in range(shards)]
        for subset, (row, workload) in zip(self.subsets, items, strict=True):
            subset.add(row, workload)
        self.subsets.sort(reverse=True)

    def merge(self, other: "_Partition") -> None:
        for subset, lighter in zip(self.subsets, reversed(other.subsets), strict=True):
            subset.merge(lighter)
        self.subsets.sort(reverse=True)

    @property
    def spread(self) -> int:
        return self.subsets[0].total - self.subsets[-1].total

    def __lt__(self, other: "_Partition") -> bool:
        # the heap pops the widest spread first
        if self.spread != other.spread:
            return self.spread > other.spread
        return self.subsets[0] > other.subsets[0]


def balanced_shards(input_tokens: Sequence[int], shards: int) -> list[list[int]]:
    """The reference's order of a rollout batch (`trainer/ppo/ray_trainer.py`, `_balance_batch`): the rows
    of each of its data-parallel ranks, in the order that rank takes them.

    The rows are split into `shards` equal groups of near-equal workload by the Karmarkar-Karp method
    (`utils/seqlen_balancing.py`, `karmarkar_karp` with `equal_size`); the workload of a sequence of n
    tokens is `24576 * n + n * n`. Inside a shard the short rows go to both ends.
    """
    workload = [24576 * n + n * n for n in input_tokens]
    by_workload = sorted((w, row) for row, w in enumerate(workload))
    heap = [
        _Partition([(row, w) for w, row in by_workload[offset : offset + shards]], shards)
        for offset in range(0, len(by_workload), shards)
    ]
    heapq.heapify(heap)
    while len(heap) > 1:
        widest = heapq.heappop(heap)
        widest.merge(heapq.heappop(heap))
        heapq.heappush(heap, widest)
    ordered = []
    for subset in heap[0].subsets:
        rows = sorted((row for row, _ in subset.items), key=lambda row: (workload[row], row))
        ordered.append(rows[::2] + rows[1::2][::-1])
    return ordered


def update_rows(shards: list[list[int]], updates: int) -> list[list[int]]:
    """The rows of each inner update: every rank cuts its shard into `updates` consecutive parts, and an
    update is the same part of all shards."""
    part = len(shards[0]) // updates
    return [
        [row for shard in shards for row in shard[part * j : part * (j + 1)]]
        for j in range(updates)
    ]


def update_loss(
    logprobs: Sequence[torch.Tensor],
    rollout_logprobs: Sequence[torch.Tensor],
    teacher_logprobs: Sequence[torch.Tensor],
    weights: Sequence[float],
    refresh: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The loss of one inner update over its sequences, the per-sequence losses, and per sequence the
    token mean of `sum over ids of (rollout logprob - current logprob)`, each difference bounded as in the
    ratio, which the reference reports as `ppo_kl`.

    Per candidate id the loss is the clipped surrogate `max(-A * ratio, -A * clamp(ratio))`, capped at
    `-3A` where A < 0, with `ratio = exp(logprob - rollout logprob)`; it is summed over the ids, averaged
    over a sequence's positions, then over the sequences. A is the sequence's weight times its reward: the
    rollout-time reward, or with `refresh` the reward of the current logprobs, taken as a constant.
    """
    per_sequence, drift = [], []
    for current, old, teacher, weight in zip(
        logprobs, rollout_logprobs, teacher_logprobs, weights, strict=True
    ):
        reward = topk_reward(current.detach() if refresh else old, teacher)
        advantage = reward * weight
        delta = torch.clamp(current - old, *LOG_RATIO_RANGE)
        ratio = torch.exp(delta)
        clipped = torch.maximum(
            -advantage * ratio, -advantage * torch.clamp(ratio, 1 - CLIP_LOW, 1 + CLIP_HIGH)
        )
        per_id = torch.where(
            advantage < 0, torch.minimum(-advantage * CLIP_NEGATIVE, clipped), clipped
        )
        per_sequence.append(per_id.sum(dim=-1).mean())
        drift.append(-delta.detach().sum(dim=-1).mean())
    losses = torch.stack(per_sequence)
    return losses.mean(), losses, torch.stack(drift)
