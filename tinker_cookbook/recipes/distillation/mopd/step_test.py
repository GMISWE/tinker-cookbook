"""The M-OPD arithmetic against the reference: `testdata/reference_step.pt` holds tensors recorded from one
step of Open-MOPD (4809a96) and values its own functions give on seeded inputs."""

import random
from pathlib import Path

import pytest
import torch

from tinker_cookbook.recipes.distillation.mopd import data, step

GOLD = torch.load(Path(__file__).parent / "testdata" / "reference_step.pt", weights_only=True)
SEQUENCES = GOLD["sequences"]
LOSS_IDS = [f"seed {case['seed']}, refresh {case['refresh']}" for case in GOLD["loss"]]


def _relative(value: float, expected: float) -> float:
    return abs(value - expected) / abs(expected)


@pytest.mark.parametrize("row", GOLD["reward"], ids=lambda row: row["domain"])
def test_the_reward_of_recorded_logprobs_is_the_recorded_reward(row):
    """softmax over the 16 ids of S, times (T - S)."""
    reward = step.topk_reward(row["student_log_prob"], row["teacher_log_prob"])
    assert (reward - row["reward"]).abs().max() < 1e-5


def test_the_domain_weights_of_the_recorded_batch_are_the_recorded_weights():
    """IF holds 1.7% of the batch's tokens, so its weight is near 25 where math's is near 0.27."""
    weights = step.domain_weights(
        SEQUENCES["domain"],
        SEQUENCES["response_tokens"],
        SEQUENCES["abs_reward_sum"],
        SEQUENCES["target_share"],
    )
    for domain, recorded in zip(SEQUENCES["domain"], SEQUENCES["domain_loss_weight"], strict=True):
        assert _relative(weights[domain], recorded) < 1e-5


def test_the_rollout_metrics_of_the_recorded_batch_are_the_reference_values():
    """The reward metrics are per response token, the per-domain ones over that domain's tokens only."""
    weights = step.domain_weights(
        SEQUENCES["domain"],
        SEQUENCES["response_tokens"],
        SEQUENCES["abs_reward_sum"],
        SEQUENCES["target_share"],
    )
    metrics = step.rollout_metrics(
        SEQUENCES["domain"],
        SEQUENCES["response_tokens"],
        SEQUENCES["reward_sum"],
        SEQUENCES["abs_reward_sum"],
        weights,
    )
    assert _relative(metrics["mt_opd/rm_scores_mean"], SEQUENCES["rm_scores_mean"]) < 1e-5
    for domain, expected in SEQUENCES["reward_abs_mean"].items():
        assert _relative(metrics[f"mt_opd/domain/{domain}/reward_abs_mean"], expected) < 1e-5
    assert [metrics[f"mt_opd/domain/{d}/prompt_count"] for d in data.DOMAINS] == [
        51.0,
        51.0,
        26.0,
    ]


@pytest.mark.parametrize("case", GOLD["loss"], ids=LOSS_IDS)
def test_the_update_loss_and_its_gradient_are_the_reference_values(case):
    """Sequences of different lengths, ratios on both sides of the clip range, ids beyond the cap on a
    negative advantage. With the refresh the reward follows the current logprobs and carries no gradient."""
    lengths = case["lengths"]
    current = [case["log_prob"][i, :n].clone().requires_grad_() for i, n in enumerate(lengths)]
    loss, per_sequence, drift = step.update_loss(
        current,
        [case["old_log_prob"][i, :n] for i, n in enumerate(lengths)],
        [case["teacher_log_prob"][i, :n] for i, n in enumerate(lengths)],
        case["domain_weight"].tolist(),
        case["refresh"],
    )
    loss.backward()
    assert _relative(float(loss), float(case["loss"])) < 1e-5
    assert torch.allclose(per_sequence, case["per_sequence_loss"], rtol=1e-5, atol=0)
    for i, n in enumerate(lengths):
        expected = case["loss_gradient"][i, :n]
        assert (current[i].grad - expected).abs().max() <= 1e-5 * expected.abs().max()
    # the gold reports the drift as one mean over the update's tokens
    tokens = torch.tensor(lengths, dtype=drift.dtype)
    assert _relative(float((drift * tokens).sum() / tokens.sum()), float(case["ppo_kl"])) < 1e-4


def _recorded_updates() -> list[list[int]]:
    membership = GOLD["membership"]
    return [
        sorted(
            n
            for n, update in zip(
                SEQUENCES["input_tokens"], membership["recorded_update"], strict=True
            )
            if update == j
        )
        for j in range(membership["updates"])
    ]


@pytest.mark.parametrize("shuffle_seed", [None, 0, 1, 2])
def test_each_update_holds_the_lengths_of_the_recorded_update(shuffle_seed):
    """The 128 recorded lengths, in the recorded order and in shuffled orders: the membership follows from
    the lengths alone."""
    lengths = list(SEQUENCES["input_tokens"])
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(lengths)
    membership = GOLD["membership"]
    updates = step.update_rows(
        step.balanced_shards(lengths, membership["ranks"]), membership["updates"]
    )
    assert [sorted(lengths[row] for row in rows) for rows in updates] == _recorded_updates()


@pytest.mark.parametrize(
    "case", GOLD["membership"]["cases"], ids=lambda case: f"seed {case['seed']}"
)
def test_the_order_of_each_shard_is_the_reference_order(case):
    """Seeded length lists, some with many equal lengths: equal workloads are ordered by row index."""
    assert step.balanced_shards(case["lengths"], GOLD["membership"]["ranks"]) == case["shards"]


def test_an_update_is_the_same_part_of_every_shard():
    """Not consecutive rows of the batch: update 0 is the first half of each of the two shards."""
    shards = [[0, 1, 2, 3], [10, 11, 12, 13]]
    assert step.update_rows(shards, 2) == [[0, 1, 10, 11], [2, 3, 12, 13]]


def test_the_best_candidates_are_the_ids_with_the_highest_logprobs():
    """20 candidates per position, the 4 to drop at known places; a kept id keeps its own logprob."""
    generator = torch.Generator().manual_seed(0)
    ids = torch.stack([torch.randperm(1000, generator=generator)[:20] for _ in range(7)])
    logprobs = -torch.rand(7, 20, generator=generator)
    dropped = torch.tensor([[0, 5, 11, 19]] * 7)
    logprobs.scatter_(1, dropped, -5.0 - torch.rand(7, 4, generator=generator))
    kept_ids, kept_logprobs = step.best_candidates(ids, logprobs, 16)
    assert kept_ids.shape == kept_logprobs.shape == (7, 16)
    for position in range(7):
        expected = {int(ids[position, c]) for c in range(20) if c not in (0, 5, 11, 19)}
        assert set(kept_ids[position].tolist()) == expected
        for token, value in zip(
            kept_ids[position].tolist(), kept_logprobs[position].tolist(), strict=True
        ):
            assert value == float(logprobs[position, ids[position].tolist().index(token)])
