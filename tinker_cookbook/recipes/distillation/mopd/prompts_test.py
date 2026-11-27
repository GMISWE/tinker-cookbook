"""The prompt sampler against the reference's: `testdata/reference_step.pt` holds seeded domain columns and
the batches Open-MOPD's `DomainWeightedSampler` draws from them."""

from itertools import islice
from pathlib import Path

import pytest
import torch

from tinker_cookbook.recipes.distillation.mopd import prompts

GOLD = torch.load(Path(__file__).parent / "testdata" / "reference_step.pt", weights_only=True)


def test_a_batch_of_128_at_two_two_one_is_51_51_26():
    """The README's batch: the one prompt left after rounding down goes to IF, the largest fraction."""
    assert prompts.batch_quotas(128, prompts.DOMAIN_RATIO) == {"math": 51, "code": 51, "if": 26}


@pytest.mark.parametrize("case", GOLD["sampler"], ids=lambda case: f"batch {case['batch_size']}")
def test_the_batches_are_the_reference_samplers(case):
    """Batch sizes 128, 20 and 9. At 9 math holds one prompt more than code and they swap on every second
    batch; the small pools are used up and reshuffled inside the compared batches."""
    batches = prompts.domain_batches(
        case["domain"], case["batch_size"], prompts.DOMAIN_RATIO, case["seed"]
    )
    assert list(islice(batches, len(case["batches"]))) == case["batches"]


def test_a_prompt_of_the_limit_is_kept_and_one_token_more_is_dropped():
    """Only the kept prompts reach the sampler, so a longer one is never drawn."""
    pool = [prompts.Prompt("math", [0] * length) for length in (5, 2048, 2049, 9000)]
    assert [len(p.tokens) for p in prompts.within_limit(pool, 2048)] == [5, 2048]


@pytest.mark.timeout(2, method="signal")
@pytest.mark.parametrize("empty", [("if",), ("math", "code")])
def test_empty_domains_with_positive_quotas_are_refused_after_filtering(empty: tuple[str, ...]):
    """The deadline turns the former infinite loop into a bounded test failure."""
    pool = [
        prompts.Prompt(d, [0] * (3 if d in empty else 2))
        for d in prompts.DOMAINS
        for _ in range(16)
    ]
    kept = prompts.within_limit(pool, 2)
    batches = prompts.domain_batches([p.domain for p in kept], 16, prompts.DOMAIN_RATIO, 0)
    with pytest.raises(
        ValueError, match=f"No prompts for domains with positive quotas: {', '.join(empty)}"
    ):
        next(batches)


def test_an_empty_domain_with_zero_quota_is_allowed():
    domains = ["math", "code"] * 8
    batches = prompts.domain_batches(domains, 4, {"math": 1.0, "code": 1.0, "if": 0.0}, 0)
    assert len(list(batches)) == 4


@pytest.mark.timeout(2, method="signal")
def test_an_empty_domain_is_refused_when_the_swapped_quota_becomes_positive():
    batches = prompts.domain_batches(["math", "math"], 1, prompts.DOMAIN_RATIO, 0)
    assert len(next(batches)) == 1
    with pytest.raises(ValueError, match="positive quotas: code"):
        next(batches)
