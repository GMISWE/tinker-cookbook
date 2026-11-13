"""The prompt sampler against the reference's: `testdata/reference_step.pt` holds seeded domain columns and
the batches Open-MOPD's `DomainWeightedSampler` draws from them."""

from itertools import islice
from pathlib import Path

import pytest
import torch

from tinker_cookbook.recipes.distillation.mopd import data

GOLD = torch.load(Path(__file__).parent / "testdata" / "reference_step.pt", weights_only=True)


def test_a_batch_of_128_at_two_two_one_is_51_51_26():
    assert data.batch_quotas(128, data.DOMAIN_RATIO) == {"math": 51, "code": 51, "if": 26}


@pytest.mark.parametrize("case", GOLD["sampler"], ids=lambda case: f"batch {case['batch_size']}")
def test_the_batches_are_the_reference_samplers(case):
    """Batch sizes 128, 20 and 9. At 9 math holds one prompt more than code and they swap on every second
    batch; the small pools are used up and reshuffled inside the compared batches."""
    batches = data.domain_batches(
        case["domain"], case["batch_size"], data.DOMAIN_RATIO, case["seed"]
    )
    assert list(islice(batches, len(case["batches"]))) == case["batches"]


def test_a_prompt_of_the_limit_is_kept_and_one_token_more_is_dropped():
    """Only the kept prompts reach the sampler, so a longer one is never drawn."""
    prompts = [data.Prompt("math", [0] * length) for length in (5, 2048, 2049, 9000)]
    assert [len(p.tokens) for p in data.within_limit(prompts, 2048)] == [5, 2048]
