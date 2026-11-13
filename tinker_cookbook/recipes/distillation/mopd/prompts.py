"""The prompts of an M-OPD run: Open-MOPD's mixed prompt set, drawn at a fixed ratio of domains per batch.

`domain_batches` reproduces the reference's `DomainWeightedSampler`
(github.com/BytedTsinghua-SIA/Open-MOPD 4809a96, `training/verl/verl/utils/dataset/`): with the same domain
column and seed it yields the same rows in the same order.
"""

import math
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import datasets
from huggingface_hub import hf_hub_download

from tinker_cookbook.tokenizer_utils import Tokenizer

DATASET = "BytedTsinghua-SIA/Open-MOPD-Data"
DATASET_REVISION = "9e897efe3257599d4300e2d5ee865a1cc714af87"
DATASET_FILE = "rl_prompt_mix/train.parquet"
# The sampler visits the domains in this order; ties in the quota go to the earlier one.
DOMAINS: tuple[str, ...] = ("math", "code", "if")
# Open-MOPD draws math : code : instruction following at 2 : 2 : 1 (arXiv 2608.19098, Table 8).
DOMAIN_RATIO = {"math": 2.0, "code": 2.0, "if": 1.0}


@dataclass(frozen=True)
class Prompt:
    domain: str
    tokens: list[int]


def batch_quotas(batch_size: int, ratio: dict[str, float]) -> dict[str, int]:
    """Prompts per domain in a batch: the ratio's shares rounded down, the remainder to the largest fractions."""
    total = sum(ratio[d] for d in DOMAINS)
    exact = {d: batch_size * ratio[d] / total for d in DOMAINS}
    quotas = {d: math.floor(exact[d]) for d in DOMAINS}
    by_fraction = sorted(DOMAINS, key=lambda d: (quotas[d] - exact[d], DOMAINS.index(d)))
    for d in by_fraction[: batch_size - sum(quotas.values())]:
        quotas[d] += 1
    return quotas


def domain_batches(
    domains: Sequence[str], batch_size: int, ratio: dict[str, float], seed: int
) -> Iterator[list[int]]:
    """One epoch of batches of row indexes into `domains`, each with the ratio's count of every domain.

    A domain's rows are shuffled and cycled, so a small domain repeats before a large one is used up. An
    epoch is as many batches as the rows fill.
    """
    rng = random.Random(seed)
    pools = {d: [row for row, name in enumerate(domains) if name == d] for d in DOMAINS}
    cursors = dict.fromkeys(DOMAINS, 0)
    for pool in pools.values():
        rng.shuffle(pool)

    def take(domain: str, count: int) -> list[int]:
        taken: list[int] = []
        pool = pools[domain]
        while len(taken) < count:
            amount = min(count - len(taken), len(pool) - cursors[domain])
            taken.extend(pool[cursors[domain] : cursors[domain] + amount])
            cursors[domain] += amount
            if cursors[domain] == len(pool):
                rng.shuffle(pool)
                cursors[domain] = 0
        return taken

    base = batch_quotas(batch_size, ratio)
    for batch_index in range(len(domains) // batch_size):
        quotas = dict(base)
        # when math holds one prompt more than code at an equal ratio, they swap on every second batch
        if (
            batch_index % 2 == 1
            and ratio["math"] == ratio["code"]
            and quotas["math"] == quotas["code"] + 1
        ):
            quotas["math"], quotas["code"] = quotas["code"], quotas["math"]
        batch = [row for d in DOMAINS for row in take(d, quotas[d])]
        rng.shuffle(batch)
        yield batch


def within_limit(prompts: Sequence[Prompt], max_prompt_tokens: int) -> list[Prompt]:
    """The prompts a batch may draw: those of at most `max_prompt_tokens` tokens, in their order."""
    return [prompt for prompt in prompts if len(prompt.tokens) <= max_prompt_tokens]


def load_prompts(tokenizer: Tokenizer) -> list[Prompt]:
    """Every prompt of the mixed set, as the tokens of the model's own chat template."""
    path = hf_hub_download(DATASET, DATASET_FILE, repo_type="dataset", revision=DATASET_REVISION)
    rows = datasets.Dataset.from_parquet(path)
    assert isinstance(rows, datasets.Dataset)
    prompts = []
    for domain, messages in zip(rows["domain"], rows["prompt"], strict=True):
        encoded = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True
        )
        # `return_dict=True` gives the encoding, never the bare text or id list of the other forms
        assert not isinstance(encoded, (str, list))
        prompts.append(Prompt(domain, [int(token) for token in encoded.data["input_ids"]]))
    return prompts
