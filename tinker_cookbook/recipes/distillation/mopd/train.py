"""Multi-teacher on-policy distillation with a top-K reward (M-OPD), as Open-MOPD runs it.

A full-weight student samples a response to each prompt of a mixed batch. At every sampled position 16
candidate ids, the trainer's best of the sampler's 20, are scored by the teacher of the prompt's domain.
The reward of an id is the student's probability, renormalised over the 16, times the teacher's logprob
minus the student's. Four clipped policy-gradient
updates follow, each on a quarter of the batch. See the README beside this file.

    python -m tinker_cookbook.recipes.distillation.mopd.train n_steps=60
"""

import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import chz
import tinker
import torch
from tinker import types

from tinker_cookbook import cli_utils
from tinker_cookbook.recipes.distillation.mopd import prompts, step
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.git_rev import recipe_user_metadata
from tinker_cookbook.utils.misc_utils import timed

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)

# The run this recipe reproduces: Open-MOPD's M-OPD stage (arXiv 2608.19098, Table 8) on its published
# checkpoints. Values without a setting below are part of the method, not choices of a run.
MODEL_FAMILY = "BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-"
STUDENT = MODEL_FAMILY + "MixSFT"
TEACHERS = {
    "math": MODEL_FAMILY + "RL-Math",
    "code": MODEL_FAMILY + "RL-Code",
    "if": MODEL_FAMILY + "RL-IF",
}
TARGET_SHARE = {"math": 0.3333333, "code": 0.3333333, "if": 0.3333334}
MAX_PROMPT_TOKENS = 2048
TOP_K = 16
# The API returns the sampler's top ids; the reward wants the trainer's. The trainer picks its 16 among
# 20 of the sampler's, so an id the two rank differently near the cut is still a candidate.
SAMPLER_CANDIDATES = 20
UPDATES_PER_STEP = 4
# The reference trains on 4 data-parallel ranks, and which sequences share an update follows from that split.
BALANCE_SHARDS = 4
ADAM = types.AdamParams(
    learning_rate=1.5e-6, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01, grad_clip_norm=1.0
)

LossFn = Callable[[list[types.Datum], list[torch.Tensor]], tuple[torch.Tensor, dict[str, float]]]


@chz.chz
class Config:
    base_url: str | None = None
    log_path: str = cli_utils.runs_path("mopd")
    # as the other recipes: a run asks before it writes into an existing directory
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"
    wandb_project: str | None = None
    wandb_name: str | None = None
    # rollout steps, four optimizer steps each
    n_steps: int = 60
    # prompts per step; a multiple of 16, so that 4 updates split evenly over 4 shards
    batch_size: int = 128
    # response limit of a rollout; the reference's 16,384 needs a server context of 18,432 tokens
    max_tokens: int = 16384
    # True rebuilds the reward from each update's own logprobs (the paper's method); False keeps the
    # rollout-time reward for all four updates
    refresh_advantage: bool = True
    # of the prompt sampler: the same seed draws the same prompts in the same order
    seed: int = 1
    # A rollout batch recorded from the reference. Every step then trains on it, in the recorded order,
    # in place of this recipe's own rollouts (`batch_size`, `max_tokens` and `seed` play no part): for
    # comparing one step with the reference's.
    replay_batch: str | None = None


@dataclass
class Rollout:
    domain: str
    prompt: list[int]
    response: list[int]
    # candidate ids at every response position, [r, C]
    candidates: torch.Tensor

    @property
    def first_response_row(self) -> int:
        """Response position i is predicted at input position `first_response_row + i`."""
        return len(self.prompt) - 1

    def response_rows(self, per_input_position: torch.Tensor) -> torch.Tensor:
        return per_input_position[self.first_response_row :].float()


@dataclass
class Batch:
    rollouts: list[Rollout]
    # the rollouts' rows as the reference's data-parallel ranks take them (`step.balanced_shards`)
    shards: list[list[int]]


def require_whole_updates(count: int, count_name: str) -> None:
    """Every shard must split into the step's updates, or the rows left over would never train."""
    if count % (BALANCE_SHARDS * UPDATES_PER_STEP) != 0:
        raise ValueError(
            f"{count_name} is {count}, not a multiple of {BALANCE_SHARDS * UPDATES_PER_STEP}"
        )


def sample_rollouts(
    student: tinker.TrainingClient, drawn: list[prompts.Prompt], max_tokens: int, name: str
) -> list[Rollout]:
    sampler = student.save_weights_and_get_sampling_client(name=name)
    # T=1, no nucleus or top-k cut, stop at the model's own end token
    params = types.SamplingParams(max_tokens=max_tokens, temperature=1.0, top_p=1.0)
    futures = [
        sampler.sample(
            types.ModelInput.from_ints(prompt.tokens),
            num_samples=1,
            sampling_params=params,
            topk_sample_logprobs=SAMPLER_CANDIDATES,
        )
        for prompt in drawn
    ]
    rollouts = []
    for prompt, future in zip(drawn, futures, strict=True):
        sequence = future.result().sequences[0]
        topk = sequence.topk_logprobs_np
        assert topk is not None, "the sampler returned no top-k ids"
        candidates = torch.from_numpy(topk.token_ids).long()
        rollouts.append(Rollout(prompt.domain, prompt.tokens, list(sequence.tokens), candidates))
    return rollouts


def own_batches(config: Config, student: tinker.TrainingClient) -> Iterator[Batch]:
    """A batch of fresh rollouts per step, the prompts drawn at the domain ratio."""
    require_whole_updates(config.batch_size, "batch_size")
    pool = prompts.within_limit(prompts.load_prompts(get_tokenizer(STUDENT)), MAX_PROMPT_TOKENS)
    if config.n_steps > len(pool) // config.batch_size:
        raise ValueError(f"n_steps {config.n_steps} is more than one pass over {len(pool)} prompts")
    draws = prompts.domain_batches(
        [p.domain for p in pool], config.batch_size, prompts.DOMAIN_RATIO, config.seed
    )
    for index, rows in enumerate(draws):
        rollouts = sample_rollouts(
            student, [pool[row] for row in rows], config.max_tokens, name=f"step{index:04d}"
        )
        lengths = [len(r.prompt) + len(r.response) for r in rollouts]
        yield Batch(rollouts, step.balanced_shards(lengths, BALANCE_SHARDS))


def load_recorded_batch(path: str) -> Batch:
    """The reference's batch dump (`trainer.data_proto_dump`, stage `post_advantage`) saved as
    `{"tensors": {...}, "domain": [...]}`: left-padded prompts, right-padded responses, its 16 ids. Its
    rows are in the reference's order, one rank's shard after the other."""
    recorded = torch.load(path, weights_only=True)
    tensors = recorded["tensors"]
    width = tensors["prompts"].shape[1]
    attention = tensors["attention_mask"].bool()
    rollouts = []
    for row, domain in enumerate(recorded["domain"]):
        tokens = tensors["input_ids"][row][attention[row]].tolist()
        prompt_len = int(attention[row, :width].sum())
        candidates = tensors["student_top_k_ids"][row, : len(tokens) - prompt_len].clone()
        rollouts.append(Rollout(str(domain), tokens[:prompt_len], tokens[prompt_len:], candidates))
    require_whole_updates(len(rollouts), f"the sequence count of {path}")
    size = len(rollouts) // BALANCE_SHARDS
    return Batch(rollouts, [list(range(size * s, size * (s + 1))) for s in range(BALANCE_SHARDS)])


def recorded_batches(path: str) -> Iterator[Batch]:
    batch = load_recorded_batch(path)
    while True:
        yield batch


def scoring_datum(rollout: Rollout, ids: torch.Tensor, with_weights: bool) -> types.Datum:
    """The sequence with `ids` ([r, K]) as the targets of its response positions; the prompt's other
    positions carry id 0."""
    tokens = rollout.prompt + rollout.response
    targets = torch.zeros(len(tokens) - 1, ids.shape[1], dtype=torch.long)
    targets[rollout.first_response_row :] = ids
    inputs = {"target_tokens": types.TensorData.from_torch(targets)}
    if with_weights:
        # `cross_entropy` takes weights; this recipe reads only the logprobs a `forward` returns
        inputs["weights"] = types.TensorData.from_torch(torch.zeros(targets.shape))
    return types.Datum(model_input=types.ModelInput.from_ints(tokens[:-1]), loss_fn_inputs=inputs)


def response_logprobs(
    client: tinker.TrainingClient, rollouts: list[Rollout], ids: list[torch.Tensor]
) -> list[torch.Tensor]:
    """The client's logprobs of `ids` at every response position of each rollout, [r, K]."""
    datums = [
        scoring_datum(rollout, row_ids, with_weights=True)
        for rollout, row_ids in zip(rollouts, ids, strict=True)
    ]
    outputs = client.forward(datums, "cross_entropy").result().loss_fn_outputs
    return [
        rollout.response_rows(output["logprobs"].to_torch())
        for rollout, output in zip(rollouts, outputs, strict=True)
    ]


def train_step(
    student: tinker.TrainingClient,
    teachers: dict[str, tinker.TrainingClient],
    batch: Batch,
    refresh_advantage: bool,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    rollouts = batch.rollouts
    domains = [rollout.domain for rollout in rollouts]
    response_tokens = [len(rollout.response) for rollout in rollouts]

    with timed("student_forward", metrics):
        candidate_logprobs = response_logprobs(student, rollouts, [r.candidates for r in rollouts])
    kept = [
        step.best_candidates(rollout.candidates, logprobs, TOP_K)
        for rollout, logprobs in zip(rollouts, candidate_logprobs, strict=True)
    ]
    ids = [kept_ids for kept_ids, _ in kept]
    student_logprobs = [kept_logprobs for _, kept_logprobs in kept]

    with timed("teacher_forwards", metrics):
        by_row: dict[int, torch.Tensor] = {}
        # One teacher at a time: a workaround with thin evidence. On a service that holds the four
        # models on shared GPUs, a step whose three teacher forwards were submitted together failed
        # in 1 run of 1 (one forward returned an error); with one forward at a time 6 runs of 6
        # completed. The cause is not established.
        for domain, rows in step.rows_by_domain(domains).items():
            scored = response_logprobs(
                teachers[domain], [rollouts[i] for i in rows], [ids[i] for i in rows]
            )
            by_row.update(zip(rows, scored, strict=True))
        teacher_logprobs = [by_row[i] for i in range(len(rollouts))]

    per_position = [
        step.topk_reward(s, t).sum(dim=-1)
        for s, t in zip(student_logprobs, teacher_logprobs, strict=True)
    ]
    reward_sum = [float(r.double().sum()) for r in per_position]
    abs_reward = [float(r.double().abs().sum()) for r in per_position]
    statistics = step.domain_statistics(domains, response_tokens, abs_reward)
    weights = step.domain_weights(statistics, TARGET_SHARE)
    metrics.update(step.rollout_metrics(statistics, reward_sum, weights))

    def loss_of(rows: list[int]) -> LossFn:
        def loss_fn(
            _data: list[types.Datum], logprobs: list[torch.Tensor]
        ) -> tuple[torch.Tensor, dict[str, float]]:
            loss, _, drift = step.update_loss(
                [rollouts[i].response_rows(row) for i, row in zip(rows, logprobs, strict=True)],
                [student_logprobs[i] for i in rows],
                [teacher_logprobs[i] for i in rows],
                [weights[domains[i]] for i in rows],
                refresh_advantage,
            )
            # the reference's conventions: each micro-batch (one sequence on one of the ranks) reports
            # its share of the update's loss, and `ppo_kl` is the mean over sequences
            return loss, {
                "pg_loss": float(loss) * BALANCE_SHARDS / len(rows),
                "ppo_kl": float(drift.mean()),
            }

        return loss_fn

    grad_norms = []
    for j, rows in enumerate(step.update_rows(batch.shards, UPDATES_PER_STEP), start=1):
        with timed(f"update_{j}", metrics):
            datums = [scoring_datum(rollouts[i], ids[i], with_weights=False) for i in rows]
            backward = student.forward_backward_custom(datums, loss_of(rows))
            optim = student.optim_step(ADAM)
            reported = backward.result().metrics
            optim_metrics = optim.result().metrics
        assert optim_metrics is not None, "the optimizer step returned no metrics"
        grad_norms.append(optim_metrics["grad_norm"])
        metrics[f"mt_opd/inner_update_{j}/grad_norm"] = optim_metrics["grad_norm"]
        metrics[f"mt_opd/inner_update_{j}/pg_loss"] = reported["pg_loss"]
        metrics[f"mt_opd/inner_update_{j}/ppo_kl"] = reported["ppo_kl"]
    metrics["actor/grad_norm"] = sum(grad_norms) / len(grad_norms)
    if refresh_advantage:
        # the reference's name for the refresh, present only when it ran
        metrics["mt_opd/m4_advantage_refreshed"] = 1.0
    return metrics


def main(config: Config) -> None:
    cli_utils.check_log_dir(config.log_path, behavior_if_exists=config.behavior_if_log_dir_exists)
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=config.wandb_project,
        wandb_name=config.wandb_name,
        config=config,
        do_configure_logging_module=True,
    )
    service = tinker.ServiceClient(
        base_url=config.base_url, user_metadata=recipe_user_metadata("recipe_mopd")
    )
    # rank 0 asks for full weights (README, Requirements). The teachers are never stepped.
    student = service.create_lora_training_client(base_model=STUDENT, rank=0).result()
    teachers = {
        domain: service.create_lora_training_client(base_model=name, rank=0).result()
        for domain, name in TEACHERS.items()
    }
    batches = (
        own_batches(config, student)
        if config.replay_batch is None
        else recorded_batches(config.replay_batch)
    )
    for step_index in range(config.n_steps):
        started = time.time()
        metrics: dict[str, float] = {}
        with timed("rollouts", metrics):
            batch = next(batches)
        metrics.update(train_step(student, teachers, batch, config.refresh_advantage))
        metrics["time/total"] = time.time() - started
        ml_logger.log_metrics(metrics, step=step_index)
    ml_logger.close()


if __name__ == "__main__":
    chz.nested_entrypoint(main)
