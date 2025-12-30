"""
Implements RL on general MDPs
"""

import asyncio
import io
import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Callable, Coroutine, Iterable, Iterator, List, Literal, Sequence, TypeVar

from tqdm import tqdm

import chz
import numpy as np
import tinker
import torch
from tinker.types import LossFnType
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.display import colorize_example
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator, SamplingClientEvaluatorBuilder
from tinker_cookbook.rl.data_processing import (
    assemble_training_data,
    compute_advantages,
    remove_constant_reward_groups,
)
from tinker_cookbook.rl.metric_util import RLTestSetEvaluator, compute_trajectory_metrics
from tinker_cookbook.rl.metrics import (
    compute_post_kl,
    compute_sampling_client_metrics,
    incorporate_kl_penalty,
)
# Note: compute_kl_sample_train removed - now using ppo_kl from Miles (Megatron-vs-Megatron)
from tinker_cookbook.rl.rollouts import do_group_rollout
from tinker_cookbook.rl.adaptive_sampling import AdaptiveSamplingConfig
from tinker_cookbook.rl.reward_history import RewardHistory
from tinker_cookbook.rl.types import (
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    TrajectoryGroup,
)
from tinker_cookbook.tokenizer_utils import Tokenizer
from tinker_cookbook.utils import logtree, ml_log
from tinker_cookbook.utils.misc_utils import safezip, split_list, timed, all_same
from tinker_cookbook.utils.trace import scope, update_scope_context, trace_init

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def gather_with_progress(
    coroutines: Iterable[Coroutine[Any, Any, T]],
    desc: str,
) -> list[T]:
    """
    Run coroutines concurrently with a progress bar that updates as each completes.

    This preserves the order of results (like asyncio.gather) while providing
    real-time progress feedback as individual coroutines complete.
    """
    coroutine_list = list(coroutines)
    pbar = tqdm(total=len(coroutine_list), desc=desc)

    async def track(coro: Coroutine[Any, Any, T]) -> T:
        result = await coro
        pbar.update(1)
        return result

    try:
        results = await asyncio.gather(*[track(coro) for coro in coroutine_list])
    finally:
        pbar.close()

    return results


def _get_evaluator_name(evaluator: SamplingClientEvaluator) -> str:
    return (
        evaluator.name
        if isinstance(evaluator, RLTestSetEvaluator) and evaluator.name is not None
        else ""
    )


@contextmanager
def _get_logtree_scope(
    log_path: str | None, num_groups_to_log: int, f_name: str, scope_name: str
) -> Iterator[None]:
    """
    Creates a context manager; all log inside this context will be logged under the section `scope_name`.
    It will create a file with the path of log_path/f_name.html
    If num_groups_to_log is 0, it will disable logging (but note that this function does not actually implement the logic for logging itself!)
    """
    if log_path is not None and num_groups_to_log > 0:
        logtree_path = os.path.join(log_path, f"{f_name}.html")
        with logtree.init_trace(scope_name, path=logtree_path):
            yield
    else:
        yield


@scope
def _select_representative_inds(scores: list[float], num_inds: int) -> list[int]:
    assert num_inds <= len(scores)
    sorted_inds = np.argsort(scores)
    uniform_inds = np.linspace(0, len(sorted_inds) - 1, num_inds).astype(int)
    return [int(sorted_inds[i]) for i in uniform_inds]


@scope
def print_group(
    traj_group: TrajectoryGroup,
    tokenizer: Tokenizer,
    advantage_estimator: Literal["center", "grpo"] = "center",
    advantage_eps: float = 1e-6,
    grpo_reward_center: bool = True,
    grpo_std_normalization: bool = True,
):
    """
    Print a subset of the trajectory group to the console.
    """
    # Cut down the number of trajectories to print
    max_trajs_to_print = 4
    if len(traj_group.trajectories_G) > max_trajs_to_print:
        inds = _select_representative_inds(traj_group.get_total_rewards(), max_trajs_to_print)
        traj_group = TrajectoryGroup(
            trajectories_G=[traj_group.trajectories_G[i] for i in inds],
            final_rewards_G=[traj_group.final_rewards_G[i] for i in inds],
            metrics_G=[traj_group.metrics_G[i] for i in inds],
        )

    rewards = traj_group.get_total_rewards()
    advantages_G = compute_advantages(
        [traj_group],
        estimator=advantage_estimator,
        eps=advantage_eps,
        grpo_reward_center=grpo_reward_center,
        grpo_std_normalization=grpo_std_normalization,
    )
    data_D, metadata_D = assemble_training_data([traj_group], advantages_G)

    buf = io.StringIO()

    @scope
    def bprint(s: str):
        print(s, file=buf)

    bprint("\n====== Trajectory Group ======")
    last_metadata = None
    for datum, metadata in safezip(data_D, metadata_D):
        idx = metadata["traj_idx"]
        if metadata != last_metadata:
            bprint(f"****** trajectory idx={idx}, reward={rewards[idx]:.3g} ******")
            # Print trajectory-level metrics
            if traj_group.metrics_G[idx]:
                bprint("Trajectory metrics:")
                for key, value in traj_group.metrics_G[idx].items():
                    bprint(f"  {key}: {value}")
            # Print per-transition metrics
            transition_metrics = [
                transition.metrics
                for transition in traj_group.trajectories_G[idx].transitions
                if transition.metrics
            ]
            if transition_metrics:
                bprint("Per-step metrics:")
                for i, metrics in enumerate(transition_metrics):
                    bprint(f"  Step {i}:")
                    for key, value in metrics.items():
                        bprint(f"    {key}: {value}")
        bprint("---- datum ----")
        bprint(colorize_example(datum, tokenizer, key="advantages"))
        last_metadata = metadata
    bprint("====== End Trajectory Group ======")
    logger.info(buf.getvalue().rstrip())


def _remove_mask(datum: tinker.Datum) -> tinker.Datum:
    return tinker.Datum(
        model_input=datum.model_input,
        loss_fn_inputs={k: v for k, v in datum.loss_fn_inputs.items() if k != "mask"},
    )


def _training_logprobs_from_fwd_bwd(
    fwd_bwd_result: tinker.ForwardBackwardOutput,
) -> list[torch.Tensor]:
    return [output["logprobs"].to_torch() for output in fwd_bwd_result.loss_fn_outputs]


def _aggregate_metrics(
    aggregated: dict[str, float], batch_metrics: dict[str, Any]
) -> None:
    """Aggregate metrics from a batch into the accumulated metrics dict (in-place)."""
    for key, value in batch_metrics.items():
        if isinstance(value, (int, float)):
            aggregated[key] = aggregated.get(key, 0.0) + float(value)


@scope
async def train_step(
    data_D: List[tinker.Datum],
    training_client: tinker.TrainingClient,
    learning_rate: float,
    num_substeps: int,
    loss_fn: LossFnType,
) -> tuple[List[torch.Tensor], dict[str, Any]]:
    """Train the model on collected trajectories.

    Pipelines forward_backward and optim_step so they land on the same clock cycle.

    Returns:
        Tuple of (training_logprobs_D, aggregated_metrics) where metrics
        are summed/averaged across substeps from Miles.
    """
    batches = split_list(data_D, min(num_substeps, len(data_D)))

    # Handle empty batch - skip training step to avoid Miles assertion error
    if len(batches) == 0 or all(len(b) == 0 for b in batches):
        logger.warning("Empty batch in train_step - skipping training step")
        return [], {"train/skipped_empty_batch": 1}

    adam_params = tinker.AdamParams(learning_rate=learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)
    training_logprobs_D: list[torch.Tensor] = []
    aggregated_metrics: dict[str, float] = {}

    # Enqueue first batch
    fwd_bwd_future = await training_client.forward_backward_async(
        [_remove_mask(d) for d in batches[0]], loss_fn=loss_fn
    )
    optim_future = await training_client.optim_step_async(adam_params)

    for i in range(len(batches)):
        # Enqueue next batch before consuming current results (to stay on same clock cycle)
        if i + 1 < len(batches):
            next_fwd_bwd_future = await training_client.forward_backward_async(
                [_remove_mask(d) for d in batches[i + 1]], loss_fn=loss_fn
            )
            next_optim_future = await training_client.optim_step_async(adam_params)
        else:
            next_fwd_bwd_future = None
            next_optim_future = None
        # Consume current results
        fwd_bwd_result = await fwd_bwd_future.result_async()
        training_logprobs_D.extend(_training_logprobs_from_fwd_bwd(fwd_bwd_result))
        _aggregate_metrics(aggregated_metrics, fwd_bwd_result.metrics)
        await optim_future.result_async()
        # Move to next iteration
        if next_fwd_bwd_future is not None and next_optim_future is not None:
            fwd_bwd_future = next_fwd_bwd_future
            optim_future = next_optim_future

    return training_logprobs_D, aggregated_metrics


@chz.chz
class StreamMinibatchConfig:
    """
    Configuration for training with minibatch streaming.
    Once we have accumulated enough trajectories for a minibatch, we will
    immediately train on them, instead of waiting for the full batch of
    trajectories to be ready.
    """

    # Total number of trajectory groups across all minibatches and substeps
    groups_per_batch: int
    # For each substep, we will divide up the number of trajectory groups
    # into this many minibatches.
    # We will do num_minibatches forward_backward() passes and one optim_step()
    # per substep.
    num_minibatches: int


@chz.chz
class AsyncConfig:
    """Configuration for async RL training"""

    # If samples are generated from a sample more than this many steps ago,
    # we will skip training on them.
    max_steps_off_policy: int
    # We will ensure all batches have at least this many groups, even
    # as we discard stale samples
    groups_per_batch: int


@chz.chz
class Config:
    learning_rate: float
    dataset_builder: RLDatasetBuilder  # also determines batch size
    model_name: str
    max_tokens: int
    temperature: float = 1.0  # Changing sampling temperature is not generally recommended; does not currently play well with KL penalty
    compute_post_kl: bool = False
    evaluator_builders: list[SamplingClientEvaluatorBuilder] = chz.field(default_factory=list)
    lora_rank: int = 32

    kl_penalty_coef: float = 0.0
    kl_discount_factor: float = 0.0

    # Loss function to use for training: "importance_sampling" or "ppo"
    loss_fn: LossFnType = "importance_sampling"
    # Advantage estimator: "center" (default) or "grpo" (z-score within group)
    advantage_estimator: Literal["center", "grpo"] = "center"
    grpo_adv_eps: float = 1e-6
    grpo_reward_center: bool = True
    grpo_std_normalization: bool = True
    global_stat_est: bool = False

    # Number of optimizer steps per training iteration.
    # Useful for very large batch sizes.
    num_substeps: int = 1

    wandb_project: str | None = None
    wandb_name: str | None = None

    log_path: str = chz.field(munger=lambda _, s: os.path.expanduser(s))
    base_url: str | None = None
    enable_trace: bool = False
    debug_train_only: bool = False  # GMI: RL needs rollout actors for sampling (Slime-specific)

    # Max batch size for training API (avoids gradient accumulation issues with logprobs)
    # Should match groups_per_batch * group_size. Default None = auto-detect from dataset_builder.
    max_batch_size: int | None = None

    remove_constant_reward_groups: bool = False
    eval_every: int = 20  # 0 = disabled
    save_every: int = 20  # 0 = disabled
    load_checkpoint_path: str | None = None

    async_config: AsyncConfig | None = None
    stream_minibatch_config: StreamMinibatchConfig | None = None

    # Logtree configuration
    num_groups_to_log: int = 4  # Number of groups to log per iteration (0 = disable logging)

    # GMI-specific: RLVE configuration for GRPO native options (use-tis, balance-data, etc.)
    rlve_config: Any | None = None

    # Adaptive Sampling Configuration
    adaptive_sampling: AdaptiveSamplingConfig | None = None


@scope
async def run_single_evaluation(evaluator, cfg, i_batch, sampling_client):
    ev_name = _get_evaluator_name(evaluator)
    with _get_logtree_scope(
        log_path=cfg.log_path,
        num_groups_to_log=cfg.num_groups_to_log,
        f_name=f"eval_{ev_name}_iteration_{i_batch:06d}",
        scope_name=f"Running evaluation {ev_name} {i_batch}",
    ):
        eval_metrics = await evaluator(sampling_client)
        return eval_metrics


@scope
async def run_evaluations_parallel(
    evaluators: list[SamplingClientEvaluator],
    sampling_client: tinker.SamplingClient,
    cfg: Config,
    i_batch: int,
) -> dict[str, Any]:
    """Run all evaluators in parallel and return aggregated metrics."""

    # Create tasks for all evaluators with names for better traceability
    tasks = []
    for i, evaluator in enumerate(evaluators):
        ev_name = _get_evaluator_name(evaluator)
        task = asyncio.create_task(
            run_single_evaluation(evaluator, cfg, i_batch, sampling_client),
            name=f"eval_{ev_name or i}_iteration_{i_batch:06d}",
        )
        tasks.append(task)

    # Wait for all to complete
    results = await asyncio.gather(*tasks)

    # Merge all metrics
    metrics = {}
    for result in results:
        metrics.update(result)

    return metrics


@scope
async def do_sync_training_with_stream_minibatch(
    start_batch: int,
    end_batch: int,
    num_batches: int,
    cfg: Config,
    training_client: tinker.TrainingClient,
    service_client: tinker.ServiceClient,
    evaluators: list[SamplingClientEvaluator],
    dataset: RLDataset,
    ml_logger: ml_log.Logger,
    tokenizer: Tokenizer,
    reward_history: RewardHistory | None = None,
):
    """
    Implements fully synchronous on-policy training with minibatch streaming.
    Once we have accumulated enough trajectories for a minibatch, we will
    immediately train on them, instead of waiting for the full batch of
    trajectories to be ready. This allows us to overlap sampling and training.
    """
    # Initial sampling client
    sampling_client, _ = await save_checkpoint_and_get_sampling_client(
        training_client,
        start_batch,
        cfg.log_path,
        cfg.save_every,
        start_batch,
        dataset=dataset,
        reward_history=reward_history,
    )

    adaptive_sampler = None
    if cfg.adaptive_sampling and cfg.adaptive_sampling.enabled:
        from tinker_cookbook.rl.adaptive_sampling import AdaptiveSampler

        adaptive_sampler = AdaptiveSampler(
            config=cfg.adaptive_sampling,
            sampling_client=sampling_client,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
        )

    for i_batch in range(start_batch, end_batch):
        metrics = {
            "progress/batch": i_batch,
            "optim/lr": cfg.learning_rate,
            "progress/done_frac": (i_batch + 1) / num_batches,
        }
        t_start = time.time()

        # Run evaluations
        if (cfg.eval_every > 0 and i_batch % cfg.eval_every == 0) or i_batch == end_batch - 1:
            with timed("run_evals", metrics):
                eval_metrics = await run_evaluations_parallel(
                    evaluators, sampling_client, cfg, i_batch
                )
                metrics.update(eval_metrics)

        with _get_logtree_scope(
            cfg.log_path,
            cfg.num_groups_to_log,
            f"train_iteration_{i_batch:06d}",
            f"RL Iteration {i_batch}",
        ):
            # Samplers will produce trajectory groups asynchronously,
            # and the trainer will consume them as soon as they are ready
            trajectory_groups_queue = asyncio.Queue[WrappedTrajectoryGroup | None]()
            env_group_builders_P = dataset.get_batch(i_batch)

            @scope
            async def trajectory_group_worker_task(
                builder: EnvGroupBuilder, enable_logging: bool
            ) -> None:
                metrics = {}
                t_start = time.time()
                if adaptive_sampler:
                    trajectory_groups, adaptive_metrics = (
                        await adaptive_sampler.multi_round_adaptive_downsampling(
                            [builder],
                            reward_history=reward_history,
                        )
                    )
                    metrics.update(adaptive_metrics)
                    trajectory_group = trajectory_groups[0] if trajectory_groups else None
                else:
                    trajectory_group = await do_group_rollout_and_filter_constant_reward(
                        sampling_client,
                        builder,
                        max_tokens=cfg.max_tokens,
                        temperature=cfg.temperature,
                        do_remove_constant_reward_groups=cfg.remove_constant_reward_groups,
                        enable_logging=enable_logging,
                    )
                metrics["time/trajectory_group_worker_loop/total"] = time.time() - t_start
                if trajectory_group is not None:
                    trajectory_groups_queue.put_nowait(
                        WrappedTrajectoryGroup(
                            trajectory_group=trajectory_group,
                            env_group_builder=builder,
                            sampling_client_step=i_batch,
                            metrics=metrics,
                        )
                    )
                else:
                    trajectory_groups_queue.put_nowait(None)

            # Sample all trajectories asynchronously. If we have multiple minibatches,
            # then sampling can overlap with training.
            for i, builder in enumerate(env_group_builders_P):
                asyncio.create_task(
                    trajectory_group_worker_task(builder, enable_logging=i < cfg.num_groups_to_log),
                    name=f"trajectory_group_worker_task_{i}",
                )

            # Run multiple optimizer substeps per training iteration
            (
                sampling_client,
                full_batch_metrics,
            ) = await do_train_step_streaming_and_get_sampling_client(
                cfg,
                i_batch,
                trajectory_groups_queue,
                training_client,
                service_client,
                tokenizer,
                dataset=dataset,
                reward_history=reward_history,
            )

        # Log metrics
        metrics.update(full_batch_metrics)
        metrics["time/total"] = time.time() - t_start
        # Log RLVE per-environment metrics if available
        if hasattr(dataset, 'rlve_manager'):
            metrics.update(dataset.rlve_manager.get_metrics())
        ml_logger.log_metrics(metrics, step=i_batch)

        if adaptive_sampler:
            adaptive_sampler.sampling_client = sampling_client
            adaptive_sampler.policy = TinkerTokenCompleter(
                sampling_client, max_tokens=cfg.max_tokens, temperature=cfg.temperature
            )


@chz.chz
class WrappedTrajectoryGroup:
    """
    A wrapper around a trajectory group that includes metadata about how it was generated.
    Used when we need to overlap sampling and training.
    """

    trajectory_group: TrajectoryGroup
    # The env group builder that produced the trajectory group.
    # Pass this along in case the sampler is too stale, and we need to
    # requeue this group.
    env_group_builder: EnvGroupBuilder
    # The step that produced this trajectory group.
    sampling_client_step: int
    metrics: dict[str, Any] = chz.field(default_factory=dict)


@scope
async def do_async_training(
    start_batch: int,
    end_batch: int,
    num_batches: int,
    cfg: Config,
    training_client: tinker.TrainingClient,
    service_client: tinker.ServiceClient,
    evaluators: list[SamplingClientEvaluator],
    dataset: RLDataset,
    ml_logger: ml_log.Logger,
    tokenizer: Tokenizer,
    reward_history: RewardHistory | None = None,
):
    """Implements async off-policy training, capped at K steps off policy."""
    assert cfg.async_config is not None

    shutdown_event = asyncio.Event()
    # We will have groups_per_batch worker generating rollouts, so cap the
    # queue size to be groups_per_batch.
    env_group_builders_queue = asyncio.Queue[EnvGroupBuilder | None](
        maxsize=cfg.async_config.groups_per_batch
    )
    trajectory_groups_queue = asyncio.Queue[WrappedTrajectoryGroup | None]()

    # Initial sampling client to use
    sampling_client, _ = await save_checkpoint_and_get_sampling_client(
        training_client,
        start_batch,
        cfg.log_path,
        cfg.save_every,
        start_batch,
        dataset=dataset,
        reward_history=reward_history,
    )
    sampling_client_step = start_batch
    sampling_client_updated_event = asyncio.Event()
    sampling_client_updated_event.set()

    adaptive_sampler = None
    if cfg.adaptive_sampling and cfg.adaptive_sampling.enabled:
        from tinker_cookbook.rl.adaptive_sampling import AdaptiveSampler

        adaptive_sampler = AdaptiveSampler(
            config=cfg.adaptive_sampling,
            sampling_client=sampling_client,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
        )

    @scope
    def shutdown_loops():
        """Trigger all loops to shutdown"""
        shutdown_event.set()
        assert cfg.async_config is not None
        for _ in range(cfg.async_config.groups_per_batch):
            env_group_builders_queue.put_nowait(None)
        sampling_client_updated_event.set()

    @scope
    async def dataloader_loop():
        """Gets the next set of env builders to run"""
        i_batch = start_batch
        while not shutdown_event.is_set() and i_batch < end_batch:
            env_group_builders_P = dataset.get_batch(i_batch)
            for env_group_builder in env_group_builders_P:
                await env_group_builders_queue.put(env_group_builder)
            i_batch += 1

    @scope
    async def trajectory_group_worker_loop():
        """Generates trajectories for a single env builder"""
        while not shutdown_event.is_set():
            env_group_builder = await env_group_builders_queue.get()
            if env_group_builder is None:
                break

            metrics = {}
            t_start = time.time()
            # Save a reference to the sampling client step in case it changes
            # while we're running the rollout
            sampling_client_step_copy = sampling_client_step

            if adaptive_sampler:
                trajectory_groups, adaptive_metrics = (
                    await adaptive_sampler.multi_round_adaptive_downsampling(
                        [env_group_builder],
                        reward_history=reward_history,
                    )
                )
                metrics.update(adaptive_metrics)
                trajectory_group = trajectory_groups[0] if trajectory_groups else None
            else:
                trajectory_group = await do_group_rollout_and_filter_constant_reward(
                    sampling_client,
                    env_group_builder,
                    max_tokens=cfg.max_tokens,
                    temperature=cfg.temperature,
                    do_remove_constant_reward_groups=cfg.remove_constant_reward_groups,
                )
            if trajectory_group is None:
                trajectory_groups_queue.put_nowait(None)
            else:
                metrics["time/trajectory_group_worker_loop/total"] = time.time() - t_start
                trajectory_groups_queue.put_nowait(
                    WrappedTrajectoryGroup(
                        trajectory_group=trajectory_group,
                        env_group_builder=env_group_builder,
                        sampling_client_step=sampling_client_step_copy,
                        metrics=metrics,
                    )
                )

    @scope
    async def training_loop():
        """
        Waits for a sufficient number of valid trajectories to be accumulated and trains on them.
        Will discard trajectories that are too stale.
        """
        assert cfg.async_config is not None

        i_batch = start_batch
        wrapped_trajectory_groups = []
        while i_batch < end_batch:
            wrapped_trajectory_group = await trajectory_groups_queue.get()
            if wrapped_trajectory_group is None:
                continue

            @scope
            def filter_stale_trajectory_group(
                wrapped_trajectory_group: WrappedTrajectoryGroup | None,
            ) -> bool:
                """Returns False if the trajectory group is too stale or not valid"""
                if wrapped_trajectory_group is None:
                    return False

                # If the samples are too stale, requeue the data so that it will be used eventually.
                # Requeue on a separate coroutine to avoid blocking the training loop
                assert cfg.async_config is not None
                if (
                    i_batch - wrapped_trajectory_group.sampling_client_step
                    > cfg.async_config.max_steps_off_policy
                ):
                    logger.info(f"[training_loop] Step {i_batch}: Samples are too stale, skipping")
                    asyncio.create_task(
                        env_group_builders_queue.put(wrapped_trajectory_group.env_group_builder),
                        name="requeue_stale_sample_task",
                    )
                    return False
                return True

            metrics = {
                "training_client/step": i_batch,
                "optim/lr": cfg.learning_rate,
                "progress/done_frac": (i_batch + 1) / num_batches,
            }
            t_start = time.time()

            nonlocal sampling_client
            nonlocal sampling_client_step
            if cfg.stream_minibatch_config is not None:
                await trajectory_groups_queue.put(wrapped_trajectory_group)
                (
                    sampling_client,
                    train_step_metrics,
                ) = await do_train_step_streaming_and_get_sampling_client(
                    cfg,
                    i_batch,
                    trajectory_groups_queue,
                    training_client,
                    service_client,
                    tokenizer,
                    filter_stale_trajectory_group,
                    dataset,
                    reward_history=reward_history,
                )
            else:
                if not filter_stale_trajectory_group(wrapped_trajectory_group):
                    continue

                # Dynamic sampling: Wait for enough trajectories to accumulate to
                # ensure all batch sizes are the same size. This avoids needing to adjust
                # the learning rate for different batch sizes.
                wrapped_trajectory_groups.append(wrapped_trajectory_group)
                if len(wrapped_trajectory_groups) < cfg.async_config.groups_per_batch:
                    continue
                logger.info(
                    f"[training_loop] Step {i_batch}: Will train on batch, num groups: {len(wrapped_trajectory_groups)}"
                )

                # Compute sampling client metrics, as samples may have been generated with
                # different sampler versions
                metrics.update(compute_sampling_client_metrics(wrapped_trajectory_groups))

                adaptive_enabled_values = []
                first_round_reward_values = []
                for wrapped_traj in wrapped_trajectory_groups:
                    if "adaptive_sampling/enabled" in wrapped_traj.metrics:
                        adaptive_enabled_values.append(
                            wrapped_traj.metrics["adaptive_sampling/enabled"]
                        )
                    if "adaptive_sampling/first_round_reward" in wrapped_traj.metrics:
                        first_round_reward_values.append(
                            wrapped_traj.metrics["adaptive_sampling/first_round_reward"]
                        )

                if adaptive_enabled_values:
                    metrics["adaptive_sampling/enabled"] = all(adaptive_enabled_values)
                if first_round_reward_values:
                    metrics["adaptive_sampling/real_reward"] = float(
                        np.mean(first_round_reward_values)
                    )

                # TODO: For proper checkpointing, we also need to save dataloader state and
                # all queued trajectory groups that haven't been trained on yet
                sampling_client, train_step_metrics = await do_train_step_and_get_sampling_client(
                    cfg,
                    i_batch,
                    training_client,
                    service_client,
                    tokenizer,
                    [g.env_group_builder for g in wrapped_trajectory_groups],
                    [g.trajectory_group for g in wrapped_trajectory_groups],
                    dataset,
                    reward_history=reward_history,
                )
            sampling_client_step = i_batch + 1
            sampling_client_updated_event.set()

            if adaptive_sampler:
                adaptive_sampler.sampling_client = sampling_client
                adaptive_sampler.policy = TinkerTokenCompleter(
                    sampling_client, max_tokens=cfg.max_tokens, temperature=cfg.temperature
                )

            # Log metrics
            metrics.update(train_step_metrics)
            metrics["time/training_loop/total"] = time.time() - t_start
            # Log RLVE per-environment metrics if available
            if hasattr(dataset, 'rlve_manager'):
                metrics.update(dataset.rlve_manager.get_metrics())
            ml_logger.log_metrics(metrics, step=i_batch)
            i_batch += 1
            wrapped_trajectory_groups = []

        shutdown_loops()

    @scope
    async def evaluation_loop():
        """Runs evals periodically"""
        if len(evaluators) == 0 or cfg.eval_every == 0:
            return

        while not shutdown_event.is_set():
            await sampling_client_updated_event.wait()
            sampling_client_updated_event.clear()

            metrics = {}
            t_start = time.time()
            # Save a reference to the original values in case it changes
            # while we're running the evals
            sampling_client_eval_step = sampling_client_step
            sampling_client_eval = sampling_client
            if cfg.eval_every > 0 and sampling_client_eval_step % cfg.eval_every == 0:
                with timed("run_evals", metrics):
                    for evaluator in evaluators:
                        eval_metrics = await evaluator(sampling_client_eval)
                        metrics.update({f"test/{k}": v for k, v in eval_metrics.items()})
                metrics["time/evaluation_loop/total"] = time.time() - t_start
                ml_logger.log_metrics(metrics, step=sampling_client_eval_step)

    await asyncio.gather(
        asyncio.create_task(dataloader_loop(), name="dataloader_loop"),
        *[
            asyncio.create_task(
                trajectory_group_worker_loop(), name=f"trajectory_group_worker_loop_{i}"
            )
            for i in range(cfg.async_config.groups_per_batch)
        ],
        asyncio.create_task(training_loop(), name="training_loop"),
        asyncio.create_task(evaluation_loop(), name="evaluation_loop"),
    )


@scope
async def do_group_rollout_and_filter_constant_reward(
    sampling_client: tinker.SamplingClient,
    env_group_builder: EnvGroupBuilder,
    max_tokens: int,
    temperature: float,
    do_remove_constant_reward_groups: bool,
    enable_logging: bool = True,
    max_resample_attempts: int = 5,
) -> TrajectoryGroup | None:
    policy = TinkerTokenCompleter(sampling_client, max_tokens=max_tokens, temperature=temperature)

    current_builder = env_group_builder
    for attempt in range(max_resample_attempts):
        with logtree.optional_enable_logging(enable_logging and attempt == 0):
            trajectory_group = await do_group_rollout(current_builder, policy)

        # Remove if all trajectories have the same reward
        trajectory_groups = [trajectory_group]
        if do_remove_constant_reward_groups:
            trajectory_groups = remove_constant_reward_groups(trajectory_groups)

        if len(trajectory_groups) > 0:
            return trajectory_groups[0]

        # Group was filtered - try to resample with a new builder
        if attempt < max_resample_attempts - 1:
            try:
                current_builder = current_builder.clone()
                logger.info(
                    f"Group filtered (zero variance rewards), resampling... "
                    f"attempt {attempt + 2}/{max_resample_attempts}"
                )
            except NotImplementedError:
                # Builder doesn't support resampling
                logger.warning(
                    f"Group filtered but builder doesn't support clone() - cannot resample"
                )
                return None

    # Max retries exceeded - return the last trajectory group anyway to avoid empty batch
    logger.warning(
        f"Max resample attempts ({max_resample_attempts}) reached, "
        f"returning last group to avoid empty batch"
    )
    return trajectory_group


@scope
async def save_checkpoint_and_get_sampling_client(
    training_client: tinker.TrainingClient,
    i_batch: int,
    log_path: str,
    save_every: int,
    start_batch: int = 0,
    dataset: RLDataset | None = None,
    reward_history: RewardHistory | None = None,
) -> tuple[tinker.SamplingClient, dict[str, Any]]:
    metrics = {}
    with timed("save_checkpoint", metrics):
        # Save dataset state if the dataset has a save_state method
        if dataset is not None and hasattr(dataset, "save_state"):
            dataset_state_path = dataset.save_state(log_path, f"{i_batch:06d}")
            logger.info(f"Saved dataset state to {dataset_state_path}")

        if save_every > 0 and i_batch >= start_batch and i_batch % save_every == 0:
            path_dict = await checkpoint_utils.save_checkpoint_async(
                training_client=training_client,
                name=f"{i_batch:06d}",
                log_path=log_path,
                loop_state={"batch": i_batch},
                kind="both",
            )
            if reward_history is not None:
                reward_history_path = os.path.join(
                    log_path, f"reward_history_{i_batch:06d}.json"
                )
                reward_history.save(reward_history_path)

                latest_path = os.path.join(log_path, "reward_history_latest.json")
                reward_history.save(latest_path)

                stats = reward_history.get_stats()
                metrics.update({f"reward_history/{k}": v for k, v in stats.items()})
            return training_client.create_sampling_client(path_dict["sampler_path"]), metrics
        else:
            return await training_client.save_weights_and_get_sampling_client_async(), metrics


@scope
async def prepare_minibatch(
    env_group_builders_P: Sequence[EnvGroupBuilder],
    trajectory_groups_P: list[TrajectoryGroup],
    tokenizer: Tokenizer,
    service_client: tinker.ServiceClient,
    model_name: str,
    kl_penalty_coef: float,
    kl_discount_factor: float,
    advantage_estimator: Literal["center", "grpo"],
    advantage_eps: float,
    grpo_reward_center: bool,
    grpo_std_normalization: bool,
    use_global_stats: bool,
    reward_history: RewardHistory | None = None,
) -> tuple[list[tinker.Datum], dict[str, Any]]:
    """Converts the trajectories into a minibatch, and provides metrics about the minibatch"""

    # Compute trajectory metrics
    metrics = {}
    taglist_P = [env_group_builder.logging_tags() for env_group_builder in env_group_builders_P]
    metrics.update(compute_trajectory_metrics(trajectory_groups_P, taglist_P))

    if advantage_estimator == "grpo" and grpo_std_normalization:
        max_group_size = max(len(group.trajectories_G) for group in trajectory_groups_P)
        if max_group_size <= 1:
            grpo_std_normalization = False
            metrics["adv/grpo_std_normalization"] = 0.0
    if advantage_estimator == "grpo" and grpo_std_normalization:
        metrics["adv/grpo_std_normalization"] = 1.0

    # Print up to two trajectory groups
    for traj_group in trajectory_groups_P[:2]:
        print_group(
            traj_group,
            tokenizer,
            advantage_estimator=advantage_estimator,
            advantage_eps=advantage_eps,
            grpo_reward_center=grpo_reward_center,
            grpo_std_normalization=grpo_std_normalization,
        )

    # Assemble training data
    with timed("assemble_training_data", metrics):
        advantages_P = compute_advantages(
            trajectory_groups_P,
            estimator=advantage_estimator,
            eps=advantage_eps,
            grpo_reward_center=grpo_reward_center,
            grpo_std_normalization=grpo_std_normalization,
            use_global_stats=use_global_stats,
            env_group_builders_P=list(env_group_builders_P),
            reward_history=reward_history,
        )
        non_empty_advantages = [advantages for advantages in advantages_P if advantages.numel() > 0]
        if non_empty_advantages:
            flat_advantages = torch.cat(non_empty_advantages, dim=0)
            metrics["adv/mean"] = float(flat_advantages.mean())
            metrics["adv/std"] = float(flat_advantages.std(unbiased=False))
        data_D, _metadata_D = assemble_training_data(trajectory_groups_P, advantages_P)

    # Incorporate KL penalty if configured
    if kl_penalty_coef > 0:
        with timed("kl_vs_base", metrics):
            kl_penalty_metrics = await incorporate_kl_penalty(
                data_D,
                service_client.create_sampling_client(base_model=model_name),
                # ^^^ TODO: replace with the model we load, if relevant
                kl_penalty_coef,
                kl_discount_factor,
            )
        metrics.update(kl_penalty_metrics)

    return data_D, metrics


@scope
async def compute_full_batch_metrics_and_get_sampling_client(
    training_client: tinker.TrainingClient,
    i_batch: int,
    data_D: list[tinker.Datum],
    training_logprobs_D: list[torch.Tensor],
    miles_metrics: dict[str, Any],
    log_path: str,
    save_every: int,
    do_compute_post_kl: bool,
    dataset: RLDataset | None = None,
    reward_history: RewardHistory | None = None,
) -> tuple[tinker.SamplingClient, dict[str, Any]]:
    """
    At the end of the iteration, this will compute metrics for the full batch
    and return the latest sampling client.

    The reason we return a sampling client is that if do_compute_post_kl is True,
    we need to create a sampling client from the post-update policy.

    Args:
        miles_metrics: Metrics from Miles (ppo_kl, pg_clipfrac, etc.) - these are
            Megatron-vs-Megatron metrics, consistent with the training engine.
    """
    metrics = {}

    # Use ppo_kl from Miles (Megatron-vs-Megatron) instead of cross-engine KL
    # Miles computes: KL(π_megatron_old || π_megatron_new) - consistent!
    # Old approach computed: KL(π_sglang || π_megatron) - cross-engine, inconsistent
    if miles_metrics:
        # Map Miles metric names to expected output names
        if "ppo_kl:sum" in miles_metrics:
            metrics["optim/ppo_kl"] = miles_metrics["ppo_kl:sum"]
        if "pg_clipfrac:mean" in miles_metrics:
            metrics["optim/pg_clipfrac"] = miles_metrics["pg_clipfrac:mean"]
        if "pg_loss:sum" in miles_metrics:
            metrics["optim/pg_loss"] = miles_metrics["pg_loss:sum"]
        if "entropy_loss:sum" in miles_metrics:
            metrics["optim/entropy_loss"] = miles_metrics["entropy_loss:sum"]
        if "total_loss:sum" in miles_metrics:
            metrics["optim/total_loss"] = miles_metrics["total_loss:sum"]

    # Get a sampling client using the new weights
    sampling_client, checkpoint_metrics = await save_checkpoint_and_get_sampling_client(
        training_client,
        i_batch,
        log_path,
        save_every,
        dataset=dataset,
        reward_history=reward_history,
    )
    metrics.update(checkpoint_metrics)

    # Compute post-KL metrics if configured
    if do_compute_post_kl:
        with timed("compute_post_kl", metrics):
            post_kl_metrics = await compute_post_kl(data_D, sampling_client)
            metrics.update(post_kl_metrics)

    return sampling_client, metrics


@scope
async def do_train_step_streaming_and_get_sampling_client(
    cfg: Config,
    i_batch: int,
    trajectory_groups_queue: asyncio.Queue[WrappedTrajectoryGroup | None],
    training_client: tinker.TrainingClient,
    service_client: tinker.ServiceClient,
    tokenizer: Tokenizer,
    trajectory_group_filter: Callable[[WrappedTrajectoryGroup | None], bool] = lambda _: True,
    dataset: RLDataset | None = None,
    reward_history: RewardHistory | None = None,
) -> tuple[tinker.SamplingClient, dict[str, Any]]:
    """
    As soon as we have enough trajectories for a minibatch, we will train on them.
    This allows us to overlap sampling and training.
    """
    assert cfg.stream_minibatch_config is not None
    assert cfg.stream_minibatch_config.groups_per_batch % cfg.num_substeps == 0, (
        f"{cfg.stream_minibatch_config.groups_per_batch=} must be divisible by {cfg.num_substeps=}"
    )
    # Number of groups across all minibatches in each optimizer substep
    groups_per_substep = cfg.stream_minibatch_config.groups_per_batch // cfg.num_substeps
    assert groups_per_substep % cfg.stream_minibatch_config.num_minibatches == 0, (
        f"{groups_per_substep} must be divisible by {cfg.stream_minibatch_config.num_minibatches=}"
    )
    # Number of groups per minibatch in each optimizer substep
    groups_per_minibatch = groups_per_substep // cfg.stream_minibatch_config.num_minibatches

    update_scope_context({"step": i_batch})

    metrics = {}

    # Run multiple optimizer substeps per training iteration
    all_data_D = []
    all_training_logprobs_D = []
    all_wrapped_trajectory_groups = []
    all_miles_metrics: dict[str, float] = {}  # Aggregate Miles metrics (ppo_kl, etc.)
    for i_substep in range(cfg.num_substeps):
        # Run multiple minibatches per substep
        # Once we have enough trajectories for a minibatch, train on them
        wrapped_trajectory_groups = []
        forward_backward_futures: list[tinker.APIFuture[tinker.ForwardBackwardOutput]] = []
        i_minibatch = 0
        while i_minibatch < cfg.stream_minibatch_config.num_minibatches:
            wrapped_trajectory_group = await trajectory_groups_queue.get()
            if not trajectory_group_filter(wrapped_trajectory_group):
                continue
            wrapped_trajectory_groups.append(wrapped_trajectory_group)

            if len(wrapped_trajectory_groups) < groups_per_minibatch:
                continue
            logger.info(
                f"[stream_minibatch] Step {i_batch}, Substep {i_substep}/{cfg.num_substeps}, Minibatch {i_minibatch}/{cfg.stream_minibatch_config.num_minibatches}: Will train on minibatch, num groups: {len(wrapped_trajectory_groups)}"
            )

            # Note: we may have removed trajectory groups that have the same reward.
            # To have the same results as the sync implementation, we will
            # remove these and train on a smaller batch.
            wrapped_trajectory_groups = [g for g in wrapped_trajectory_groups if g is not None]

            # Skip minibatch if all groups were filtered (zero variance rewards)
            if len(wrapped_trajectory_groups) == 0:
                logger.warning(
                    f"[stream_minibatch] Step {i_batch}, Substep {i_substep}, Minibatch {i_minibatch}: "
                    "All groups filtered (zero variance). Skipping minibatch."
                )
                i_minibatch += 1
                wrapped_trajectory_groups = []
                continue

            data_D, prepare_minibatch_metrics = await prepare_minibatch(
                [g.env_group_builder for g in wrapped_trajectory_groups],
                [g.trajectory_group for g in wrapped_trajectory_groups],
                tokenizer,
                service_client,
                model_name=cfg.model_name,
                kl_penalty_coef=cfg.kl_penalty_coef,
                kl_discount_factor=cfg.kl_discount_factor,
                advantage_estimator=cfg.advantage_estimator,
                advantage_eps=cfg.grpo_adv_eps,
                grpo_reward_center=cfg.grpo_reward_center,
                grpo_std_normalization=cfg.grpo_std_normalization,
                use_global_stats=cfg.global_stat_est,
                reward_history=reward_history,
            )
            metrics.update(prepare_minibatch_metrics)

            # Enqueue forward-backward (we'll await results after all minibatches are enqueued)
            with timed(f"train/fwd_bwd_substep_{i_substep}_mb_{i_minibatch}_enqueue", metrics):
                forward_backward_futures.append(
                    await training_client.forward_backward_async(
                        [_remove_mask(d) for d in data_D], loss_fn=cfg.loss_fn
                    )
                )
            all_data_D.extend(data_D)
            all_wrapped_trajectory_groups.extend(wrapped_trajectory_groups)
            i_minibatch += 1
            wrapped_trajectory_groups = []

        # Enqueue optim_step before awaiting results (so they land on same clock cycle)
        adam_params = tinker.AdamParams(
            learning_rate=cfg.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
        )
        with timed(f"train/optim_substep_{i_substep}_enqueue", metrics):
            optim_future = await training_client.optim_step_async(adam_params)

        # Now consume all forward-backward results
        for i_mb, fwd_bwd_future in enumerate(forward_backward_futures):
            with timed(f"train/fwd_bwd_substep_{i_substep}_mb_{i_mb}_consume", metrics):
                fwd_bwd_result = await fwd_bwd_future.result_async()
                all_training_logprobs_D.extend(_training_logprobs_from_fwd_bwd(fwd_bwd_result))
                _aggregate_metrics(all_miles_metrics, fwd_bwd_result.metrics)

        with timed(f"train/optim_substep_{i_substep}_consume", metrics):
            await optim_future.result_async()

    # Aggregate metrics across the entire batch
    metrics.update(compute_sampling_client_metrics(all_wrapped_trajectory_groups))
    metrics.update(
        compute_trajectory_metrics(
            [g.trajectory_group for g in all_wrapped_trajectory_groups],
            [g.env_group_builder.logging_tags() for g in all_wrapped_trajectory_groups],
        )
    )

    adaptive_enabled_values = []
    first_round_reward_values = []
    for wrapped_traj in all_wrapped_trajectory_groups:
        if "adaptive_sampling/enabled" in wrapped_traj.metrics:
            adaptive_enabled_values.append(wrapped_traj.metrics["adaptive_sampling/enabled"])
        if "adaptive_sampling/first_round_reward" in wrapped_traj.metrics:
            first_round_reward_values.append(
                wrapped_traj.metrics["adaptive_sampling/first_round_reward"]
            )

    if adaptive_enabled_values:
        metrics["adaptive_sampling/enabled"] = all(adaptive_enabled_values)
    if first_round_reward_values:
        metrics["adaptive_sampling/real_reward"] = float(np.mean(first_round_reward_values))
    (
        sampling_client,
        full_batch_metrics,
    ) = await compute_full_batch_metrics_and_get_sampling_client(
        training_client,
        # NOTE: saving the checkpoint as the i + 1 step
        i_batch + 1,
        all_data_D,
        all_training_logprobs_D,
        all_miles_metrics,  # Pass Miles metrics (ppo_kl, etc.)
        cfg.log_path,
        cfg.save_every,
        cfg.compute_post_kl,
        dataset,
        reward_history=reward_history,
    )
    metrics.update(full_batch_metrics)
    return sampling_client, metrics


@scope
async def do_train_step_and_get_sampling_client(
    cfg: Config,
    i_batch: int,
    training_client: tinker.TrainingClient,
    service_client: tinker.ServiceClient,
    tokenizer: Tokenizer,
    env_group_builders_P: Sequence[EnvGroupBuilder],
    trajectory_groups_P: list[TrajectoryGroup],
    dataset: RLDataset | None = None,
    reward_history: RewardHistory | None = None,
) -> tuple[tinker.SamplingClient, dict[str, Any]]:
    update_scope_context({"step": i_batch})

    metrics = {}
    data_D, prepare_minibatch_metrics = await prepare_minibatch(
        env_group_builders_P,
        trajectory_groups_P,
        tokenizer,
        service_client,
        model_name=cfg.model_name,
        kl_penalty_coef=cfg.kl_penalty_coef,
        kl_discount_factor=cfg.kl_discount_factor,
        advantage_estimator=cfg.advantage_estimator,
        advantage_eps=cfg.grpo_adv_eps,
        grpo_reward_center=cfg.grpo_reward_center,
        grpo_std_normalization=cfg.grpo_std_normalization,
        use_global_stats=cfg.global_stat_est,
        reward_history=reward_history,
    )
    metrics.update(prepare_minibatch_metrics)

    with timed("train", metrics):
        training_logprobs_D, miles_metrics = await train_step(
            data_D,
            training_client,
            cfg.learning_rate,
            cfg.num_substeps,
            cfg.loss_fn,
        )

    sampling_client, full_batch_metrics = await compute_full_batch_metrics_and_get_sampling_client(
        training_client,
        # NOTE: saving the checkpoint as the i + 1 step
        i_batch + 1,
        data_D,
        training_logprobs_D,
        miles_metrics,  # Pass Miles metrics (ppo_kl, etc.)
        cfg.log_path,
        cfg.save_every,
        cfg.compute_post_kl,
        dataset=dataset,
        reward_history=reward_history,
    )
    metrics.update(full_batch_metrics)

    return sampling_client, metrics


@scope
async def do_sync_training(
    start_batch: int,
    end_batch: int,
    num_batches: int,
    cfg: Config,
    training_client: tinker.TrainingClient,
    service_client: tinker.ServiceClient,
    evaluators: list[SamplingClientEvaluator],
    dataset: RLDataset,
    ml_logger: ml_log.Logger,
    tokenizer: Tokenizer,
    reward_history: RewardHistory | None = None,
):
    """Implements fully synchronous on-policy training"""
    # Initial sampling client
    sampling_client, _ = await save_checkpoint_and_get_sampling_client(
        training_client,
        start_batch,
        cfg.log_path,
        cfg.save_every,
        start_batch,
        dataset=dataset,
        reward_history=reward_history,
    )

    adaptive_sampler = None
    if cfg.adaptive_sampling and cfg.adaptive_sampling.enabled:
        from tinker_cookbook.rl.adaptive_sampling import AdaptiveSampler

        adaptive_sampler = AdaptiveSampler(
            config=cfg.adaptive_sampling,
            sampling_client=sampling_client,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
        )

    for i_batch in range(start_batch, end_batch):
        metrics = {
            "progress/batch": i_batch,
            "optim/lr": cfg.learning_rate,
            "progress/done_frac": (i_batch + 1) / num_batches,
        }
        t_start = time.time()

        # Run evaluations
        if cfg.eval_every > 0 and i_batch % cfg.eval_every == 0:
            with timed("run_evals", metrics):
                eval_metrics = await run_evaluations_parallel(
                    evaluators, sampling_client, cfg, i_batch
                )
                metrics.update(eval_metrics)

        # Get batch and sample trajectories
        env_group_builders_P = dataset.get_batch(i_batch)

        # Initialize logtree trace for this iteration if logging is enabled
        with _get_logtree_scope(
            log_path=cfg.log_path,
            num_groups_to_log=cfg.num_groups_to_log,
            f_name=f"train_iteration_{i_batch:06d}",
            scope_name=f"RL Iteration {i_batch}",
        ):
            if adaptive_sampler:
                trajectory_groups_P, adaptive_metrics = (
                    await adaptive_sampler.multi_round_adaptive_downsampling(
                        list(env_group_builders_P),
                        reward_history=reward_history,
                    )
                )
                metrics.update(adaptive_metrics)
            else:
                trajectory_groups_P = await gather_with_progress(
                    (
                        do_group_rollout_and_filter_constant_reward(
                            sampling_client,
                            builder,
                            max_tokens=cfg.max_tokens,
                            temperature=cfg.temperature,
                            do_remove_constant_reward_groups=cfg.remove_constant_reward_groups,
                            enable_logging=i < cfg.num_groups_to_log,
                        )
                        for i, builder in enumerate(env_group_builders_P)
                    ),
                    desc=f"Sampling batch {i_batch}",
                )
                trajectory_groups_P = [
                    trajectory_group
                    for trajectory_group in trajectory_groups_P
                    if trajectory_group is not None
                ]

        # Skip training step if all groups were filtered (zero variance rewards)
        if len(trajectory_groups_P) == 0:
            logger.warning(
                f"Batch {i_batch}: All {len(env_group_builders_P)} groups were filtered "
                "(zero variance rewards after max resample attempts). Skipping training step."
            )
            # Still need to get a sampling client for the next iteration
            sampling_client = await training_client.save_weights_for_sampler_async()
            train_step_metrics = {"train/skipped_empty_batch": 1}
        else:
            # Train step
            sampling_client, train_step_metrics = await do_train_step_and_get_sampling_client(
                cfg,
                i_batch,
                training_client,
                service_client,
                tokenizer,
                env_group_builders_P,
                trajectory_groups_P,
                dataset=dataset,
                reward_history=reward_history,
            )

        if adaptive_sampler:
            adaptive_sampler.sampling_client = sampling_client
            adaptive_sampler.policy = TinkerTokenCompleter(
                sampling_client, max_tokens=cfg.max_tokens, temperature=cfg.temperature
            )

        # Log metrics
        metrics.update(train_step_metrics)
        metrics["time/total"] = time.time() - t_start
        # Log RLVE per-environment metrics if available
        if hasattr(dataset, 'rlve_manager'):
            metrics.update(dataset.rlve_manager.get_metrics())
        ml_logger.log_metrics(metrics, step=i_batch)


@scope
async def main(
    cfg: Config,
):
    """Main training loop for MDP RL."""
    ml_logger = ml_log.setup_logging(
        log_dir=cfg.log_path,
        wandb_project=cfg.wandb_project,
        config=cfg,
        wandb_name=cfg.wandb_name,
    )
    if cfg.enable_trace:
        # Get and rename the current (main) task
        current_task = asyncio.current_task()
        if current_task is not None:
            current_task.set_name("main")
        trace_events_path = os.path.join(cfg.log_path, "trace_events.jsonl")
        logger.info(f"Tracing is enabled. Trace events will be saved to {trace_events_path}")
        logger.info(
            f"Run `python tinker_cookbook/utils/trace.py {trace_events_path} trace.json` and visualize in chrome://tracing or https://ui.perfetto.dev/"
        )
        trace_init(output_file=trace_events_path)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("pylatexenc").setLevel(logging.WARNING)

    resume_info = checkpoint_utils.get_last_checkpoint(cfg.log_path)
    if resume_info:
        start_batch = resume_info["batch"]
    else:
        start_batch = 0

    service_client = tinker.ServiceClient(base_url=cfg.base_url)
    if resume_info:
        # Resuming interrupted training - load optimizer state for proper continuation
        training_client = (
            await service_client.create_training_client_from_state_with_optimizer_async(
                resume_info["state_path"]
            )
        )
        logger.info(f"Resumed training from {resume_info['state_path']}")
    elif cfg.load_checkpoint_path:
        # Starting fresh from a checkpoint - load weights only (fresh optimizer)
        training_client = await service_client.create_training_client_from_state_async(
            cfg.load_checkpoint_path
        )
        logger.info(f"Loaded weights from {cfg.load_checkpoint_path}")
    else:
        # Calculate max_batch_size from dataset_builder if not explicitly set
        max_batch_size = cfg.max_batch_size
        if max_batch_size is None:
            # Try to auto-detect from dataset_builder attributes
            batch_size = getattr(cfg.dataset_builder, 'batch_size', None)
            group_size = getattr(cfg.dataset_builder, 'group_size', None)
            if batch_size is not None and group_size is not None:
                max_batch_size = batch_size * group_size
                logger.info(f"Auto-detected max_batch_size={max_batch_size} (batch_size={batch_size} * group_size={group_size})")
            else:
                max_batch_size = 4096  # Default fallback
                logger.warning(f"Could not auto-detect batch size, using default max_batch_size={max_batch_size}")

        training_client = await service_client.create_lora_training_client_async(
            cfg.model_name,
            rank=cfg.lora_rank,
            debug_train_only=cfg.debug_train_only,
            max_batch_size=max_batch_size,
            rlve_config=cfg.rlve_config,
        )

    # Get tokenizer from training client
    tokenizer = training_client.get_tokenizer()

    # Create dataset from thunk
    dataset, maybe_test_dataset = await cfg.dataset_builder()
    evaluators = [evaluator() for evaluator in cfg.evaluator_builders]
    if maybe_test_dataset is not None:
        evaluators.append(RLTestSetEvaluator(maybe_test_dataset, max_tokens=cfg.max_tokens))

    reward_history = None
    if cfg.global_stat_est:
        reward_history = RewardHistory()
        logger.info("Using global statistics estimation for advantages")
        if resume_info:
            reward_history_path = os.path.join(
                cfg.log_path, f"reward_history_{start_batch:06d}.json"
            )
            if os.path.exists(reward_history_path):
                reward_history.load(reward_history_path)
            else:
                latest_path = os.path.join(cfg.log_path, "reward_history_latest.json")
                if os.path.exists(latest_path):
                    reward_history.load(latest_path)
                else:
                    logger.warning("No reward history found, starting fresh")

    # Load dataset state if resuming from checkpoint
    if resume_info and hasattr(dataset, "load_state"):
        # Construct the dataset state path from the checkpoint name
        checkpoint_name = resume_info.get("name", f"{resume_info['batch']:06d}")
        dataset_state_path = os.path.join(
            cfg.log_path, f"rlve_manager_{checkpoint_name}.json"
        )
        if os.path.exists(dataset_state_path):
            dataset.load_state(dataset_state_path)
            logger.info(f"Loaded dataset state from {dataset_state_path}")
        else:
            logger.warning(
                f"Dataset state file not found at {dataset_state_path}, starting with fresh state"
            )

    num_batches = len(dataset)
    logger.info(f"Will train on {num_batches} batches")

    # Training loop
    if cfg.async_config is not None:
        training_func = do_async_training
    elif cfg.stream_minibatch_config is not None:
        training_func = do_sync_training_with_stream_minibatch
    else:
        training_func = do_sync_training
    await training_func(
        start_batch=start_batch,
        end_batch=num_batches,
        num_batches=num_batches,
        cfg=cfg,
        training_client=training_client,
        service_client=service_client,
        evaluators=evaluators,
        dataset=dataset,
        ml_logger=ml_logger,
        tokenizer=tokenizer,
        reward_history=reward_history,
    )

    # Save final checkpoint
    if start_batch < num_batches:
        _ = await checkpoint_utils.save_checkpoint_async(
            training_client=training_client,
            name="final",
            log_path=cfg.log_path,
            kind="both",
            loop_state={"batch": num_batches},
        )
        if reward_history is not None:
            final_reward_history_path = os.path.join(cfg.log_path, "reward_history_final.json")
            reward_history.save(final_reward_history_path)
    else:
        logger.info("Training was already complete; nothing to do")

    # Cleanup
    ml_logger.close()
    logger.info("Training completed successfully")
