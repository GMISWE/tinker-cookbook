from __future__ import annotations

import torch
import tinker

from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl.data_processing import compute_advantages
from tinker_cookbook.rl.types import Trajectory, TrajectoryGroup, Transition


def _make_traj(reward: float) -> Trajectory:
    ob = tinker.ModelInput.from_ints([1])
    ac = TokensWithLogprobs(tokens=[2], maybe_logprobs=[-0.1])
    transition = Transition(ob=ob, ac=ac, reward=reward, episode_done=True)
    return Trajectory(transitions=[transition], final_ob=ob)


def main() -> None:
    group1 = TrajectoryGroup(
        trajectories_G=[_make_traj(1.0), _make_traj(3.0)],
        final_rewards_G=[0.0, 0.0],
        metrics_G=[{}, {}],
    )
    group2 = TrajectoryGroup(
        trajectories_G=[_make_traj(2.0), _make_traj(6.0)],
        final_rewards_G=[0.0, 0.0],
        metrics_G=[{}, {}],
    )

    advantages = compute_advantages(
        [group1, group2],
        estimator="grpo",
        eps=1e-6,
        grpo_reward_center=True,
        grpo_std_normalization=True,
    )
    got = torch.cat(advantages)
    # With unbiased=True std (N-1 divisor), 2-sample groups give 1/sqrt(2) ≈ 0.7071
    expected = torch.tensor([-0.7071067811865476, 0.7071067811865476, -0.7071067811865476, 0.7071067811865476])
    assert torch.allclose(got, expected, atol=1e-5), (got, expected)
    print("OK", got.tolist())


if __name__ == "__main__":
    main()
