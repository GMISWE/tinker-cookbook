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


def test_compute_advantages_center():
    group = TrajectoryGroup(
        trajectories_G=[_make_traj(1.0), _make_traj(3.0), _make_traj(5.0)],
        final_rewards_G=[0.0, 0.0, 0.0],
        metrics_G=[{}, {}, {}],
    )
    advantages = compute_advantages([group])[0]
    expected = torch.tensor([-2.0, 0.0, 2.0])
    assert torch.allclose(advantages, expected, atol=1e-6)


def test_compute_advantages_grpo():
    group = TrajectoryGroup(
        trajectories_G=[_make_traj(1.0), _make_traj(3.0), _make_traj(5.0)],
        final_rewards_G=[0.0, 0.0, 0.0],
        metrics_G=[{}, {}, {}],
    )
    advantages = compute_advantages([group], estimator="grpo", eps=1e-6)[0]
    expected = torch.tensor([-1.2247449, 0.0, 1.2247449])
    assert torch.allclose(advantages, expected, atol=1e-5)


def test_compute_advantages_grpo_no_std_norm():
    group = TrajectoryGroup(
        trajectories_G=[_make_traj(1.0), _make_traj(3.0), _make_traj(5.0)],
        final_rewards_G=[0.0, 0.0, 0.0],
        metrics_G=[{}, {}, {}],
    )
    advantages = compute_advantages(
        [group], estimator="grpo", eps=1e-6, grpo_std_normalization=False
    )[0]
    expected = torch.tensor([-2.0, 0.0, 2.0])
    assert torch.allclose(advantages, expected, atol=1e-6)
