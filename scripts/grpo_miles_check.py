from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch


def _add_miles_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    miles_root = repo_root / "miles"
    sys.path.insert(0, str(miles_root))


def main() -> None:
    _add_miles_to_path()

    try:
        from miles.ray.rollout import RolloutManager
        from miles.utils.types import Sample
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "miles/ray dependencies are required to run this check (missing module)."
        ) from exc

    args = SimpleNamespace(
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=True,
        n_samples_per_prompt=2,
        rollout_batch_size=2,
        reward_key=None,
    )
    fake_manager = SimpleNamespace(args=args, custom_reward_post_process_func=None)

    samples = [Sample(reward=r) for r in [1.0, 3.0, 2.0, 6.0]]
    _raw_rewards, rewards = RolloutManager._post_process_rewards(fake_manager, samples)

    got = torch.tensor(rewards, dtype=torch.float32)
    expected = torch.tensor([-1.0, 1.0, -1.0, 1.0])
    assert torch.allclose(got, expected, atol=1e-6), (got, expected)
    print("OK", got.tolist())


if __name__ == "__main__":
    main()
