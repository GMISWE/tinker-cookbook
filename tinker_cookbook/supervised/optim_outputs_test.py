"""The supervised loop against the optimizer-step response of the published Tinker SDK."""

import asyncio
from types import SimpleNamespace
from typing import TypeVar
from unittest.mock import MagicMock

import chz
import pytest
import tinker
import torch

from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.types import SupervisedDataset, SupervisedDatasetBuilder

T = TypeVar("T")
LOGPROBS = [-0.5, -1.0, -2.0]


async def _resolved(value: T) -> T:
    return value


def _logprobs() -> dict[str, tinker.TensorData]:
    return {"logprobs": tinker.TensorData.from_torch(torch.tensor(LOGPROBS))}


class _Dataset(SupervisedDataset):
    def __len__(self) -> int:
        return 1

    def set_epoch(self, seed: int = 0) -> None:
        pass

    def get_batch(self, index: int) -> list[tinker.Datum]:
        return [
            tinker.Datum(
                model_input=tinker.ModelInput.from_ints([1, 2, 3]),
                loss_fn_inputs={
                    "target_tokens": tinker.TensorData.from_torch(torch.tensor([2, 3, 4])),
                    "weights": tinker.TensorData.from_torch(torch.ones(3)),
                },
            )
        ]


@chz.chz
class _DatasetBuilder(SupervisedDatasetBuilder):
    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        return _Dataset(), None


class _TrainingClient:
    def __init__(self, deferred: bool):
        self.deferred = deferred

    def forward_backward(self, data: list[tinker.Datum], loss_fn: str):
        return _resolved(
            tinker.ForwardBackwardOutput(
                loss_fn_output_type="CrossEntropyLossReturn",
                loss_fn_outputs=[] if self.deferred else [_logprobs()],
                metrics={},
            )
        )

    def optim_step(self, adam_params: tinker.AdamParams):
        if self.deferred:
            # a backend that defers the loss returns the logprobs with the step
            return _resolved(SimpleNamespace(metrics={}, loss_fn_outputs=[_logprobs()]))
        # the published SDK's OptimStepResponse declares metrics only
        return _resolved(tinker.OptimStepResponse(metrics={}))


class _ServiceClient:
    deferred = False

    def __init__(self, **kwargs: object):
        pass

    def __enter__(self) -> "_ServiceClient":
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    async def create_lora_training_client(self, **kwargs: object) -> _TrainingClient:
        return _TrainingClient(self.deferred)


class _CheckpointManager:
    def __init__(self, **kwargs: object):
        pass

    def restore_last_saved_tokens(self, tokens: int) -> None:
        pass

    async def maybe_save_async(self, **kwargs: object) -> None:
        pass

    async def save_final_async(self, **kwargs: object) -> None:
        pass


@pytest.mark.parametrize("deferred", [False, True])
def test_train_mean_nll_reads_whichever_response_carries_the_logprobs(
    tmp_path, monkeypatch, deferred: bool
):
    logged: list[dict[str, object]] = []
    ml_logger = MagicMock(store=None)
    ml_logger.get_logger_url.return_value = None
    ml_logger.log_metrics.side_effect = lambda metrics, step: logged.append(dict(metrics))
    monkeypatch.setattr(train.ml_log, "setup_logging", lambda **kwargs: ml_logger)
    monkeypatch.setattr(_ServiceClient, "deferred", deferred)
    monkeypatch.setattr(train.tinker, "ServiceClient", _ServiceClient)
    monkeypatch.setattr(train.checkpoint_utils, "CheckpointManager", _CheckpointManager)
    monkeypatch.setattr(train, "fetch_model_pricing", lambda name: None)
    monkeypatch.setattr(train, "get_tokenizer", lambda name: MagicMock())
    monkeypatch.setattr(train, "colorize_example", lambda datum, tokenizer: "")
    monkeypatch.setattr(train, "compute_bpb", lambda *args: 0.0)
    config = train.Config(
        log_path=str(tmp_path),
        model_name="Qwen/Qwen3-8B",
        recipe_name="test",
        dataset_builder=_DatasetBuilder(),
        eval_every=0,
        infrequent_eval_every=0,
    )
    asyncio.run(train.main(config))
    assert len(logged) == 1
    assert logged[0]["train_mean_nll"] == pytest.approx(-sum(LOGPROBS) / len(LOGPROBS))
