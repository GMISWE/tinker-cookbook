import itertools
import logging

import tinker
from tinker_cookbook.eval.evaluators import TrainingClientEvaluator
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_cookbook.supervised.types import SupervisedDataset

logger = logging.getLogger(__name__)


class NLLEvaluator(TrainingClientEvaluator):
    EVAL_BATCH_SIZE = 64  # Conservative batch size that works with Miles

    def __init__(self, data: list[tinker.Datum], name: str = "test"):
        self.name = name
        self.data = data

    async def __call__(self, training_client: tinker.TrainingClient) -> dict[str, float]:
        # Process in batches to avoid Miles batch size limitations
        # Miles uses floor division which can drop samples if batch doesn't divide evenly
        all_logprobs = []

        for i in range(0, len(self.data), self.EVAL_BATCH_SIZE):
            batch = self.data[i : i + self.EVAL_BATCH_SIZE]
            future = await training_client.forward_async(batch, loss_fn="cross_entropy")
            result = await future.result_async()
            batch_logprobs = [x["logprobs"] for x in result.loss_fn_outputs]
            all_logprobs.extend(batch_logprobs)

        weights = [datum.loss_fn_inputs["weights"] for datum in self.data]

        # Defensive check
        if len(all_logprobs) != len(weights):
            logger.warning(
                f"NLLEvaluator: logprobs count ({len(all_logprobs)}) != weights count ({len(weights)})"
            )
            min_len = min(len(all_logprobs), len(weights))
            all_logprobs = all_logprobs[:min_len]
            weights = weights[:min_len]

        nll = compute_mean_nll(all_logprobs, weights)
        key = f"{self.name}/nll"
        return {key: nll}

    @classmethod
    def from_dataset(cls, dataset: SupervisedDataset, name: str = "test") -> "NLLEvaluator":
        all_data = list(itertools.chain(*[dataset.get_batch(i) for i in range(len(dataset))]))
        return cls(all_data, name=name)
