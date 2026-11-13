"""The recipe's data layout: where a response position sits in a datum, and how a recorded batch is read."""

import pytest
import torch

from tinker_cookbook.recipes.distillation.mopd import train


def _recorded(tmp_path, rows: int):
    """`rows` sequences in the reference's layout: prompts left-padded to 4, responses right-padded to 5."""
    prompt_lens = [1 + row % 4 for row in range(rows)]
    response_lens = [1 + row % 5 for row in range(rows)]
    attention = torch.zeros(rows, 9, dtype=torch.long)
    for row, (p, r) in enumerate(zip(prompt_lens, response_lens, strict=True)):
        attention[row, 4 - p : 4 + r] = 1
    tensors = {
        "prompts": torch.zeros(rows, 4, dtype=torch.long),
        "input_ids": torch.arange(rows * 9).reshape(rows, 9),
        "attention_mask": attention,
        "student_top_k_ids": torch.arange(rows * 5 * 16).reshape(rows, 5, 16),
    }
    path = tmp_path / "batch.pt"
    torch.save({"tensors": tensors, "domain": ["math", "code", "if", "code"] * (rows // 4)}, path)
    return str(path), prompt_lens, response_lens


def test_a_recorded_batch_keeps_each_rows_own_tokens_and_ids(tmp_path):
    """Padding on both sides is dropped; the shards are the recorded order cut in four."""
    path, prompt_lens, response_lens = _recorded(tmp_path, 16)
    batch = train.load_recorded_batch(path)
    assert [len(r.prompt) for r in batch.rollouts] == prompt_lens
    assert [len(r.response) for r in batch.rollouts] == response_lens
    third = batch.rollouts[3]
    assert third.prompt == [27, 28, 29, 30] and third.response == [31, 32, 33, 34]
    assert third.candidates.shape == (4, 16) and int(third.candidates[0, 0]) == 3 * 5 * 16
    assert batch.shards == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]


def test_a_recorded_batch_that_does_not_split_into_the_updates_is_refused(tmp_path):
    """12 rows give shards of 3, which four updates cannot share: the rows left over would never train."""
    path, _, _ = _recorded(tmp_path, 12)
    with pytest.raises(ValueError, match="is 12, not a multiple"):
        train.load_recorded_batch(path)


def test_a_batch_size_that_does_not_split_into_the_updates_is_refused():
    """20 prompts give shards of 5: one row of each would be sampled and never trained. Refused before
    any prompt is loaded."""
    config = train.Config(batch_size=20)
    with pytest.raises(ValueError, match="batch_size is 20"):
        next(train.own_batches(config, student=None))  # type: ignore[arg-type]


def test_a_datum_carries_the_ids_at_the_positions_that_predict_the_response():
    """Prompt of 3 and response of 2: input positions 2 and 3 predict the response, and the last token is
    no input."""
    rollout = train.Rollout(
        "math", [5, 6, 7], [8, 9], candidates=torch.zeros(2, 16, dtype=torch.long)
    )
    ids = torch.arange(100, 132).reshape(2, 16)
    datum = train.scoring_datum(rollout, ids, with_weights=True)
    assert datum.model_input.to_ints() == [5, 6, 7, 8]
    targets = datum.loss_fn_inputs["target_tokens"].to_torch()
    assert targets.shape == (4, 16)
    assert torch.equal(targets[2:], ids) and int(targets[:2].abs().sum()) == 0
    assert torch.equal(rollout.response_rows(targets), ids.float())
    assert "weights" not in train.scoring_datum(rollout, ids, with_weights=False).loss_fn_inputs
