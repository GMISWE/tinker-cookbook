"""
Data for the bio_classification recipe — protein secondary structure.

ESM2 is an encoder, so this is real per-residue **token classification** (not
verbalized). Each amino acid is one token; the class ids ride in the Datum's
`loss_fn_inputs["labels"]` (an int64 TensorData), aligned 1:1 with the tokens, with -100 at
the CLS/EOS positions (the server converter has no CLS offset).

Dataset: proteinea/secondary_structure_prediction (NetSurfP train / CB513 test),
columns `input` (AA sequence), `dssp3` (Q3), `dssp8` (Q8) — all 1:1 per residue.
"""

import datasets
import numpy as np
from tinker import types
from transformers import AutoTokenizer

REPO = "proteinea/secondary_structure_prediction"
IGNORE = -100
Q3_MAP = {"C": 0, "E": 1, "H": 2}
Q8_MAP = {c: i for i, c in enumerate("BCEGHIST")}


LABEL_MAPS = {"q3": Q3_MAP, "q8": Q8_MAP}
COLUMNS = {"q3": "dssp3", "q8": "dssp8"}


def label_map(task: str) -> dict:
    return LABEL_MAPS[task]  # KeyError names an unknown task


def num_labels(task: str) -> int:
    return len(label_map(task))


def _col(task: str) -> str:
    return COLUMNS[task]


def _build_datum(tok, seq: str, ss: str, lmap: dict, max_res: int) -> types.Datum:
    seq, ss = seq[:max_res], ss[:max_res]
    ids = tok(seq)["input_ids"]  # [CLS, res..., EOS]
    labels = [IGNORE] * len(ids)
    for i, c in enumerate(ss):
        pos = 1 + i  # residue i -> token pos 1+i
        if 0 < pos < len(ids) - 1:
            labels[pos] = lmap[c]  # KeyError: a label outside the dataset's alphabet
    return types.Datum(
        model_input=types.ModelInput.from_ints(ids),
        loss_fn_inputs={"labels": types.TensorData.from_numpy(np.asarray(labels, dtype=np.int64))},
    )


def load(
    model_name: str,
    task: str,
    max_res: int = 1022,
    subset: int = 0,
) -> tuple[list[types.Datum], list[types.Datum]]:
    """Return (train_datums, cb513_test_datums)."""
    tok = AutoTokenizer.from_pretrained(model_name)
    ds = datasets.load_dataset(
        REPO, data_files={"train": "training_hhblits.csv", "test": "CB513.csv"}
    )
    lmap, col = label_map(task), _col(task)
    train = ds["train"].select(range(subset)) if subset else ds["train"]

    def build(split) -> list[types.Datum]:
        return [
            _build_datum(tok, s, ss, lmap, max_res)
            for s, ss in zip(list(split["input"]), list(split[col]))
        ]

    return build(train), build(ds["test"])
