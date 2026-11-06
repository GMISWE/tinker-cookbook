"""Splice-site data for the Evo2 sequence-classification recipe.

Loads Nucleotide-Transformer `splice_sites_all` (30K/3K, 600 bp, 3-class),
tokenizes the DNA with the nucleotide tokenizer, and builds tinker Datums:
  model_input      = ModelInput.from_ints(dna_token_ids)
  loss_fn_inputs   = {"labels": TensorData(int64 [class_id])}   # ONE label per sequence (seq-cls)

Unlike the ESM2 per-residue recipe (data.py), labels are sequence-level. The
tokenizer is the recipe's nucleotide tokenizer (or any HF DNA tokenizer path).
"""

import datasets
import numpy as np
from tinker import types
from transformers import AutoTokenizer

NUM_LABELS = 3  # none / acceptor / donor
_REPO = "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised"


def load(
    tokenizer_path: str,
    max_len: int = 1024,
    subset: int = 0,
) -> tuple[list[types.Datum], list[types.Datum]]:
    """Return (train, test) Datums for splice_sites_all."""
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    ds = datasets.load_dataset(
        _REPO,
        data_files={"train": "splice_sites_all/train.parquet",
                    "test": "splice_sites_all/test.parquet"},
    )

    def build(split) -> list[types.Datum]:
        rows = ds[split]
        seqs, labs = list(rows["sequence"]), list(rows["label"])
        if subset and split == "train":
            seqs, labs = seqs[:subset], labs[:subset]
        out = []
        for s, y in zip(seqs, labs):
            ids = tok(s.upper(), add_special_tokens=True, truncation=True,
                      max_length=max_len)["input_ids"]
            out.append(types.Datum(
                model_input=types.ModelInput.from_ints(ids),
                loss_fn_inputs={"labels": types.TensorData.from_numpy(np.asarray([int(y)], dtype=np.int64))},
            ))
        return out

    return build("train"), build("test")
