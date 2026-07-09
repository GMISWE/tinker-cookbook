"""Splice-site data for the Evo2 sequence-classification recipe (feature 004 CS2).

Loads Nucleotide-Transformer `splice_sites_all` (30K/3K, 600 bp, 3-class),
tokenizes the DNA with the nucleotide tokenizer, and builds tinker Datums:
  model_input      = ModelInput.from_ints(dna_token_ids)
  loss_fn_inputs   = {"labels": [class_id]}   # ONE label per sequence (seq-cls)

Unlike the ESM2 per-residue recipe (data.py), labels are sequence-level. The
tokenizer is the recipe's nucleotide tokenizer (or any HF DNA tokenizer path).
"""
from typing import List, Tuple

import datasets
from transformers import AutoTokenizer

from tinker import types

NUM_LABELS = 3  # none / acceptor / donor
_REPO = "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised"


def load(
    tokenizer_path: str,
    max_len: int = 1024,
    subset: int = 0,
) -> Tuple[List[types.Datum], List[types.Datum]]:
    """Return (train, test) Datums for splice_sites_all."""
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    ds = datasets.load_dataset(
        _REPO,
        data_files={"train": "splice_sites_all/train.parquet",
                    "test": "splice_sites_all/test.parquet"},
    )

    def build(split) -> List[types.Datum]:
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
                loss_fn_inputs={"labels": [int(y)]},
            ))
        return out

    return build("train"), build("test")
