# Bio Classification — ESM2 protein secondary structure

LoRA fine-tuning of a **bio foundation model** (ESM2) for protein **secondary
structure** prediction — per-residue **token classification** (Q3 / Q8), through
TinkerCloud's classification objective (feature 004).

Unlike [`vlm_classifier`](../vlm_classifier) (which is *verbalized* — a causal VLM
generating class names as text), ESM2 is an **encoder** with no generation, so
this recipe uses a real classification **head**. The head is declared by the
model directory's own `config.json` (`architectures` ending in
`ForTokenClassification`, explicit `num_labels` / `id2label`); the client sends
only the model path. The class ids ride in `loss_fn_inputs["labels"]` as an
int64 `TensorData`; no new wire type.

## Requirements

- The TinkerCloud server must run the **`automodel`** backend:
  `TINKERCLOUD_BACKEND=automodel`. ESM2 loads as `EsmForTokenClassification`
  (encoder), which the LM-only backends (miles / nemo_rl) reject.
- A model directory the server can read. `train.py` prepares one under
  `~/.cache/tinker-cookbook/bio_models/` when `model_dir` is unset, or build it
  yourself:

  ```bash
  python -m tinker_cookbook.recipes.bio_classification.prepare_model \
      base=facebook/esm2_t33_650M_UR50D head=token labels=C,E,H out=/data/models/esm2-650m-q3
  ```

- Evo2 (`train_evo2.py`, `megatron_bridge` backend): declare the head beside the
  Megatron-Bridge checkpoint with `prepare_model.py checkpoint_dir=/data/evo2_1b_mbridge
  head=sequence labels=none,acceptor,donor`; the recipe's training knobs are the
  server's `MEGATRON_BRIDGE_*` configuration.

## Run

```bash
# quick (650M, ~3 GB): Q3 3-state
python -m tinker_cookbook.recipes.bio_classification.train \
    model_name=facebook/esm2_t33_650M_UR50D task=q3 lora_rank=8

# reproduce the blog with ESM2-3B on CB513:
python -m tinker_cookbook.recipes.bio_classification.train \
    model_name=facebook/esm2_t36_3B_UR50D task=q3   # Q3 ~0.85 (blog 0.848)
python -m tinker_cookbook.recipes.bio_classification.train \
    model_name=facebook/esm2_t36_3B_UR50D task=q8   # Q8 ~0.77 (blog 0.743)
```

`task=q3` is 3-state (C/E/H), `task=q8` is 8-state. Config knobs (chz):
`lora_rank`, `learning_rate` (default 5e-4), `num_epochs`, `batch_size`,
`model_dir`, `subset` (0 = full 10,792), `base_url`.

## Data

`proteinea/secondary_structure_prediction` — train `training_hhblits.csv`
(NetSurfP, 10,792), test **CB513** `CB513.csv` (511). `dssp3`/`dssp8` are
per-residue label strings aligned 1:1 with the amino-acid `input`. Labels are
tokenized to `-100` at CLS/EOS so they're ignored in the loss / accuracy.

## Expected

CB513 per-residue accuracy converges in 2–3 epochs, then mildly overfits — the
reported number is the best epoch. ESM2-3B reaches ~0.88 Q3 / ~0.77 Q8, meeting
or exceeding the BioNeMo blog's 0.848 / 0.743.
