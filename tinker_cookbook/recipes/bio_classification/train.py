"""

## Bio classification — ESM2 protein secondary structure (LoRA token classification)

Fine-tunes a bio foundation model (ESM2, an encoder) for per-residue secondary
structure via a classification **head**. The head is declared by the model
directory's own config.json (`prepare_model.py` builds one from the HF
snapshot); the client sends nothing but the model path. Requires the server to
run the `automodel` backend and to read the
model directory.

```bash
python -m tinker_cookbook.recipes.bio_classification.train \
    model_name=facebook/esm2_t33_650M_UR50D task=q3 lora_rank=8
# reproduce the blog with the 3B model:
python -m tinker_cookbook.recipes.bio_classification.train \
    model_name=facebook/esm2_t36_3B_UR50D task=q3   # Q3 on CB513: see README "Expected"
    #                                        task=q8
```

Unlike `vlm_classifier` (verbalized, causal path), ESM2 is an encoder with no
generation, so this is a real classification head — it does not use the causal
supervised-learning machinery.
"""

import random
from pathlib import Path

import chz
import tinker
from tinker import types

from tinker_cookbook.recipes.bio_classification import data as ss_data
from tinker_cookbook.recipes.bio_classification import prepare_model


@chz.chz
class Config:
    model_name: str = "facebook/esm2_t33_650M_UR50D"
    task: str = "q3"  # q3 (3-state) | q8 (8-state)
    lora_rank: int = 8
    learning_rate: float = 5e-4
    num_epochs: int = 5
    batch_size: int = 8
    eval_batch_size: int = 16
    max_res: int = 1022  # + CLS + EOS = 1024
    model_dir: str | None = None  # head-declaring model dir; None = prepare one under the cache
    subset: int = 0  # 0 = full train (10,792)
    base_url: str | None = None


def evaluate(tc, data: list[types.Datum], bs: int, task: str) -> float:
    """Per-residue CB513 accuracy from the model's class logits."""
    correct = total = 0
    for i in range(0, len(data), bs):
        batch = data[i : i + bs]
        result = tc.forward(batch, "cross_entropy").result()
        for d, out in zip(batch, result.loss_fn_outputs):
            logits = out["logits"].to_numpy()  # [S, C]
            labels = d.loss_fn_inputs["labels"].to_numpy()  # [S]
            preds = logits[: len(labels)].argmax(-1)
            for p, lab in zip(preds, labels):
                if int(lab) == ss_data.IGNORE:
                    continue
                total += 1
                correct += int(int(p) == int(lab))
    if total == 0:
        raise ValueError("the eval set has no labelled positions")
    return correct / total


def main(cfg: Config):
    service = (
        tinker.ServiceClient(base_url=cfg.base_url) if cfg.base_url else tinker.ServiceClient()
    )
    labels = [name for name, _ in sorted(ss_data.label_map(cfg.task).items(), key=lambda kv: kv[1])]
    model_dir = (
        Path(cfg.model_dir)
        if cfg.model_dir
        else prepare_model.prepare_hf_model(
            cfg.model_name,
            "token",
            labels,
            prepare_model.default_out(cfg.model_name, "token", len(labels)),
        )
    )
    tc = service.create_lora_training_client(base_model=str(model_dir), rank=cfg.lora_rank)
    train_data, test_data = ss_data.load(cfg.model_name, cfg.task, cfg.max_res, cfg.subset)
    print(f"model={cfg.model_name} task={cfg.task} train={len(train_data)} test={len(test_data)}")

    acc = evaluate(tc, test_data, cfg.eval_batch_size, cfg.task)
    print(f"[epoch 0] CB513 {cfg.task} acc (init) = {acc:.4f}")
    best = acc
    for ep in range(cfg.num_epochs):
        random.Random(ep).shuffle(train_data)
        for i in range(0, len(train_data), cfg.batch_size):
            batch = train_data[i : i + cfg.batch_size]
            fb = tc.forward_backward(batch, "cross_entropy")
            opt = tc.optim_step(types.AdamParams(learning_rate=cfg.learning_rate))
            fb.result()
            opt.result()
        acc = evaluate(tc, test_data, cfg.eval_batch_size, cfg.task)
        best = max(best, acc)
        print(f"[epoch {ep + 1}] CB513 {cfg.task} acc = {acc:.4f} best={best:.4f}")
    print(f"\nFINAL CB513 {cfg.task} acc = {acc:.4f}  BEST = {best:.4f}")


if __name__ == "__main__":
    main(chz.entrypoint(Config))
