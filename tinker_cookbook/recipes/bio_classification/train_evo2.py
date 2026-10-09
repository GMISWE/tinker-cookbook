"""Evo2-1B DNA splice-site **sequence classification** via the tinker SDK
(feature 004 CS2). Drives the faithful bionemo-recipes Evo2 classifier through
TinkerCloud's `megatron_bridge` backend over the wire, no client-side model code.

The head is declared beside the checkpoint (`prepare_model.py checkpoint_dir=...
head=sequence labels=none,acceptor,donor`); the recipe's own knobs (tokenizer,
splits, seq_length, iters, lr) are server configuration, `MEGATRON_BRIDGE_*`.

Requires the server on the megatron_bridge backend (cu13 recipe env):
    deploy_tinkercloud.sh --profile megatron_bridge      # then start the server

    python -m tinker_cookbook.recipes.bio_classification.train_evo2 \
        base_url=http://localhost:8200 model_dir=/data/evo2_1b_mbridge \
        tokenizer_path=/workspace/evo2_megatron/tokenizers/nucleotide_fast_tokenizer_512

Pin the known-good server config (28500 train, 1000 iters, lr 5e-4, r16, mean-pool);
reproduces ~0.858 test acc (blog LoRA+head 0.966). See P5-TINKER-BACKEND.md.
"""
import random

import chz
import tinker
from tinker import types

from tinker_cookbook.recipes.bio_classification import data_evo2 as sd


@chz.chz
class Config:
    model_dir: str = "/data/evo2_1b_mbridge"          # MBridge checkpoint + head-declaring config.json
    tokenizer_path: str = "/workspace/evo2_megatron/tokenizers/nucleotide_fast_tokenizer_512"
    lora_rank: int = 16
    learning_rate: float = 5e-4
    num_epochs: int = 3
    batch_size: int = 8
    eval_batch_size: int = 16
    seq_length: int = 1024
    subset: int = 6000                                # 0 = full 30K
    eval_subset: int = 0                              # 0 = full 3K test
    base_url: str | None = None


def evaluate(tc, data: list[types.Datum], bs: int) -> float:
    correct = total = 0
    for i in range(0, len(data), bs):
        batch = data[i:i + bs]
        result = tc.forward(batch, "cross_entropy").result()
        for d, out in zip(batch, result.loss_fn_outputs):
            logits = out["logits"].to_numpy()              # [C]
            label = int(d.loss_fn_inputs["labels"].to_numpy().reshape(-1)[0])
            total += 1
            correct += int(int(logits.argmax(-1)) == label)
    return correct / max(total, 1)


def main(cfg: Config):
    service = (tinker.ServiceClient(base_url=cfg.base_url)
               if cfg.base_url else tinker.ServiceClient())
    tc = service.create_lora_training_client(base_model=cfg.model_dir, rank=cfg.lora_rank)
    train_data, test_data = sd.load(cfg.tokenizer_path, cfg.seq_length, cfg.subset)
    print(f"model={cfg.model_dir} train={len(train_data)} test={len(test_data)}")

    eval_data = test_data[: cfg.eval_subset] if cfg.eval_subset else test_data
    acc = evaluate(tc, eval_data, cfg.eval_batch_size)
    best = acc
    print(f"[epoch 0] splice test acc (init) = {acc:.4f}")
    for ep in range(cfg.num_epochs):
        random.Random(ep).shuffle(train_data)
        for i in range(0, len(train_data), cfg.batch_size):
            batch = train_data[i:i + cfg.batch_size]
            fb = tc.forward_backward(batch, "cross_entropy")
            opt = tc.optim_step(types.AdamParams(learning_rate=cfg.learning_rate))
            fb.result()
            opt.result()
        acc = evaluate(tc, eval_data, cfg.eval_batch_size)
        best = max(best, acc)
        print(f"[epoch {ep + 1}] splice test acc = {acc:.4f} best={best:.4f}")
    print(f"\nFINAL splice acc = {acc:.4f}  BEST = {best:.4f}")


if __name__ == "__main__":
    main(chz.entrypoint(Config))
