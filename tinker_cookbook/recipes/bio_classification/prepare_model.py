"""Build a model directory whose config.json declares the classification head.

TinkerCloud reads the head from the model's own config (`architectures` ending in
ForSequenceClassification / ForTokenClassification, plus an explicit `num_labels`
and `id2label`): the client declares nothing. For an HF encoder (ESM2) the
directory holds symlinks to the downloaded snapshot plus a rewritten config.json;
for a Megatron-Bridge checkpoint (Evo2) it adds a minimal config.json beside the
checkpoint files. The server must be able to read the directory.

    python -m tinker_cookbook.recipes.bio_classification.prepare_model \
        base=facebook/esm2_t33_650M_UR50D head=token labels=C,E,H out=/data/models/esm2-650m-q3
"""

import json
import os
from pathlib import Path

import chz

HEAD_SUFFIX = {"token": "ForTokenClassification", "sequence": "ForSequenceClassification"}


def head_config(base_config: dict, head: str, labels: list[str]) -> dict:
    """The base config with the head declared: architecture renamed, labels explicit."""
    archs = base_config.get("architectures") or []
    if archs:
        stem = archs[0].split("For", 1)[0]
    else:
        stem = str(base_config["model_type"]).capitalize()
    cfg = dict(base_config)
    cfg["architectures"] = [stem + HEAD_SUFFIX[head]]
    cfg["num_labels"] = len(labels)
    cfg["id2label"] = {str(i): name for i, name in enumerate(labels)}
    cfg["label2id"] = {name: i for i, name in enumerate(labels)}
    return cfg


def prepare_hf_model(base: str, head: str, labels: list[str], out: Path) -> Path:
    """Symlink the HF snapshot into `out` and write the head-declaring config.json."""
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(base))
    out.mkdir(parents=True, exist_ok=True)
    for src in snapshot.iterdir():
        if src.name == "config.json":
            continue
        dst = out / src.name
        if not dst.exists():
            os.symlink(src.resolve(), dst)
    with (snapshot / "config.json").open() as f:
        base_config = json.load(f)
    with (out / "config.json").open("w") as f:
        json.dump(head_config(base_config, head, labels), f, indent=2)
    return out


def prepare_checkpoint_dir(
    checkpoint_dir: Path, model_type: str, head: str, labels: list[str]
) -> Path:
    """Declare the head beside a non-HF checkpoint (Evo2 under Megatron-Bridge)."""
    cfg = head_config({"model_type": model_type}, head, labels)
    with (checkpoint_dir / "config.json").open("w") as f:
        json.dump(cfg, f, indent=2)
    return checkpoint_dir


@chz.chz
class Config:
    base: str
    head: str = "token"  # token | sequence
    labels: str = "C,E,H"  # comma-separated class names, in id order
    out: str = ""  # model directory to create; default under the cookbook cache
    checkpoint_dir: str = ""  # non-HF checkpoint (Evo2): declare the head in place
    model_type: str = "evo2"


def default_out(base: str, head: str, n: int) -> Path:
    return (
        Path.home()
        / ".cache"
        / "tinker-cookbook"
        / "bio_models"
        / f"{base.replace('/', '--')}-{head}{n}"
    )


def main(cfg: Config) -> None:
    labels = cfg.labels.split(",")
    if cfg.checkpoint_dir:
        out = prepare_checkpoint_dir(Path(cfg.checkpoint_dir), cfg.model_type, cfg.head, labels)
    else:
        out = prepare_hf_model(
            cfg.base,
            cfg.head,
            labels,
            Path(cfg.out) if cfg.out else default_out(cfg.base, cfg.head, len(labels)),
        )
    print(out)


if __name__ == "__main__":
    chz.nested_entrypoint(main)
