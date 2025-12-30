## GRPO Consistency Checks

This folder includes two lightweight scripts to compare GRPO reward normalization between
`miles` and `tinker-cookbook` using fixed inputs.

### What "match" means

Both scripts use the same fixed reward values and compute GRPO-style group normalization:
group centering + optional std normalization. A match means:

- The resulting normalized reward/advantage tensors are identical (within float tolerance).
- This confirms the core GRPO normalization behavior is aligned between the two codepaths.

This does **not** validate full training parity (e.g., sampling, loss functions, or KL terms).
It only checks that the GRPO normalization logic matches for a deterministic input.

### How to run

```bash
python scripts/grpo_miles_check.py
python scripts/grpo_cookbook_check.py
```

### Expected output

Both scripts should print (values ≈ ±1/√2 due to unbiased std with N-1 divisor):

```
OK [-0.7071067811865476, 0.7071067811865476, -0.7071067811865476, 0.7071067811865476]
```

If outputs differ, GRPO normalization behavior is not aligned between `miles` and
`tinker-cookbook` for the tested inputs.
