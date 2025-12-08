# GMI Wrapper Test Scripts

This directory contains test scripts for the GMI Wrapper (kgateway) integration with Miles RL training cluster.

## Architecture

```
Tinker Client → GMI Wrapper (kgateway) → Miles Instance
```

- **Tinker Client**: Uses tinker-cookbook recipes to run RL training
- **GMI Wrapper**: kgateway service that implements Tinker API endpoints
- **Miles Instance**: Ray-based distributed training cluster with Megatron backend

## Scripts

### test_math_rl_arithmetic.sh

Runs a successful math RL training test with arithmetic environment.

**Usage:**
```bash
./test_math_rl_arithmetic.sh
```

**Prerequisites:**
1. Kubernetes cluster with miles-gmi-tinker namespace
2. miles-training-0 pod running with:
   - Ray cluster initialized (DP=2, PP=2, TP=1)
   - Model: `/data/models/Qwen2.5-0.5B-Instruct`
   - tinker_gmi and tinker-cookbook packages installed
3. kgateway-training deployment running
4. Service: `kgateway-training.miles-gmi-tinker:8000`

**Parameters:**
- `group_size=4`: 4 samples per trajectory group
- `groups_per_batch=2`: 2 groups = 8 total samples
- `num_substeps=1`: **Critical fix** - send all samples in one batch
- `learning_rate=2e-4`: Adam learning rate
- `max_tokens=64`: Maximum generation length

## Batch Size Fix: num_substeps=1

### The Problem

**Symptom:**
```
AssertionError at miles/backends/megatron_utils/data.py:231
assert len(set(sum(micro_batch_indices, []))) == num_local_samples
```

**Root Cause:**

The `num_substeps` parameter controls how Tinker client splits batches before sending to the server:

```python
# tinker-cookbook/tinker_cookbook/rl/train.py:160
batches_md = split_list(data_D, min(num_substeps, len(data_D)))
```

With `num_substeps=10` and 8 samples:
- `min(10, 8) = 8` chunks
- Each chunk has 1 sample
- `forward_backward()` is called 8 times, each with 1 sample

With `num_substeps=1` and 8 samples:
- `min(1, 8) = 1` chunk
- Single chunk has all 8 samples
- `forward_backward()` is called 1 time with 8 samples

**Miles Expectation:**

Miles's Megatron backend expects a fixed `global_batch_size` set during model creation:

```python
# kgateway/python/ai_extension/training/api.py:419
'--global-batch-size', '8',  # Hardcoded
```

With DP (data parallel) = 2:
```python
# miles/backends/megatron_utils/data.py:182
num_local_gbs = args.global_batch_size // dp_size  # 8 // 2 = 4
```

Each DP rank expects exactly 4 samples per forward_backward call.

**Mismatch:**
- Client sends: 1 sample (with num_substeps=10)
- Miles expects: 4 samples per rank
- Result: Assertion fails

### The Solution

Set `num_substeps=1` to send all 8 samples together:
- Client sends: 8 samples
- Miles receives: 8 samples total, 4 per DP rank
- Result: Batch size matches, training succeeds

### Why Not Override args.global_batch_size?

Modifying `args.global_batch_size` dynamically **does not work** because:

1. **Ray serializes args**: When creating actors, Ray pickles args and sends copies
2. **Megatron global state**: `set_args(args)` stores args in immutable module-level state
3. **Actor isolation**: Each Ray actor uses its own `self.args` set during initialization

Evidence:
```python
# actor_group.py:114
actor.init.remote(args)  # Ray serializes args → actor gets a copy

# initialize.py:53
set_args(args)  # Stored in Megatron global_vars, immutable after init
```

Modifying `client_info["args"]` in kgateway has **zero effect** on Ray actors' state.

### Alternative Solutions (Not Chosen)

1. **Accumulate chunks in kgateway** - Complex state management, breaks request/response contract
2. **Modify Miles to accept dynamic batch sizes** - Requires Miles code changes (against constraints)
3. **Set global_batch_size=1** - Kills performance, defeats distributed training purpose
4. **Fork Tinker client** - Not maintainable, diverges from upstream

## Test Results

**Expected Output:**
```
Creating TrainingClient for model_id='model_...'
Will train on 100 batches
Starting sample
sample took 1.51 seconds

====== Trajectory Group ======
****** trajectory idx=0, reward=1 ******
---- datum ----
<|im_start|>user
What is 4 + 5?<|im_end|>
<|im_start|>assistant
<think>
9<|im_end|>
...

Exit Code: 0 ✅ SUCCESS
```

All trajectories should have `reward=1` (correct answers) and the test should complete successfully.

## Troubleshooting

### Error: Connection refused
```bash
# Check kgateway pod is running
kubectl get pods -n miles-gmi-tinker -l app=kgateway-training

# Check service
kubectl get svc -n miles-gmi-tinker kgateway-training
```

### Error: Model not found in /data/models
```bash
# Check model exists in miles-training-0
kubectl exec -n miles-gmi-tinker miles-training-0 -- ls -la /data/models/
```

### Error: tinker_gmi module not found
```bash
# Reinstall packages
kubectl cp /tmp/tmux-tmp/tinker_gmi miles-gmi-tinker/miles-training-0:/tmp/tinker_gmi
kubectl cp /tmp/tmux-tmp/tinker-cookbook miles-gmi-tinker/miles-training-0:/tmp/tinker-cookbook
kubectl exec -n miles-gmi-tinker miles-training-0 -- bash -c "cd /tmp/tinker_gmi && pip install -e . && cd /tmp/tinker-cookbook && pip install -e ."
```

### Error: Batch size mismatch (after changing num_substeps)
If you get batch size errors with different `num_substeps` values:

**Formula:**
```
total_samples = group_size × groups_per_batch
num_chunks = min(num_substeps, total_samples)
samples_per_chunk = total_samples / num_chunks

For Miles to accept:
samples_per_chunk × dp_size = global_batch_size
```

**Example:**
- global_batch_size = 8 (hardcoded)
- dp_size = 2
- Required: samples_per_chunk × 2 = 8 → samples_per_chunk = 4

Options:
1. num_substeps=1: 8 samples / 1 chunk = 8 samples per chunk ✅ (8 × 2 = 16, but Miles accepts multiples)
2. num_substeps=2: 8 samples / 2 chunks = 4 samples per chunk ✅ (4 × 2 = 8, perfect match)

## Future Work

1. **Make global_batch_size configurable**: Add parameter to `create_model` API
2. **Dynamic batch size support in Miles**: Modify Miles to infer batch size from data
3. **Better error messages**: Add validation in kgateway to detect mismatches early

---

## Tinker-Cookbook Verification Tests (Tests 3-7)

The following scripts verify that tinker-cookbook recipes work with the GMI/Miles environment. These correspond to 5 of the 7 post-training examples from tinker-cookbook/recipes.

### Prerequisites (All Tests)

1. Kubernetes cluster with miles-gmi-tinker namespace
2. miles-training-0 pod running with:
   - Ray cluster initialized (DP=2, PP=2, TP=1)
   - Model: `/data/models/Qwen2.5-0.5B-Instruct`
   - tinker_gmi and tinker-cookbook packages installed
3. kgateway-training deployment running
4. Service: `kgateway-training.miles-gmi-tinker:8000`

### Test 3: Preference Learning - Shorter Responses

**Files:** `test_preference_shorter.sh` + `test_preference_shorter.py`

Tests pairwise preference RL with a simple preference model that always prefers shorter responses.

**Usage:**
```bash
./test_preference_shorter.sh
```

**Expected Results:**
- `ac_tokens_per_turn` should drop significantly after 40 steps
- The policy generates progressively shorter responses
- Training completes without errors

**Type:** RL (requires `debug_train_only=False` for sampling)

**Parameters:**
- `batch_size=8`: Number of prompts per batch
- `group_size=4`: Samples per prompt (4×2=8 total)
- `learning_rate=3e-5`: RL learning rate
- `max_tokens=64`: Maximum generation length

---

### Test 4: Prompt Distillation - Language Classification

**Files:** `test_prompt_distillation.sh` + `minimal_lang_data.jsonl`

Tests prompt distillation (context distillation) to internalize a language classification task. Uses a minimal synthetic dataset with 20 language classification examples.

**Usage:**
```bash
./test_prompt_distillation.sh
```

**Expected Results:**
- Training loss decreases over epochs
- Model learns to classify language without seeing the original prompt
- Training completes on the small dataset (20 examples)

**Type:** SFT (uses `debug_train_only=True`)

**Parameters:**
- `batch_size=4`: Small batch for minimal dataset
- `num_epochs=2`: Train for 2 epochs on the small dataset
- `learning_rate=1e-4`: SFT learning rate
- `file_path=/tmp/minimal_lang_data.jsonl`: Synthetic dataset

**Note:** This is a simplified version that skips the teacher model sampling step. In production, you'd first run `create_data.py` to generate the distillation dataset using a teacher model.

---

### Test 5: Multi-Agent - Guess the Number

**File:** `test_multiplayer_guess_number.sh`

Tests multi-turn RL where the policy learns to guess a target number (1-100) with feedback on whether the guess is too high or low.

**Usage:**
```bash
./test_multiplayer_guess_number.sh
```

**Expected Results:**
- Policy learns to efficiently narrow down the target number
- Number of turns decreases over training
- Successful guessing within fewer attempts

**Type:** RL (requires `debug_train_only=False` for sampling)

**Parameters:**
- `group_size=4`: Samples per prompt
- `batch_size=8`: Number of prompts per batch
- `learning_rate=3e-5`: RL learning rate
- `max_tokens=64`: Maximum generation length per turn

---

### Test 6: Model Distillation - Off-policy SFT

**File:** `test_distillation_off_policy.sh`

Tests supervised fine-tuning on the OpenThoughts3 dataset, which contains reasoning traces with chain-of-thought style responses.

**Usage:**
```bash
./test_distillation_off_policy.sh
```

**Expected Results:**
- Training loss (NLL) decreases over steps
- Model learns reasoning patterns from the dataset
- Streaming dataset loads and trains correctly

**Type:** SFT (uses `debug_train_only=True`)

**Parameters:**
- `batch_size=8`: Small batch for testing
- `max_prompts=100`: Limited to 100 prompts for quick testing (default: 384,000)
- `learning_rate=1e-4`: SFT learning rate
- `max_length=8192`: Context window
- `buffer_size=1024`: Shuffle buffer for streaming dataset

---

### Test 7: Direct Preference Optimization (DPO)

**File:** `test_dpo.sh`

Tests DPO training on the HHH (Helpful, Harmless, Honest) preference dataset. DPO uses a custom loss function to learn from preference pairs.

**Usage:**
```bash
./test_dpo.sh
```

**Expected Results:**
- DPO loss decreases over steps
- Model learns from preference pairs
- Reference model comparison works correctly

**Type:** SFT-like (uses custom DPO loss, `debug_train_only=True`)

**Parameters:**
- `dataset=hhh`: HHH preference dataset
- `batch_size=8`: Small batch for testing
- `learning_rate=1e-5`: DPO learning rate (lower than SFT)
- `dpo_beta=0.1`: DPO temperature parameter
- `max_length=8192`: Context window

---

## Running All Tests

To run tests 3-7 sequentially:

```bash
cd /tmp/tmux-tmp/tinker-cookbook/gmi_scripts

# Cleanup command
CLEANUP="kubectl exec -n miles-gmi-tinker miles-training-0 -- python -u /tmp/tinker_gmi/tests_integration/cleanup_test_env.py"

# Test 3: Preference/Shorter (RL)
echo "=== Test 3: Preference Learning - Shorter ===" && \
$CLEANUP && ./test_preference_shorter.sh

# Test 4: Prompt Distillation (SFT)
echo "=== Test 4: Prompt Distillation ===" && \
$CLEANUP && ./test_prompt_distillation.sh

# Test 5: Guess Number (RL)
echo "=== Test 5: Multi-Agent - Guess Number ===" && \
$CLEANUP && ./test_multiplayer_guess_number.sh

# Test 6: Distillation (SFT)
echo "=== Test 6: Model Distillation - Off-policy ===" && \
$CLEANUP && ./test_distillation_off_policy.sh

# Test 7: DPO (SFT-like)
echo "=== Test 7: DPO ===" && \
$CLEANUP && ./test_dpo.sh
```

**Important:** Always run cleanup between tests to:
- Delete model and free GPU resources
- Clean up old futures from database
- Reset training state

---

## Summary of All Tests

| Test # | Recipe | Type | Script | Status |
|--------|--------|------|--------|--------|
| 1 | Chat SL (No Robots) | SFT | (skipped) | ⏭️ User skipped |
| 2 | Math RL (Arithmetic) | RL | `test_math_rl_arithmetic.sh` | ✅ Working |
| 3 | Preference/Shorter | RL | `test_preference_shorter.sh` | ✅ Created |
| 4 | Prompt Distillation | SFT | `test_prompt_distillation.sh` | ✅ Created |
| 5 | Multi-Agent (Guess Number) | RL | `test_multiplayer_guess_number.sh` | ✅ Created |
| 6 | Model Distillation (Off-policy) | SFT | `test_distillation_off_policy.sh` | ✅ Created |
| 7 | DPO (HHH) | SFT-like | `test_dpo.sh` | ✅ Created |

**Test Coverage:** 6 out of 7 post-training recipes covered (86%)

---

## Common Issues

### Issue: "Model not found"
Check model exists in pod:
```bash
kubectl exec -n miles-gmi-tinker miles-training-0 -- ls -la /data/models/
```

### Issue: "Connection refused to kgateway"
Check kgateway pod is running:
```bash
kubectl get pods -n miles-gmi-tinker -l app=kgateway-training
kubectl logs -n miles-gmi-tinker deployment/kgateway-training --tail=50
```

### Issue: "tinker_gmi module not found"
Reinstall packages:
```bash
kubectl cp /tmp/tmux-tmp/tinker_gmi miles-gmi-tinker/miles-training-0:/tmp/tinker_gmi
kubectl cp /tmp/tmux-tmp/tinker-cookbook miles-gmi-tinker/miles-training-0:/tmp/tinker-cookbook
kubectl exec -n miles-gmi-tinker miles-training-0 -- bash -c "cd /tmp/tinker_gmi && pip install -e . && cd /tmp/tinker-cookbook && pip install -e ."
```

### Issue: Ray actors not spawning (RL tests)
Check Ray cluster status:
```bash
kubectl exec -n miles-gmi-tinker miles-training-0 -- ray list actors --filter "state=ALIVE"
```

If actors are not running, restart miles pod, then restart kgateway pod.
