#!/bin/bash
# Test 3: Preference Learning - Shorter Responses
#
# This tests the PairwisePreferenceRLDatasetBuilder with a simple preference model
# that always prefers shorter responses. This is an RL task.
#
# Expected: ac_tokens_per_turn should drop significantly after 40 steps
#
# Prerequisites:
# - slime-training-0 pod running with Ray cluster (DP=2, PP=2)
# - kgateway-training deployment up and running
# - tinker_gmi and tinker-cookbook installed in slime-training-0 pod

# Copy the test script to the pod
kubectl cp /tmp/tmux-tmp/tinker-cookbook/gmi_scripts/test_preference_shorter.py slime-gmi/slime-training-0:/tmp/test_preference_shorter.py

# Run the test
kubectl exec -n slime-gmi slime-training-0 -- bash -c 'export TINKER_API_KEY=slime-dev-key && python -u /tmp/test_preference_shorter.py 2>&1'
