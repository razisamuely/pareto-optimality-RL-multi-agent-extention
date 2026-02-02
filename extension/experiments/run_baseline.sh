#!/bin/bash
# Run Baseline PAC experiments

# 1. Penalty Game (k=-100)
# 'gymma' wraps the gym environment. We pass the gymEnv=ID to it.
# We assume 'pac' is the correct config name for Vanilla PAC based on standard EPyMARL conventions.
# If 'pac' isn't found, we'll need to check src/config/algs

echo "Running Baseline: Penalty-100"
python3 src/main.py --config=pac_dcg_ns --env-config=gymma with env_args.key="penalty-100-nostate-v0" t_max=100000 save_model=False use_cuda=False

echo "Running Baseline: Climbing"
python3 src/main.py --config=pac_dcg_ns --env-config=gymma with env_args.key="climbing-nostate-v0" t_max=100000 save_model=False use_cuda=False
