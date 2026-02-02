#!/bin/bash
# Run PAC Extension Variants

# Environments
ENVS=("penalty-100-nostate-v0" "climbing-nostate-v0")

# 1. Adaptive Optimism
# Experiment A1: Linear 50k
echo "Running Adaptive Linear 50k"
for env in "${ENVS[@]}"; do
    python3 extension/custom_main.py --config=adaptive_pac --env-config=gymma with env_args.key=$env t_max=100000 optimism_schedule="linear" optimism_decay_steps=50000 name="pac_adaptive_linear_50k" save_model=False use_cuda=False
done

# Experiment A2: Exponential
echo "Running Adaptive Exponential"
for env in "${ENVS[@]}"; do
    python3 extension/custom_main.py --config=adaptive_pac --env-config=gymma with env_args.key=$env t_max=100000 optimism_schedule="exp" optimism_decay_steps=100000 name="pac_adaptive_exp" save_model=False use_cuda=False
done

# 2. CVaR Optimism
# Experiment C1: Alpha 0.1
echo "Running CVaR Alpha=0.1"
for env in "${ENVS[@]}"; do
    python3 extension/custom_main.py --config=cvar_pac --env-config=gymma with env_args.key=$env t_max=100000 cvar_alpha=0.1 name="pac_cvar_0.1" save_model=False use_cuda=False
done

# Experiment C2: Alpha 0.25
echo "Running CVaR Alpha=0.25"
for env in "${ENVS[@]}"; do
    python3 extension/custom_main.py --config=cvar_pac --env-config=gymma with env_args.key=$env t_max=100000 cvar_alpha=0.25 name="pac_cvar_0.25" save_model=False use_cuda=False
done

# Experiment C3: Alpha 0.50
echo "Running CVaR Alpha=0.50"
for env in "${ENVS[@]}"; do
    python3 extension/custom_main.py --config=cvar_pac --env-config=gymma with env_args.key=$env t_max=100000 cvar_alpha=0.50 name="pac_cvar_0.50" save_model=False use_cuda=False
done
