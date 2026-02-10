#!/bin/bash

# Run Baseline PAC (regular, non-DCG) for 500k steps with 5 seeds on BOTH games
# This uses pac_ns config which has critic_type: "pac_critic_ns" — same as Adaptive/CVaR extensions

run_batch() {
    env=$1
    echo "--- Running PAC NS Baseline on $env ---"
    
    for seed in {1..5}
    do
        echo "Starting Seed $seed for pac_ns on $env"
        python3 extension/custom_main.py --config=pac_ns --env-config=gymma with env_args.key=$env t_max=500000 name=baseline_pac_ns seed=$seed save_model=False use_cuda=False > /dev/null 2>&1 &
    done
    
    wait
    echo "Completed PAC NS Baseline on $env"
}

run_batch "climbing-nostate-v0"
run_batch "penalty-100-nostate-v0"

echo "All PAC NS Baseline runs completed!"
