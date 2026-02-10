#!/bin/bash
# Run Climbing Game Study (20 runs total)

ENV="climbing-nostate-v0"
SEEDS=(1 2 3 4 5)
STEPS=500000

echo "Starting Climbing Study on $ENV for $STEPS steps..."

# Function to run a batch of seeds in parallel
run_batch() {
    local variant_name=$1
    local config=$2
    shift 2
    local extra_args=("$@")

    echo "--- Running $variant_name ---"
    for seed in "${SEEDS[@]}"; do
        echo "Starting Seed $seed for $variant_name"
        python3 extension/custom_main.py \
            --config=$config \
            --env-config=gymma \
            with \
            env_args.key=$ENV \
            t_max=$STEPS \
            name="${variant_name}" \
            seed=$seed \
            save_model=False \
            use_cuda=False \
            "${extra_args[@]}" > /dev/null 2>&1 &
    done
    wait
    echo "Finished $variant_name"
}

# 1. CVaR Alpha=0.25
run_batch "climbing_cvar_0.25" "cvar_pac" cvar_alpha=0.25

# 2. CVaR Alpha=0.1
run_batch "climbing_cvar_0.1" "cvar_pac" cvar_alpha=0.1

# 3. Adaptive Exponential
run_batch "climbing_adaptive_exp" "adaptive_pac" optimism_schedule="exp" optimism_decay_steps=100000

# 4. Adaptive Linear
run_batch "climbing_adaptive_linear" "adaptive_pac" optimism_schedule="linear" optimism_decay_steps=50000

echo "Climbing Study Complete."

