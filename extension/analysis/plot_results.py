
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Setup paths
RESULTS_DIR = "/home/corsound/workspace/epymarl/results/sacred"
OUTPUT_DIR = "/home/corsound/workspace/epymarl/extension/docs/report"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "climbing_results.png")

def load_results():
    data = defaultdict(list)
    
    # Walk through results directory
    for root, dirs, files in os.walk(RESULTS_DIR):
        if "info.json" in files:
            try:
                with open(os.path.join(root, "info.json"), 'r') as f:
                    info = json.load(f)
                
                with open(os.path.join(root, "config.json"), 'r') as f:
                    config = json.load(f)
                
                if "test_return_mean" not in info or "test_return_mean_T" not in info:
                    continue
                
                returns = [x["value"] for x in info["test_return_mean"]]
                steps = info["test_return_mean_T"]
                
                if not returns:
                    continue

                env_name = config["env_args"]["key"]
                alg_name = config["name"]
                
                # Identify variant
                variant = "baseline"
                if "adaptive" in alg_name:
                    variant = f"Adaptive-{config.get('optimism_schedule', 'unknown')}"
                elif "cvar" in alg_name:
                    variant = f"CVaR-{config.get('cvar_alpha', 'unknown')}"
                elif "pac" in alg_name:
                    variant = "Baseline (PAC)"
                
                if env_name == "climbing-nostate-v0":
                     # Filter: only keep runs that reached close to 500k steps
                     if steps[-1] >= 450000:
                         data[variant].append((steps, returns))
                
            except Exception as e:
                # print(f"Skipping {root}: {e}")
                continue
    return data

def plot_results(data):
    plt.figure(figsize=(10, 6))
    
    # Define colors/styles
    colors = {
        "Adaptive-linear": "green",
        "Adaptive-exp": "blue",
        "CVaR-0.1": "red",
        "CVaR-0.25": "orange",
        "CVaR-0.5": "purple",
        "Baseline (PAC)": "gray"
    }

    # Common x-axis for interpolation
    target_steps = np.linspace(0, 500000, 500)

    for variant, runs in data.items():
        if not runs:
            continue
            
        interp_returns = []
        
        for steps, returns in runs:
            # Interpolate to target_steps
            # We use np.interp. It requires steps to be increasing.
            val_interp = np.interp(target_steps, steps, returns)
            interp_returns.append(val_interp)
            
        interp_returns = np.array(interp_returns)
        
        mean = np.mean(interp_returns, axis=0)
        std = np.std(interp_returns, axis=0)
        
        color = colors.get(variant, None)
        
        plt.plot(target_steps, mean, label=variant, color=color, linewidth=2)
        plt.fill_between(target_steps, mean - std, mean + std, color=color, alpha=0.15)

    plt.title("Climbing Game: Algorithm Comparison (5 Seeds)", fontsize=14)
    plt.xlabel("Timesteps", fontsize=12)
    plt.ylabel("Test Return", fontsize=12)
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(OUTPUT_FILE)
    print(f"Plot saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    print("Loading results...")
    data = load_results()
    print(f"Found data for {len(data)} variants:")
    for v, runs in data.items():
        print(f"  - {v}: {len(runs)} runs")
        
    plot_results(data)
