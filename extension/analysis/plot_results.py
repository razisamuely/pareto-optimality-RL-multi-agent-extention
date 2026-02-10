
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
    data = defaultdict(lambda: defaultdict(list))
    
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
                    if "dcg" in alg_name:
                        variant = "Baseline (DCG)"
                    else:
                        variant = "Baseline (PAC)"
                
                if env_name in ["climbing-nostate-v0", "penalty-100-nostate-v0"]:
                     # Filter: only keep runs that reached 500k
                     if steps[-1] >= 450000:
                         data[env_name][variant].append((steps, returns))
                
            except Exception as e:
                # print(f"Skipping {root}: {e}")
                continue
    return data

def plot_results(data):
    # data structure changed to data[env][variant] = list of runs
    
    # Define colors/styles
    colors = {
        "Adaptive-linear": "green",
        "Adaptive-exp": "blue",
        "CVaR-0.1": "red",
        "CVaR-0.25": "orange",
        "CVaR-0.5": "purple",
        "Baseline (PAC)": "black",
        "Baseline (DCG)": "gray"
    }

    # Common x-axis for interpolation
    target_steps = np.linspace(0, 500000, 500)

    for env_name, variants_data in data.items():
        plt.figure(figsize=(10, 6))
        
        for variant, runs in variants_data.items():
            if not runs:
                continue
                
            interp_returns = []
            
            for steps, returns in runs:
                val_interp = np.interp(target_steps, steps, returns, left=np.nan, right=np.nan)
                max_t = steps[-1]
                mask = target_steps > max_t
                val_interp[mask] = np.nan
                interp_returns.append(val_interp)
                
            interp_returns = np.array(interp_returns)
            
            # Use nanmean and nanstd
            mean = np.nanmean(interp_returns, axis=0)
            std = np.nanstd(interp_returns, axis=0)
            
            color = colors.get(variant, None)
            valid_mask = ~np.isnan(mean)
            
            if np.any(valid_mask):
                 plt.plot(target_steps[valid_mask], mean[valid_mask], label=variant, color=color, linewidth=2)
                 plt.fill_between(target_steps[valid_mask], (mean - std)[valid_mask], (mean + std)[valid_mask], color=color, alpha=0.15)

        plt.title(f"{env_name}: Algorithm Comparison (500k Steps)", fontsize=14)
        plt.xlabel("Timesteps", fontsize=12)
        plt.ylabel("Test Return", fontsize=12)
        plt.legend(loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_file = os.path.join(OUTPUT_DIR, f"{env_name}_results.png")
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    print("Loading results...")
    data = load_results()
    print(f"Found data for {len(data)} environments.")
    plot_results(data)
