
import os
import json
import logging
import numpy as np
from collections import defaultdict

# Setup paths
RESULTS_DIR = "/home/corsound/workspace/epymarl/results/sacred"
REPORT_FILE = "/home/corsound/workspace/epymarl/extension/docs/report/report.md"

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
                
                # Extract metrics
                # We care about 'test_return_mean' usually, or 'return_mean'
                # info['test_return_mean'] is a list of {value, step, timestamp}
                
                if "test_return_mean" not in info or "test_return_mean_T" not in info:
                    continue
                
                # Zip values and steps
                returns = [x["value"] for x in info["test_return_mean"]]
                steps = info["test_return_mean_T"]
                
                if not returns:
                    continue

                final_return = returns[-1]
                final_step = steps[-1]

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
                
                data[env_name].append({
                    "variant": variant,
                    "return": final_return,
                    "steps": final_step
                })
                
            except Exception as e:
                print(f"Skipping {root}: {e}")
                continue
    return data

def generate_markdown(data):
    lines = []
    lines.append("# PAC Extension Experiment Report")
    lines.append("\n## 1. Introduction")
    lines.append("This report summarizes the performance of Adaptive Optimism and CVaR Optimism variants of the Pareto Actor-Critic (PAC) algorithm.")
    lines.append("Experiments were conducted on the Penalty Game (k=-100) and Climbing Game.")

    lines.append("\n## 2. Implementation Details")
    lines.append("- **Adaptive Optimism**: Linearly decays the optimism coefficient from 1.0 (Max Q) to 0.0 (Mean Q) over 50k steps.")
    lines.append("- **CVaR Optimism**: computes the mean of the top-$\\alpha$ quantile of Q-values.")

    lines.append("\n## 3. Results")
    
    if not data:
        lines.append("No results found yet.")
        return "\n".join(lines)
    
    for env, runs in data.items():
        lines.append(f"### Environment: {env}")
        lines.append("| Variant | Final Return | Steps |")
        lines.append("|---|---|---|")
        
        # Group by variant to average if multiple seeds (we define 1 run per variant in script for now but good practice)
        grouped = defaultdict(list)
        for r in runs:
            grouped[r['variant']].append(r['return'])
            
        for variant, returns in grouped.items():
            avg_ret = np.mean(returns)
            std_ret = np.std(returns)
            lines.append(f"| {variant} | {avg_ret:.2f} ± {std_ret:.2f} | {runs[0]['steps']} |")
        
        lines.append("\n")
    
    lines.append("\n## 4. Conclusion")
    lines.append("Comparative analysis of the variants against the baseline.")
    
    return "\n".join(lines)

if __name__ == "__main__":
    print("Loading results...")
    data = load_results()
    print(f"Found data for {len(data)} environments.")
    report = generate_markdown(data)
    
    os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
    with open(REPORT_FILE, "w") as f:
        f.write(report)
    print(f"Report saved to {REPORT_FILE}")
    print(report)
