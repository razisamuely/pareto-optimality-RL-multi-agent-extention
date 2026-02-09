import os
import json
import pandas as pd
import numpy as np

def generate_table(results_dir="results/sacred"):
    data = []
    
    if not os.path.exists(results_dir):
        print(f"Directory {results_dir} not found.")
        return

    # Iterate over all run directories
    for run_id in os.listdir(results_dir):
        run_path = os.path.join(results_dir, run_id)
        if not os.path.isdir(run_path):
            continue
            
        config_path = os.path.join(run_path, "config.json")
        info_path = os.path.join(run_path, "info.json")
        
        if not os.path.exists(config_path) or not os.path.exists(info_path):
            continue
            
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            with open(info_path, 'r') as f:
                info = json.load(f)
                
            # Extract relevant info
            algo = config.get("name", "unknown")
            group = config.get("wandb_run_group", "unknown")
            seed = config.get("seed", -1)
            
            # Determine variant based on group or algo/args
            variant = "Unknown"
            if "Vanilla" in group:
                variant = "Vanilla"
            elif "Adaptive" in group:
                if "Linear" in group:
                    variant = "Adaptive (Linear)"
                elif "Exp" in group:
                    variant = "Adaptive (Exp)"
            elif "CVaR" in group:
                alpha = config.get("cvar_alpha", "?")
                variant = f"CVaR (alpha={alpha})"
            
            # Get final reward (test_return_mean is usually better for Eval)
            # If test_return_mean exists, take the mean of the last few evaluations or the absolute last?
            # Usually final performance is the average of the last X evaluations or just the last one.
            # Let's take the mean of the last 5 evaluations to be robust, or just the max, or the last.
            # Let's take the LAST logged test_return_mean.
            
            metric_key = "test_return_mean"
            if metric_key not in info:
                # Fallback to return_mean (training reward)
                metric_key = "return_mean"
                
            if metric_key in info:
                # Check if it's a list and not empty
                values = info[metric_key]
                if isinstance(values, list) and len(values) > 0:
                    # Take average of last 3 data points to smooth slightly? 
                    # Or just the last one. Let's take last one.
                    final_score = values[-1]
                    if isinstance(final_score, dict): # metrics sometimes stored as {steps: val} or similar? 
                        # Sacred usually stores as list of values.
                         pass 
                    
                    # Store data
                    data.append({
                        "Algorithm": "Pareto-AC", # Base algo name
                        "Variant": variant,
                        "Group": group,
                        "Seed": seed,
                        "Score": final_score
                    })
        except Exception as e:
            # print(f"Error processing {run_id}: {e}")
            continue

    if not data:
        print("No valid data found.")
        return

    df = pd.DataFrame(data)
    
    # Group by Variant and calculate stats
    summary = df.groupby(["Variant"]).agg(
        Mean_Score=("Score", "mean"),
        Std_Dev=("Score", "std"),
        Count=("Score", "count")
    ).reset_index()
    
    # Sort for better readability
    summary = summary.sort_values(by="Mean_Score", ascending=False)
    
    print("\n" + "="*50)
    print("FINAL RESULTS TABLE")
    print("="*50)
    print(summary.to_markdown(index=False, floatfmt=".2f"))
    print("\n")
    
    # Also save to file
    summary.to_csv("final_results_summary.csv", index=False)
    print("Summary saved to 'final_results_summary.csv'")

if __name__ == "__main__":
    generate_table()
