import subprocess
import sys

def run_sequential_experiments():
    seeds = [1, 2, 3, 4, 5]
    
    # Base parameters
    env = "matrixgames:climbing-nostate-v0"
    steps = 500000
    
    # Common hyperparameters
    base_hyperparams = [
        "hidden_dim=64",
        "initial_entropy_coef=4",
        "final_entropy_coef=0.1",
        "entropy_end_ratio=0.8",
        "lr=0.0003",
        "use_rnn=False"
    ]

    experiments = [
        {
            "name_prefix": "Pareto-AC-Adaptive-Linear-Seed",
            "group": "Pareto-AC-Adaptive-Linear-5Seeds",
            "algo": "adaptive_pac",
            "extra_args": ["optimism_schedule=linear"]
        },
        {
            "name_prefix": "Pareto-AC-Adaptive-Exp-Seed",
            "group": "Pareto-AC-Adaptive-Exp-5Seeds",
            "algo": "adaptive_pac",
            "extra_args": ["optimism_schedule=exp"]
        },
        {
            "name_prefix": "Pareto-AC-CVaR-0.5-Seed",
            "group": "Pareto-AC-CVaR-0.5-5Seeds",
            "algo": "cvar_pac",
            "extra_args": ["cvar_alpha=0.5"]
        },
        {
            "name_prefix": "Pareto-AC-CVaR-0.25-Seed",
            "group": "Pareto-AC-CVaR-0.25-5Seeds",
            "algo": "cvar_pac",
            "extra_args": ["cvar_alpha=0.25"]
        },
        {
            "name_prefix": "Pareto-AC-CVaR-0.05-Seed",
            "group": "Pareto-AC-CVaR-0.05-5Seeds",
            "algo": "cvar_pac",
            "extra_args": ["cvar_alpha=0.05"]
        }
    ]

    for exp in experiments:
        for seed in seeds:
            run_name = f"{exp['name_prefix']}{seed}"
            print(f"----------------------------------------------------------------")
            print(f"Starting Run: {run_name} (Seed {seed})")
            print(f"----------------------------------------------------------------")
            
            # Construct command
            command = [
                sys.executable, "run_experiments_main.py",
                "--algo", exp["algo"],
                "--env", env,
                "--steps", str(steps),
                "--group", exp["group"],
                "--name", run_name,
                f"seed={seed}"
            ]
            
            # Add base hyperparams
            command.extend(base_hyperparams)
            
            # Add experiment specific args
            command.extend(exp["extra_args"])
            
            cmd_str = " ".join(command)
            print(f"Executing: {cmd_str}")
            
            try:
                subprocess.run(command, check=True)
                print(f"Run {run_name} completed successfully.")
                
            except subprocess.CalledProcessError as e:
                print(f"Run {run_name} FAILED with return code {e.returncode}")
                print("Stopping sequential execution due to error.")
                return # Stop on first failure to save time/compute if config is wrong
            except KeyboardInterrupt:
                print("Sequential execution interrupted by user.")
                sys.exit(1)
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                return

if __name__ == "__main__":
    run_sequential_experiments()
