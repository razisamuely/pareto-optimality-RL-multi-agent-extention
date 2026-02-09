import subprocess
import sys
import time

def run_sequential_experiments():
    seeds = [1, 2, 3] # 3 seeds per operator
    
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
        "use_rnn=False",
        "use_cuda=False"
    ]

    experiments = [
        # Vanilla
        {
            "name_prefix": "Pareto-AC-Vanilla-Seed",
            "group": "Pareto-AC-Vanilla-5Seeds", # Keeping same group name for consistency or "3Seeds" if preferred
            "algo": "pac_ns",
            "extra_args": []
        },
        # Adaptive
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
        # CVaR
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

    total_runs = len(experiments) * len(seeds)
    current_run = 0

    print(f"Plan: Running {total_runs} experiments sequentially.")

    for exp in experiments:
        for seed in seeds:
            current_run += 1
            run_name = f"{exp['name_prefix']}{seed}"
            print(f"\n[{current_run}/{total_runs}] Starting Run: {run_name} (Seed {seed})")
            print(f"Algorithm: {exp['algo']}")
            
            # Construct command
            python_executable = "C:/Users/eliad/Desktop/phd/course_project/pareto-optimality-RL-multi-agent-extention/.venv/Scripts/python.exe"
            command = [
                python_executable, "run_experiments_main.py",
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
            # print(f"Executing: {cmd_str}")
            
            try:
                start_time = time.time()
                subprocess.run(command, check=True)
                duration = time.time() - start_time
                print(f"Run {run_name} completed successfully in {duration:.2f}s.")
                
            except subprocess.CalledProcessError as e:
                print(f"Run {run_name} FAILED with return code {e.returncode}")
                # We optionally continue here to try other experiments
            except KeyboardInterrupt:
                print("Sequential execution interrupted by user.")
                sys.exit(1)
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    run_sequential_experiments()
