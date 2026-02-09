import subprocess
import sys

def run_sequential_experiments():
    seeds = [1, 2, 3, 4, 5]
    group_name = "Pareto-AC-Vanilla-5Seeds"
    
    # Base parameters
    algo = "pac_ns"
    env = "matrixgames:climbing-nostate-v0"
    steps = 500000
    
    # Specific hyperparameters for Climbing Game
    hyperparams = [
        "hidden_dim=64",
        "initial_entropy_coef=4",
        "final_entropy_coef=0.1",
        "entropy_end_ratio=0.8",
        "lr=0.0003",
        "use_rnn=False"
    ]

    for seed in seeds:
        run_name = f"Pareto-AC-Vanilla-Seed{seed}"
        print(f"----------------------------------------------------------------")
        print(f"Starting Run: {run_name} (Seed {seed})")
        print(f"----------------------------------------------------------------")
        
        # Construct command
        # We use the current python executable to call run_experiments_main.py
        command = [
            sys.executable, "run_experiments_main.py",
            "--algo", algo,
            "--env", env,
            "--steps", str(steps),
            "--group", group_name,
            "--name", run_name,
            f"seed={seed}"
        ]
        
        # Add hyperparams
        command.extend(hyperparams)
        
        cmd_str = " ".join(command)
        print(f"Executing: {cmd_str}")
        
        try:
            # shell=True might be needed on Windows if sys.executable path has spaces without quotes, 
            # but subprocess handles list arguments safely usually. 
            # However, run_experiments_main.py expects arguments.
            # Let's use list format which is safer.
            result = subprocess.run(command, check=True)
            print(f"Run {run_name} completed successfully.")
            
        except subprocess.CalledProcessError as e:
            print(f"Run {run_name} FAILED with return code {e.returncode}")
            # Decide whether to continue or stop. 
            # Usually better to stop to investigate, but user wants all runs. 
            # I'll stop to avoid wasting time if configuration is wrong.
            print("Stopping sequential execution due to error.")
            break
        except KeyboardInterrupt:
            print("Sequential execution interrupted by user.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break

if __name__ == "__main__":
    run_sequential_experiments()
