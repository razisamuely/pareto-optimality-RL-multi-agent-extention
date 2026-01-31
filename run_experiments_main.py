import argparse
import subprocess
import sys
import os

def run_experiment(env_key, algorithm="qmix", t_max=10_000, use_wandb=True, wandb_project="epymarl-demo", wandb_team="eliads-ben-gurion-university-of-the-negev", wandb_mode="online", run_name=None, extra_args=None):
    """
    Runs an experiment using src/main.py with the specified parameters.
    """
    
    # Base command
    # We use sys.executable to ensure we use the same python interpreter
    command = [sys.executable, "src/main.py"]
    
    # Configuration arguments
    command.append(f"--config={algorithm}")
    command.append("--env-config=gymma")
    
    # Variable arguments using 'with'
    # Sacred config updates require correct formatting
    config_updates = []
    
    # Environment
    config_updates.append(f"env_args.key={env_key}")
    
    # Time limit (defaults to 2.05M for full run)
    config_updates.append(f"t_max={t_max}")
    
    # WandB
    if use_wandb:
        config_updates.append("use_wandb=True")
        config_updates.append(f"wandb_project={wandb_project}")
        config_updates.append(f"wandb_team={wandb_team}")
        config_updates.append(f"wandb_mode={wandb_mode}")
        if run_name:
            config_updates.append(f"wandb_run_name={run_name}")
    else:
        config_updates.append("use_wandb=False")

    if extra_args:
        config_updates.extend(extra_args)

    if config_updates:
        command.append("with")
        command.extend(config_updates)
        
    print(f"Running command: {' '.join(command)}")
    
    try:
        # Run the command
        process = subprocess.Popen(command, env=os.environ.copy())
        process.wait()
        
        if process.returncode == 0:
            print("Experiment completed successfully.")
        else:
            print(f"Experiment failed with return code {process.returncode}.")
            
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")
        process.terminate()
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EPyMARL experiments modularly.")
    
    # Defaults for "climbing game"
    parser.add_argument("--env", type=str, default="matrixgames:climbing-nostate-v0", help="Gymnasium environment key")
    parser.add_argument("--algo", type=str, default="qmix", help="Algorithm configuration name (e.g., qmix, vdn)")
    parser.add_argument("--steps", type=int, default=10_000, help="Total timesteps (t_max)")
    parser.add_argument("--wandb", action="store_true", default=True, help="Enable WandB logging")
    parser.add_argument("--no-wandb", action="store_false", dest="wandb", help="Disable WandB logging")
    parser.add_argument("--project", type=str, default="epymarl-demo", help="WandB project name")
    parser.add_argument("--team", type=str, default="eliads-ben-gurion-university-of-the-negev", help="WandB team/entity name")
    parser.add_argument("--offline", action="store_true", help="Run WandB in offline mode")
    parser.add_argument("--name", type=str, default=None, help="WandB run name")

    args, unknown = parser.parse_known_args()
    
    mode = "offline" if args.offline else "online"
    
    run_experiment(
        env_key=args.env,
        algorithm=args.algo,
        t_max=args.steps,
        use_wandb=args.wandb,
        wandb_project=args.project,
        wandb_team=args.team,
        wandb_mode=mode,
        run_name=args.name,
        extra_args=unknown
    )
