# PAC Extension Experiment Notes

## Summary of Work
Recent updates focused on fixing and extending the Pareto Actor-Critic (PAC) learners for multi-agent reinforcement learning.

### Key Changes
1.  **Bug Fixes**:
    - Resolved `TypeError: 'DCGCriticNS' object is not callable` in `PACAdaptiveLearner` and `PACCvarLearner`. The critics are now correctly called via their `.forward()` method.
    - Fixed configuration key conflicts in `pac_sarsa`.
2.  **New Implementations**:
    - **Adaptive Optimism (`PACAdaptiveLearner`)**: Dynamically adjusts optimism validation scaling based on a schedule (linear or exponential).
    - **CVaR Optimism (`PACCvarLearner`)**: Uses Conditional Value at Risk to handle uncertainty in Q-value estimation, controlled by `alpha`.
3.  **Scripts**:
    - Added automated scripts to run experiment variants and baselines.
    - Added a reporting script to parse Sacred results and generate a markdown table.

---

## Running Experiments

### 1. Run Variants
To run the full suite of experimental variants (Adaptive and CVaR):

```bash
# Make sure to run from the root directory
bash extension/experiments/run_variants.sh
```

**What this runs:**
- **Environments**: `penalty-100-nostate-v0`, `climbing-nostate-v0`
- **Adaptive Variants**:
    - Linear decay (50k steps)
    - Exponential decay
- **CVaR Variants**:
    - Alpha = 0.1
    - Alpha = 0.25
    - Alpha = 0.5

### 2. Run Baseline
To run the standard PAC baseline:

```bash
bash extension/experiments/run_baseline.sh
```

---

## Configuration & Customization

### Configuration Files
Configs are located in `extension/configs/`.

| Config File | Algo Name | Description |
| :--- | :--- | :--- |
| `adaptive_pac.yaml` | `pac_adaptive_learner` | Adaptive optimism scaling. |
| `cvar_pac.yaml` | `pac_cvar_learner` | CVaR-based optimism. |

### How to Change Configuration
You can override configuration parameters via the command line using Sacred's `with` syntax or by editing the yaml files directly.

**Example: Changing CVaR Alpha**
```bash
python3 extension/custom_main.py \
    --config=cvar_pac \
    --env-config=gymma \
    with \
    env_args.key="penalty-100-nostate-v0" \
    cvar_alpha=0.75 \
    name="my_custom_run"
```

**Key Parameters:**

- **Adaptive PAC**:
    - `optimism_schedule`: `linear` or `exp`
    - `optimism_decay_steps`: Number of steps to decay the optimism factor (e.g. `50000`).
    
- **CVaR PAC**:
    - `cvar_alpha`: Quantile level for CVaR (0.0 < alpha <= 1.0). Lower values are more risk-averse (focusing on the tail).

---

## File Locations

| Component | File Path |
| :--- | :--- |
| **Main Entry Point** | `extension/custom_main.py` |
| **Adaptive Learner** | `extension/learners/pac_adaptive_learner.py` |
| **CVaR Learner** | `extension/learners/pac_cvar_learner.py` |
| **Optimism Module** | `extension/modules/optimism.py` |
| **Report Generator** | `extension/analysis/generate_report.py` |

---

## Analysis

Results are stored in `results/sacred/` by default.

To generate a summary report of the latest runs:

```bash
python3 extension/analysis/generate_report.py
```

This will output a summary table to `extension/docs/report/report.md`.
