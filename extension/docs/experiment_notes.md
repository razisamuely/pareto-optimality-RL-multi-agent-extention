# PAC Extension Experiment Notes

## 1. Overview
This document serves as a comprehensive guide to the **Pareto Optimality RL Multi-Agent Extension**. It covers the recent implementations of Adaptive Optimism and CVaR (Conditional Value at Risk) Optimism for the PAC (Pareto Actor-Critic) algorithm.

---

## 2. Codebase Structure & Logic

### Learner Hierarchy
The extension builds upon `PACActorCriticLearner`. The core idea is to modify how the "optimistic" Q-value is calculated for the Advantage function, while generally keeping the Target Q-value based on the standard Max operator for stability.

- **Base Class**: `learners.actor_critic_pac_learner.PACActorCriticLearner`
- **Adaptive Variant**: `extension.learners.pac_adaptive_learner.PACAdaptiveLearner`
- **CVaR Variant**: `extension.learners.pac_cvar_learner.PACCvarLearner`

### Implementation Details

#### A. Adaptive Optimism (`PACAdaptiveLearner`) [Logic](file:///home/corsound/workspace/epymarl/extension/learners/pac_adaptive_learner.py)
**Goal**: Transition from optimistic exploration (Max Q) to rational exploitation (Mean Q) over time.

- **Mechanism**:
  $$Q_{adaptive} = \alpha \cdot Q_{max} + (1 - \alpha) \cdot Q_{mean}$$
  Where $\alpha$ decays from 1.0 to 0.0.
  
- **Optimism Scheduler** (`extension.modules.optimism.OptimismScheduler`):
  - **Linear**: Simple linear interpolation over `optimism_decay_steps`.
  - **Exp**: Exponential decay targeting approx 1% of range at end steps.

#### B. CVaR Optimism (`PACCvarLearner`) [Logic](file:///home/corsound/workspace/epymarl/extension/learners/pac_cvar_learner.py)
**Goal**: Use a risk-aware measure for the optimistic Estimate. Instead of taking the absolute Max (which can be unstable or over-optimistic), we take the average of the top-$\alpha$ quantile.

- **Mechanism**:
  $$Q_{CVaR} = \mathbb{E}[Q \mid Q \geq \text{VaR}_\alpha(Q)]$$
  Implemented as the mean of the top $k = \lceil \alpha \cdot N \rceil$ values of the joint Q-distribution.
  
- **Functions**:
  - `cvar_q(q_values, alpha)` in `extension.modules.optimism`.

---

## 3. Configuration & Running

### Configuration Files (`extension/configs/`)
- `adaptive_pac.yaml`: Sets `learner: "pac_adaptive_learner"`.
- `cvar_pac.yaml`: Sets `learner: "pac_cvar_learner"`.

### Key Arguments
You can pass these via CLI or edit the YAMLs.

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `optimism_schedule` | string | "linear" | "linear" or "exp" (Adaptive only). |
| `optimism_decay_steps` | int | 50000 | Duration of decay. |
| `optimism_start` | float | 1.0 | Starting alpha (usually 1.0). |
| `optimism_end` | float | 0.0 | Ending alpha (usually 0.0). |
| `cvar_alpha` | float | 0.1 | Quantile size (0.1 = top 10%). |
| `name` | string | "pac_run" | Experiment name tag for Sacred. |

### Execution Commands

```bash
# Run Adaptive (Linear 50k)
python3 extension/custom_main.py \
    --config=adaptive_pac \
    --env-config=gymma \
    with \
    env_args.key="penalty-100-nostate-v0" \
    optimism_schedule="linear" \
    optimism_decay_steps=50000 \
    name="adaptive_lin_50k"

# Run CVaR (Alpha 0.25)
python3 extension/custom_main.py \
    --config=cvar_pac \
    --env-config=gymma \
    with \
    env_args.key="climbing-nostate-v0" \
    cvar_alpha=0.25 \
    name="cvar_0.25"
```

---

## 4. Debugging & Common Issues

### "Object is not callable" TypeError
**Symptoms**: `TypeError: 'DCGCriticNS' object is not callable`
**Cause**: The critics in EPyMARL (specifically `DCGCriticNS`) are custom classes, not pure PyTorch Modules, and did not implement `__call__`.
**Fix**: Always use `.forward(batch)` explicitly instead of `critic(batch)`.
**Where**: In `train_critic` methods of learners.

### Shape Mismatches
**Context**: `q_all` shape handling.
**Detail**: The critic returns `q_values` with shape typically `[batch, t, n_agents, n_actions, n_others_joint_actions]`.
**Critical**: Ensure you operate on the correct dimension.
- `dim=3` (or -2) is usually `n_actions`.
- `dim=4` (or -1) is usually the "other agents" joint space.
- Max/Mean/CVaR operations usually happen over the LAST dimension (the distributions of other agents' responses).

---

## 5. Development Guide

### Adding a New Optimism Schedule
1. Open `extension/modules/optimism.py`.
2. Edit `OptimismScheduler.get_alpha()`.
3. Add a new `elif self.decay_type == "your_type":` block.

### Adding a New Metric
1. Initial implementation in Learner (`train` method logging).
2. Ensure it appears in `info.json` (Sacred automatically captures `log_stat` calls).
3. Update `extension/analysis/generate_report.py` to parse the new key if you want it in the summary report.

### Analyzing Results
Results are in `results/sacred/`.
- `info.json`: Contains time-series metrics (`test_return_mean`, `grad_norm`, etc.).
- `config.json`: The exact config used for the run.
- `cout.txt`: Console output.

Use `extension/analysis/generate_report.py` to auto-summarize:
```bash
python3 extension/analysis/generate_report.py
```
