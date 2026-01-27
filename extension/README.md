# PAC Extension: Equilibrium Selection Variants

Three modifications to Pareto Actor-Critic for improved equilibrium selection in cooperative multi-agent reinforcement learning.

## Variants

1. **Adaptive Optimism** - Gradually decay from optimistic to realistic Q-value estimates during training
2. **CVaR Optimism** - Average top-α Q-values instead of taking the maximum
3. **Communication** - Add message passing channel before action selection

## Setup

All variants import from base EPyMARL. No modifications to core codebase required.

## Structure

```
extension/
├── learners/          # PAC variant implementations
├── modules/           # New components (optimism, communication)
├── configs/           # YAML configurations
├── experiments/       # Run scripts
├── analysis/          # Plotting and statistics
├── results/           # Experiment outputs
├── docs/              # LaTeX report
└── tests/             # Unit and integration tests
```

## Running Experiments

See `experiments/` directory for run scripts:

```bash
# Run single variant
./extension/experiments/run_adaptive.sh

# Run all variants
./extension/experiments/run_all_variants.sh
```

## Testing

```bash
# Run all tests
pytest extension/tests/ -v

# Run specific variant tests
pytest extension/tests/test_adaptive_learner.py -v
```


plan - /home/corsound/.gemini/antigravity/brain/cfeb0fe0-d4d7-4076-98cb-3d91c33ffdaa/setup_testing_plan.md.resolved