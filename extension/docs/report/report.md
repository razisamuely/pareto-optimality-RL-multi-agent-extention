# PAC Extension Experiment Report

## 1. Introduction
This report summarizes the performance of Adaptive Optimism and CVaR Optimism variants of the Pareto Actor-Critic (PAC) algorithm.
Experiments were conducted on the Penalty Game (k=-100) and Climbing Game.

## 2. Implementation Details
- **Adaptive Optimism**: Linearly decays the optimism coefficient from 1.0 (Max Q) to 0.0 (Mean Q) over 50k steps.
- **CVaR Optimism**: computes the mean of the top-$\alpha$ quantile of Q-values.

## 3. Results
### Environment: penalty-100-nostate-v0
| Variant | Final Return | Steps |
|---|---|---|
| CVaR-0.5 | -4305.12 ± 0.00 | 101000 |
| CVaR-0.25 | -4043.96 ± 0.00 | 101000 |
| CVaR-0.1 | -4842.96 ± 0.00 | 101000 |
| Adaptive-linear | -4245.56 ± 0.00 | 101000 |
| Adaptive-exp | -3559.96 ± 0.00 | 101000 |
| Baseline (PAC) | -2807.50 ± 268.14 | 101000 |


### Environment: climbing-nostate-v0
| Variant | Final Return | Steps |
|---|---|---|
| CVaR-0.1 | -601.26 ± 2466.40 | 501000 |
| CVaR-0.25 | -625.22 ± 2284.70 | 501000 |
| CVaR-0.5 | -751.56 ± 0.00 | 101000 |
| Adaptive-linear | 729.82 ± 598.95 | 501000 |
| Adaptive-exp | 726.24 ± 606.94 | 501000 |
| Baseline (PAC) | -35.54 ± 641.42 | 501000 |


### Environment: matrixgames:climbing-nostate-v0
| Variant | Final Return | Steps |
|---|---|---|
| baseline | 1400.00 ± 0.00 | 2050100 |
| Baseline (PAC) | -618.11 ± 18.09 | 1000 |


### Environment: matrixgames:penalty-100-nostate-v0
| Variant | Final Return | Steps |
|---|---|---|
| baseline | 0.00 ± 0.00 | 25 |
| Baseline (PAC) | -965.56 ± 0.00 | 250 |



## 4. Conclusion
Comparative analysis of the variants against the baseline.