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
| CVaR-0.5 | -4341.52 ± 0.00 | 1000 |
| CVaR-0.25 | -4043.96 ± 0.00 | 1000 |
| CVaR-0.1 | -4842.96 ± 0.00 | 1000 |
| Adaptive-linear | -4245.56 ± 0.00 | 1000 |
| Adaptive-exp | -3559.96 ± 0.00 | 1000 |
| Baseline (PAC) | -2807.50 ± 268.14 | 1000 |


### Environment: climbing-nostate-v0
| Variant | Final Return | Steps |
|---|---|---|
| CVaR-0.25 | -828.90 ± 0.00 | 101000 |
| CVaR-0.1 | -692.92 ± 0.00 | 101000 |
| Adaptive-exp | -630.92 ± 0.00 | 101000 |
| Adaptive-linear | -609.48 ± 0.00 | 101000 |
| Baseline (PAC) | -614.95 ± 17.99 | 101000 |


### Environment: matrixgames:penalty-100-nostate-v0
| Variant | Final Return | Steps |
|---|---|---|
| Baseline (PAC) | -965.56 ± 0.00 | 250 |



## 4. Conclusion
Comparative analysis of the variants against the baseline.