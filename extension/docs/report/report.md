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
| CVaR-0.25 | -19.55 ± 21.00 | 501000 |
| CVaR-0.1 | -19.55 ± 21.00 | 501000 |
| CVaR-0.5 | -14.65 ± 12.60 | 501000 |
| Adaptive-exp | -3744.24 ± 240.70 | 501000 |
| Adaptive-linear | -3787.56 ± 223.82 | 501000 |
| Baseline (PAC) | -3592.34 ± 614.93 | 501000 |


### Environment: climbing-nostate-v0
| Variant | Final Return | Steps |
|---|---|---|
| CVaR-0.1 | -582.93 ± 2701.43 | 501000 |
| CVaR-0.25 | -582.93 ± 2701.43 | 501000 |
| CVaR-0.5 | -584.48 ± 2701.02 | 501000 |
| Adaptive-linear | 997.68 ± 1.44 | 501000 |
| Adaptive-exp | 997.68 ± 1.44 | 501000 |
| Baseline (PAC) | 196.22 ± 622.78 | 501000 |


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