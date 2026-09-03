# Cerebellar Model Coding Directory

This document provides the canonical reference codes for all formal models evaluated in this project, explicitly separating them by their target metric (Spatial vs. Meta-Decision).

## 1. Spatial Models (\-SP\)
*Evaluated on the exact physical choice (Left vs. Right).*

| Code | Stan File | Description |
| :--- | :--- | :--- |
| **\M012-SP\** | \src/stan/m012_spatial.stan\ | The 4-Node Cerebellar Temporal Reservoir. The Cerebellum predicts reversals and directly votes on the drift rate, outperforming Q-learning on spatial targets. |
| **\VOPT-SP\** | \src/stan/vopt_spatial.stan\ | The Optimal Q-Learning Baseline without a Cerebellum. Fails to predict block reversals efficiently. |
| **\WSLS-SP\** | \src/stan/wsls_spatial.stan\ | The Win-Stay, Lose-Shift hardcoded heuristic model. Acts as a baseline reflex/acquired meta-strategy to benchmark empirical choices. |

## 2. Meta-Decision Models (\-MD\)
*Evaluated on the reactive strategic choice (Stay vs. Switch).*

| Code | Stan File | Description |
| :--- | :--- | :--- |
| **\M012-MD\** | \src/stan/m012_ss3.stan\ | Evaluates M012 on the Stay/Switch target. Collapses completely because the Cerebellum tracks temporal rhythm, not reactive reflexes. |
| **\VOPT-MD\** | \src/stan/vopt_ss3.stan\ | Evaluates Q-learning on Stay/Switch. Mathematically outperforms M012 here, proving the Cerebellar tracker's failure is specific to non-temporal reactive targets. |
| **\WSLS-MD\** | \src/stan/wsls_ss3.stan\ | The WSLS heuristic evaluated against Stay/Switch behavior. |

