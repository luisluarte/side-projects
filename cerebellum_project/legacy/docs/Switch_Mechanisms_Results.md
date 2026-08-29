# Switch Prediction Mechanisms: Results

Comparing the baseline **Temporal Decay Model** (7 params) against the new **Combined Switch Mechanisms Model** (8 params).

## 1. Deviance Check (Is the 8th parameter justified?)
Mean Deviance Decay: 279.19
Mean Deviance Switch: 240.68
Paired t-test p-value: 1.7511e-04

## 2. Combined Switch Mechanisms (New Model)
| Actual \ Predicted | Predict Stay | Predict Switch |
| :--- | :--- | :--- |
| **Actual Stay** | 86.72% (±10.77%) | 13.28% (±10.77%) |
| **Actual Switch** | 49.01% (±6.21%) | 50.99% (±6.21%) |

## 3. Temporal Decay (Baseline)
| Actual \ Predicted | Predict Stay | Predict Switch |
| :--- | :--- | :--- |
| **Actual Stay** | 86.23% (±11.22%) | 13.77% (±11.22%) |
| **Actual Switch** | 48.74% (±9.08%) | 51.26% (±9.08%) |
