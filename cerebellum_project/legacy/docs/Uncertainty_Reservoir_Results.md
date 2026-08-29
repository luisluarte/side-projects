# Uncertainty Reservoir: Results

Comparing the **Baseline Cortical RPE** against the new **Uncertainty Reservoir** (Learned Directed Exploration).

## 1. Deviance Check
Mean Deviance Baseline: 231.75
Mean Deviance Uncertainty Reservoir: 298.28
Paired t-test p-value: 1.7251e-04

## 2. Learned Exploration Rate ($\epsilon$)
Mean $\epsilon$ on Stay Trials: 0.5101
Mean $\epsilon$ on Switch Trials: 0.5275

## 3. Uncertainty Reservoir (New Model)
| Actual \ Predicted | Predict Stay | Predict Switch |
| :--- | :--- | :--- |
| **Actual Stay** | 48.90% (±28.65%) | 51.10% (±28.65%) |
| **Actual Switch** | 47.12% (±7.43%) | 52.88% (±7.43%) |

## 4. Baseline Model
| Actual \ Predicted | Predict Stay | Predict Switch |
| :--- | :--- | :--- |
| **Actual Stay** | 86.60% (±10.93%) | 13.40% (±10.93%) |
| **Actual Switch** | 49.58% (±5.25%) | 50.42% (±5.25%) |
