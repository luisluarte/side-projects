# Golgi Cell Bounded ReLU: Results

Comparing the **Golgi (tanh)** against the new **Golgi (ReLU)**.

## 1. Deviance Check
Mean Deviance Golgi (tanh): 207.88
Mean Deviance Golgi (ReLU): 217.05
Paired t-test p-value: 8.2195e-02

## 2. Switch Confidence (Probability assigned to True Switch)
Mean Switch Confidence Golgi (tanh): 52.35%
Mean Switch Confidence Golgi (ReLU): 52.21%

## 3. Golgi Inhibition (ReLU Model)
| Actual \ Predicted | Predict Stay | Predict Switch |
| :--- | :--- | :--- |
| **Actual Stay** | 85.74% (±11.09%) | 14.26% (±11.09%) |
| **Actual Switch** | 49.35% (±5.74%) | 50.65% (±5.74%) |

## 4. Golgi Inhibition (tanh Model)
| Actual \ Predicted | Predict Stay | Predict Switch |
| :--- | :--- | :--- |
| **Actual Stay** | 86.30% (±11.07%) | 13.70% (±11.07%) |
| **Actual Switch** | 48.23% (±7.67%) | 51.77% (±7.67%) |
