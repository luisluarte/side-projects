# Multiplexed Meta-Learning Architecture: Results

Comparing the **Cortical RPE** baseline against the new **Multiplexed** model where the Cerebellum explicitly outputs Stay and Switch meta-values rather than Target-specific Q-values.

## 1. Deviance Check
Mean Deviance Baseline (Cortical RPE): 231.75
Mean Deviance Multiplexed: 236.18
Paired t-test p-value: 9.7433e-01

## 2. Multiplexed Architecture (New Model)
| Actual \ Predicted | Predict Stay | Predict Switch |
| :--- | :--- | :--- |
| **Actual Stay** | 86.55% (±11.43%) | 13.45% (±11.43%) |
| **Actual Switch** | 49.50% (±6.69%) | 50.50% (±6.69%) |

## 3. Standard Target Architecture (Baseline)
| Actual \ Predicted | Predict Stay | Predict Switch |
| :--- | :--- | :--- |
| **Actual Stay** | 86.60% (±10.93%) | 13.40% (±10.93%) |
| **Actual Switch** | 49.58% (±5.25%) | 50.42% (±5.25%) |

## 4. Confidence Change (Probability assigned to True Choice)
| Behavior | Standard Target Output | Multiplexed Stay/Switch Output |
| :--- | :--- | :--- |
| **Stay Trials** | 78.70% (±10.51%) | 79.39% (±9.80%) |
| **Switch Trials** | 51.94% (±2.13%) | 52.06% (±2.11%) |
