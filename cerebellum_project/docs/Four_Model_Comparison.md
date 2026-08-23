# 4-Model Comparison: Confusion Matrices & PR-AUC

Comparing **WSLS**, **Q-Learning with Counterfactual Update**, **Baseline ECCM (Cortical RPE)**, and **MF-Gated Choice Reversal** (Model 18).

Switch is the **positive class** for PR-AUC.

## WSLS

| Actual \ Predicted | Predict Stay | Predict Switch |
| :--- | :--- | :--- |
| **Actual Stay** | 78.24% (±9.39%) | 21.76% (±9.39%) |
| **Actual Switch** | 21.49% (±14.48%) | 78.51% (±14.48%) |

*   **Aggregate Switch Recall:** 73.94%
*   **Mean PR-AUC (Switch+):** 0.4731 (±0.1794)

## Q-Learning (CF)

| Actual \ Predicted | Predict Stay | Predict Switch |
| :--- | :--- | :--- |
| **Actual Stay** | 89.80% (±6.25%) | 10.20% (±6.25%) |
| **Actual Switch** | 40.39% (±9.58%) | 59.61% (±9.58%) |

*   **Aggregate Switch Recall:** 57.21%
*   **Mean PR-AUC (Switch+):** 0.6080 (±0.1317)

## Baseline ECCM

| Actual \ Predicted | Predict Stay | Predict Switch |
| :--- | :--- | :--- |
| **Actual Stay** | 86.96% (±10.94%) | 13.04% (±10.94%) |
| **Actual Switch** | 49.83% (±7.30%) | 50.17% (±7.30%) |

*   **Aggregate Switch Recall:** 50.67%
*   **Mean PR-AUC (Switch+):** 0.5699 (±0.1689)

## MF-Gated Reversal

| Actual \ Predicted | Predict Stay | Predict Switch |
| :--- | :--- | :--- |
| **Actual Stay** | 79.68% (±18.00%) | 20.32% (±18.00%) |
| **Actual Switch** | 30.88% (±12.86%) | 69.12% (±12.86%) |

*   **Aggregate Switch Recall:** 72.12%
*   **Mean PR-AUC (Switch+):** 0.5683 (±0.1736)

---

## Summary Table

| Model | Switch Recall | PR-AUC (Switch+) | Stay Accuracy |
| :--- | :---: | :---: | :---: |
| **WSLS** | 73.94% | 0.4731 | 78.24% |
| **Q-Learning (CF)** | 57.21% | 0.6080 | 89.80% |
| **Baseline ECCM** | 50.67% | 0.5699 | 86.96% |
| **MF-Gated Reversal** | 72.12% | 0.5683 | 79.68% |
