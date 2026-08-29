# 5-Model Comparison: Asymmetric Exploration

Does splitting θ_explore into separate **win** and **loss** gains improve switch prediction and PR-AUC?

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

## MF-Reversal (Sym)

| Actual \ Predicted | Predict Stay | Predict Switch |
| :--- | :--- | :--- |
| **Actual Stay** | 79.68% (±18.00%) | 20.32% (±18.00%) |
| **Actual Switch** | 30.88% (±12.86%) | 69.12% (±12.86%) |

*   **Aggregate Switch Recall:** 72.12%
*   **Mean PR-AUC (Switch+):** 0.5683 (±0.1736)

## MF-Reversal (Asym)

| Actual \ Predicted | Predict Stay | Predict Switch |
| :--- | :--- | :--- |
| **Actual Stay** | 81.44% (±11.63%) | 18.56% (±11.63%) |
| **Actual Switch** | 25.59% (±9.97%) | 74.41% (±9.97%) |

*   **Aggregate Switch Recall:** 72.73%
*   **Mean PR-AUC (Switch+):** 0.5934 (±0.1531)

---

## Summary Table

| Model | Switch Recall | PR-AUC (Switch+) | Stay Accuracy |
| :--- | :---: | :---: | :---: |
| **WSLS** | 73.94% | 0.4731 | 78.24% |
| **Q-Learning (CF)** | 57.21% | 0.6080 | 89.80% |
| **Baseline ECCM** | 50.67% | 0.5699 | 86.96% |
| **MF-Reversal (Sym)** | 72.12% | 0.5683 | 79.68% |
| **MF-Reversal (Asym)** | 72.73% | 0.5934 | 81.44% |

## Asymmetric Parameters

*   **θ_explore_win (median):** 0.1068
*   **θ_explore_loss (median):** 0.9430
*   **Loss/Win Ratio:** 8.83x

## Deviance Comparison (Model 18 vs 19)

*   **Model 18 (Symmetric) Mean Deviance:** 194.12
*   **Model 19 (Asymmetric) Mean Deviance:** 194.68
*   **Paired t-test p-value:** 8.8548e-01
