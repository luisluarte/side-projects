# Model 18: MF-Gated Choice Reversal

Instead of opposing current drift (Model 17), this directly biases toward the
OPPOSITE of the previous choice via a sigmoid-gated reversal drift triggered by
ΔMF energy spikes. Uses unconstrained threshold so the sigmoid can learn when to fire.

## Switch Recall (Target: >55%)
*   **Model 12 (Baseline Golgi Tanh):** 51.03%
*   **Model 18 (MF-Gated Choice Reversal):** 72.12%
*   **Delta:** +21.09%

## Deviance
*   **Model 12 Mean Deviance:** 200.80
*   **Model 18 Mean Deviance:** 194.12
*   **Paired t-test p-value:** 5.1260e-01

## Switch Confidence
*   **Model 12:** 52.66%
*   **Model 18:** 61.04%

## Confusion Matrix (Model 18)
| Actual \ Predicted | Predict Stay | Predict Switch |
| :--- | :--- | :--- |
| **Actual Stay** | 79.68% (±18.00%) | 20.32% (±18.00%) |
| **Actual Switch** | 30.88% (±12.86%) | 69.12% (±12.86%) |

## Fitted Parameters (Median)
*   **MF Threshold (unconstrained):** 0.3348
*   **Explore Gain:** 0.4725
