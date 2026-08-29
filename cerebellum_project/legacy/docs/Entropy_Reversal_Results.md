# ECCM Model 16: Entropy-Driven Reversal

We tested whether directly mapping GC Layer Entropy to a drift-inverting exploration term forces actual switches.

## Switch Recall Results
*   **Model 12 (Baseline Tanh Divisive):** 50.91%
*   **Model 16 (Entropy Reversal):** 49.94%

## Confusion Matrix (Model 16)
| Actual \ Predicted | Predict Stay | Predict Switch |
| :--- | :--- | :--- |
| **Actual Stay** | 82.17% (±19.21%) | 17.83% (±19.21%) |
| **Actual Switch** | 48.85% (±8.73%) | 51.15% (±8.73%) |

## Confidence & Deviance
*   **Mean Switch Confidence:** 50.50%
*   **Mean Deviance:** 245.57
