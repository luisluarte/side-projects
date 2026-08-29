# Model 17: MF-Gated Exploration (Golgi + Counter-Drift)

Uses ΔMF Energy (the proven strongest switch predictor from brute-force analysis)
to inject a counter-drift that pushes AGAINST the current choice when MF energy
spikes above a learned threshold.

## Switch Recall (Target: >55%)
*   **Model 12 (Baseline Golgi Tanh):** 51.03%
*   **Model 17 (MF-Gated Exploration):** 51.39%
*   **Delta:** +0.36%

## Deviance
*   **Model 12 Mean Deviance:** 200.80
*   **Model 17 Mean Deviance:** 196.44
*   **Paired t-test p-value:** 4.5654e-02

## Switch Confidence
*   **Model 12:** 52.66%
*   **Model 17:** 52.58%

## Confusion Matrix (Model 17)
| Actual \ Predicted | Predict Stay | Predict Switch |
| :--- | :--- | :--- |
| **Actual Stay** | 86.48% (±10.92%) | 13.52% (±10.92%) |
| **Actual Switch** | 47.80% (±7.58%) | 52.20% (±7.58%) |

## Fitted Exploration Parameters (Median)
*   **MF Threshold:** 1.1404
*   **Explore Gain:** 0.4161
