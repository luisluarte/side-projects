# Cortico-Cerebellar Disagreement Modulator: Results

Testing Option 4 (Disagreement Modulator) in isolation, compared to the Temporal Decay baseline.

## 1. Confusion Matrix (Row Normalized)
| Actual \ Predicted | Predict Stay | Predict Switch |
| :--- | :--- | :--- |
| **Actual Stay** | 86.63% (±10.77%) | 13.37% (±10.77%) |
| **Actual Switch** | 48.96% (±5.53%) | 51.04% (±5.53%) |

## 2. Confidence Change (Probability assigned to True Choice)
The percentage represents the mean likelihood mass the model assigned to the correct behavior before thresholding.

| Behavior | Temporal Decay (Baseline) | Disagreement Modulator |
| :--- | :--- | :--- |
| **Stay Trials** | 0.81% (±0.10%) | 0.79% (±0.10%) |
| **Switch Trials** | 0.51% (±0.02%) | 0.52% (±0.02%) |
