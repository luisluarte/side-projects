# Golgi Variants: Pushing for +5% Switch Recall

We tested two new biologically-inspired algorithms explicitly designed to maximize entropy during volatility.

## Switch Recall Results
*   **Model 12 (Baseline Tanh Divisive):** 50.91%
*   **Model 14 (ReLU Ceiling Inhibition):** 51.15%
*   **Model 15 (Temperature Softmax):** 50.30%

## Analysis
Model 14 forces entropy by applying a dynamic Ceiling (shunting) that drops during high MF energy, capping all highly active nodes at exactly the same small value (uniformity).
Model 15 explicitly guarantees max entropy by using a Softmax function where the Temperature parameter rises with MF energy, making the distribution perfectly uniform during high volatility.

