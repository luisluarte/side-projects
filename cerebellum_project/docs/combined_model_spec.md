# Combined ECCM Specification

## 1. Core Architecture
The Extended Cerebellar Cognitive Model (ECCM) models a continuous recurrent system representing the cerebellar microcircuitry, mapping internal and external state via Mossy Fibers (MF), through a Granular Cell (GC) hidden layer, and finally to a Purkinje Cell / Molecular Layer Interneuron (MLI) readout.

**Dimensionality:** 
*   `N_MF` = 20
*   `N_GC` = 200
*   `N_MLI` = 40

## 2. Input State (Mossy Fibers)
15-dimensional sliding window vector ($d \in [0, 14]$):
1.  Previous Choice ($C_{t-1} \in \{-1, 1\}$)
2.  Previous Outcome ($O_{t-1} \in \{-1, 1\}$)
3.  Current Expected Value
4.  Current Expected Value Difference
5.  Previous Reaction Time (normalized)
6.  Previous Inter-Trial Interval (normalized)

Mossy fibers possess non-linear activation $\sigma(\beta x)$ with temporal delay kernels $d \in [0, 14]$.

## 3. Granular Layer & Golgi Cell (GoC) Feedback
Each GC receives input from 4 MFs.
GoC explicitly subtracts the global mean MF activation to force extreme sparsity on the Granular layer:
$G^{in}_t = \frac{1}{N_{GC}} \sum_{i} \sum_{k} W^{MF \to GC}_{i, k} U^{MF}_t$

$Z^{GC}_{i, t} = \max\left(0, \sum_k W_{i, k} U^{MF}_t + \gamma Z^{GC}_{i, t-1} - \alpha_{GoC} G^{in}_t\right)$

## 4. Molecular Layer Interneurons (MLI) & DAT
MLIs pool from all GCs via sparse weights.
Dynamic Adaptive Thresholding (DAT) scales the activation threshold based on global MLI activity to maintain stable sparse firing.
$H_t = \frac{1}{N_{MLI}} \sum_k \sum_i W^{GC \to MLI}_{k, i} Z^{GC}_{i, t}$
$\Theta_t = \Theta_0 + \kappa_{th} H_t$
$h^{MLI}_{k, t} = \max(0, \sum_i W^{GC \to MLI} Z^{GC}_{i, t} - \Theta_t)$

## 5. Purkinje Cell Output & Action Selection
Two Purkinje Cell populations (PC1, PC2) linearly read out from the MLIs:
$y^{PC1}_t = \sum_k W^{PF1}_k h^{MLI}_{k, t}$
$y^{PC2}_t = \sum_k W^{PF2}_k h^{MLI}_{k, t}$

The drift rate for the HDDM is driven by the difference in PC activity:
$v_t = \beta_v (y^{PC1}_t - y^{PC2}_t)$

## 6. Plasticity (LTD / LTP)
Current plasticity is an error-driven delta rule on the active PF weights.
$Target = \begin{cases} 1 & \text{if Reward} \\ -1 & \text{if No Reward} \end{cases}$

$\Delta IO_{PC1} = Target - y^{PC1}_t$
$\Delta W^{PF1}_k = \eta \Delta IO_{PC1} h^{MLI}_{k, t} - \lambda W^{PF1}_k$
*(Same for PC2)*

## 7. Current Weakness
The current error signal pushes the network to predict the reward directly. When the participant receives a "No Reward" ($-1$) for Choice 1, the model simply lowers the value of $W^{PF1}$ towards $-1$. However, this does not explicitly push the model to strongly favor Choice 2 on the next trial. The model struggles to replicate the instantaneous "Win-Stay, Lose-Shift" behavior observed in humans.
