# Extended Cerebellar Cognitive Model (ECCM) - Current Specification

## 1. Input Layer (Mossy Fibers)
The state space $S_t$ consists of 6 dimensions:
1. Previous Choice: $c_{t-1} \in \{-1, 1\}$
2. Previous Outcome: $o_{t-1} \in \{-1, 1\}$
3. Current Kinematic Match 1: $m_{curr} \in [-1, 1]$
4. Kinematic Difference: $m_{alt} - m_{curr} \in [-1, 1]$
5. Previous RT: Normalized
6. Previous ITI: Normalized

Mossy fibers project this state through randomized receptive fields with a delay lag $d \sim Poisson(\lambda_d)$.
$$u_{j,t} = \sigma(\beta_j (S_{t, d_j}^{(c_j)}))$$

## 2. Granule Cell Layer (Reservoir)
Granule cells integrate mossy fiber inputs and GoC inhibition.
$$in_i = \sum_{k=1}^4 W_{i,k} u_{\text{map}(i,k), t}$$
$$G_t = \frac{1}{N_{GC}} \sum in_i$$
$$z_{i, t} = \max(0, in_i + \gamma_i z_{i, t-1} - \alpha_{GoC} G_t)$$
where $\gamma_i = \rho_{base} + (1-\rho_{base})\exp(-\Delta t / \tau_i)$.

## 3. Molecular Layer Interneurons (MLI)
MLIs pool GC activity and apply dynamic thresholding.
$$H_{k,t} = \sum_{i} W^{GC \to MLI}_{k,i} z_{i,t}$$
$$\theta_t = \theta_{th} + \kappa_{th} \bar{H}_t$$
$$h_{k,t} = \max(0, H_{k,t} - \theta_t)$$

## 4. Purkinje Cells (PC) & Plasticity
Two populations of Purkinje cells encode the evidence for Choice 1 and Choice 2.
$$y_{PC1} = \sum W^{PF1}_k h_{k,t}, \quad y_{PC2} = \sum W^{PF2}_k h_{k,t}$$

Plasticity is driven by the Inferior Olive (IO) error signal via climbing fibers, modulated by an eligibility trace.
$$e_{k,t} = \gamma_e e_{k,t-1} + h_{k,t}$$
$$\delta_{IO,1} = Target_{ch} - y_{PC1}$$
$$W^{PF1}_k \leftarrow W^{PF1}_k + \eta \cdot \delta_{IO,1} \cdot e_{k,t} - \lambda W^{PF1}_k$$

## 5. Deep Cerebellar Nuclei (DCN) & Drift Rate
DCN integrates the PC difference via a low-pass filter:
$$D_t = (1-\alpha_{dcn})D_{t-1} + \alpha_{dcn}(y_{PC1} - y_{PC2})$$

Drift rate in the DDM:
$$v_t = \beta_v D_t$$
$$a_t = a_0 + \kappa_a S_{MLI}$$

## Limitations (Why it fails out-of-sample)
- **Topological Smearing**: The random mapping from MF to GC destroys the strict logical boundary needed for WSLS ($c_{t-1}$ XOR $o_{t-1}$).
- **Catastrophic Forgetting**: The eligibility trace and continuous weight updates cause the PC weights to drift wildly in out-of-sample trials if the error signals are noisy.
- **Lack of Discrete State**: The WSLS is a discrete heuristic (if win -> stay, if lose -> shift). The reservoir is purely continuous, making it incapable of strict logical branching without massive parameter fine-tuning that overfits to the training set.
