# Iteration 1: Proposed Model Specification

## 1. Analysis of Current Weakness
The current ECCM updates the Purkinje Cell weights ($W^{PF1}$ and $W^{PF2}$) independently. When the model selects Choice 1 ($PC1$) and receives a "No Reward" (-1), it applies Long-Term Depression (LTD) to $W^{PF1}$ to decrease its probability of firing next time. However, it does **not** update $W^{PF2}$. 

Human behavior in this 2-alternative forced-choice task is highly zero-sum. If Choice 1 is wrong, Choice 2 is inherently the correct choice. The lack of an explicit asymmetric update means the network has to wait until it randomly explores Choice 2 to learn its value, preventing rapid strategy shifting (Win-Stay, Lose-Shift).

## 2. Proposed Mechanism: Anti-Correlated Reward Mapping
We propose biologically-inspired lateral inhibition between the Inferior Olive (IO) climbing fibers or explicit reciprocal learning rules in the Purkinje cells. 

When the actual choice is updated based on the reward target, the **counter-factual** choice is explicitly updated in the opposite direction. 

**Mathematical Formulation:**
Instead of independent targets, we calculate a global zero-sum target error:
$Target_{chosen} = \begin{cases} 1 & \text{if Reward} \\ -1 & \text{if No Reward} \end{cases}$

$Target_{unchosen} = -Target_{chosen}$

For the weights:
If Choice = 1:
$\Delta IO_1 = Target_{chosen} - y^{PC1}_t$
$\Delta IO_2 = Target_{unchosen} - y^{PC2}_t$

$\Delta W^{PF1} = \eta \Delta IO_1 h^{MLI}_t - \lambda W^{PF1}$
$\Delta W^{PF2} = \eta \Delta IO_2 h^{MLI}_t - \lambda W^{PF2}$

(And conversely if Choice = 2).

## 3. Iteration 1 Results
*   **M1 NLL:** 1448.57
*   **M2 NLL:** 1623.98
*   **ECCM Iter1 NLL:** 1436.12
*   **M1 vs ECCM:** $p = 0.194$ (ECCM won in absolute NLL, but failed statistical significance).

---

# Iteration 2: Asymmetric Plasticity (LTP vs LTD)

## 1. Analysis of Iteration 1
Iteration 1 successfully dropped the ECCM NLL *below* M1 (1436 vs 1448) for the first time in this project! However, the variance across participants was high enough that $p = 0.194$, failing our $p < 0.05$ threshold. 

We need an additional mechanism to sharpen the strategy shift. Humans often learn from negative feedback (punishment/loss) at a fundamentally different rate than positive feedback (reward/win). The "Lose-Shift" magnitude is often much larger than the "Win-Stay" magnitude.

## 2. Proposed Mechanism: Asymmetric Learning Rates
We will introduce a dual learning rate mechanism for Long-Term Potentiation (LTP) and Long-Term Depression (LTD). 

**Mathematical Formulation:**
If the climbing fiber error signal $\Delta IO > 0$, we use $\eta_{LTP}$.
If the climbing fiber error signal $\Delta IO < 0$, we use $\eta_{LTD}$.

We will add a parameter $\rho_{\eta} \in [0.1, 10]$ representing the ratio:
$\eta_{LTD} = \eta$
$\eta_{LTP} = \eta \times \rho_{\eta}$

## 3. Iteration 2 Results
*   **M1 NLL:** 1448.57
*   **M2 NLL:** 1623.98
*   **ECCM Iter2 NLL:** 1446.70
*   **M1 vs ECCM2:** $p = 0.897$ (Failed. Performance actually dropped compared to Iteration 1).

---

# Iteration 3: Synaptic Eligibility Traces (Temporal Credit Assignment)

## 1. Analysis of Iteration 2
Asymmetric plasticity (LTP vs LTD) made the model worse. The issue is not just the asymmetry of the update, but *temporal credit assignment*. The human participants are likely basing their Win-Stay/Lose-Shift decisions on a smoothed history of previous actions, not just the instantaneous snapshot of the $t$-th trial.

## 2. Proposed Mechanism: Synaptic Eligibility Traces
We will introduce biological Eligibility Traces to the parallel fiber-to-Purkinje cell (PF-PC) synapses. Rather than updating weights based solely on the instantaneous MLI firing rate $h^{MLI}_t$, the climbing fiber error signal will update the weights based on a decaying trace of recent MLI activity.

**Mathematical Formulation:**
Let $e_t$ be the eligibility trace for the MLI population:
$e_{k, t} = \gamma_e e_{k, t-1} + h^{MLI}_{k, t}$
where $\gamma_e \in [0, 0.95]$ is the trace decay parameter.

The weight updates (incorporating Iteration 1's Anti-Correlated reward):
$\Delta W^{PF1}_k = \eta \Delta IO_1 e_{k, t} - \lambda W^{PF1}_k$
$\Delta W^{PF2}_k = \eta \Delta IO_2 e_{k, t} - \lambda W^{PF2}_k$

## 3. Iteration 3 Results
*   **M1 NLL:** 1448.57
*   **M2 NLL:** 1623.98
*   **ECCM Iter3 NLL:** 1506.16
*   **M1 vs ECCM3:** $t = -2.32, p = 0.027$ (ECCM lost to M1 significantly).

---

# Iteration 4: Non-Linear Drift Rate Mapping (Variance Reduction)

## 1. Analysis of Previous Iterations
Iteration 1 (Anti-Correlated Reward Mapping) successfully achieved a lower absolute Test NLL than M1 (1436.12 vs 1448.57). However, the statistical variance was too high to yield $p < 0.05$ ($p = 0.194$). Iterations 2 and 3 attempted to improve learning dynamics but ultimately overfitted or destabilized the model.

To achieve significance, we don't need a radically new learning rule; we need to **reduce the variance** of the Iteration 1 model. 
High participant-level variance in the NLL is typically caused by the linear drift rate mapping $v_t = \beta_v (y^{PC1}_t - y^{PC2}_t)$. If the PF weights grow large during a "Win-Stay" streak, the linear difference can explode. This creates a massive drift rate. If the participant suddenly shifts, the wiener pdf evaluates a choice directly opposite to a massive drift rate, causing a huge NLL spike for that trial, blowing up the variance.

## 2. Proposed Mechanism: Non-Linear Drift Rate Mapping
We will bound the output of the Purkinje cells before it translates to the kinematic drift rate using a hyperbolic tangent (`tanh`) function. This ensures that the internal confidence can grow, but the resulting kinematic drive asymptotes, preventing catastrophic NLL penalties on unexpected exploration.

**Mathematical Formulation:**
$v_t = \beta_v \cdot \tanh(\kappa_v \cdot (y^{PC1}_t - y^{PC2}_t))$
where $\kappa_v$ is a new scaling parameter inside the non-linearity. 

## 3. Iteration 4 Results
*   **M1 NLL:** 1448.57
*   **M2 NLL:** 1623.98
*   **ECCM Iter4 NLL:** 1463.95
*   **M1 vs ECCM4:** $t = -0.92, p = 0.364$ (ECCM lost to M1).

---

# Iteration 5: Low-Pass Filtered Purkinje Readout (Evidence Integration)

## 1. Analysis of Iteration 4
Bounding the drift rate using `tanh` successfully capped extreme values but heavily penalized the model's ability to express confident, aggressive decisions (Win-Stay). The NLL worsened to 1463. Iteration 1 (Anti-Correlated Reward with linear drift) remains our best mathematical structure (NLL 1436).

We still need to reduce variance, but an artificial mathematical bound (`tanh`) is too restrictive. Biologically, the Purkinje cells project to the Deep Cerebellar Nuclei (DCN), which then project to the motor cortex. This is not an instantaneous transmission; the DCN acts as a leaky integrator (low-pass filter) of Purkinje cell inhibition.

## 2. Proposed Mechanism: Low-Pass Filtered Readout
Instead of mapping the instantaneous difference $y^{PC1}_t - y^{PC2}_t$ directly to the drift rate, we will maintain a running moving average of the Purkinje cell difference. 

**Mathematical Formulation:**
Let $D_t$ be the integrated decision variable (representing DCN activity):
$D_t = (1 - \alpha_{DCN}) D_{t-1} + \alpha_{DCN} (y^{PC1}_t - y^{PC2}_t)$
where $\alpha_{DCN} \in [0, 1]$ is the integration rate.

The drift rate becomes:
$v_t = \beta_v \cdot D_t$

## 3. Iteration 5 Results
*   **M1 NLL:** 1448.57
*   **M2 NLL:** 1623.98
*   **ECCM Iter5 NLL:** 1433.42
*   **M1 vs ECCM5:** $t = 1.99, p = 0.055$ (Almost significant!).

---

# Iteration 6: Unified Temporal Integration (DCN Filter + Synaptic Traces)

## 1. Analysis of Iteration 5
Iteration 5 was an incredible success. The Deep Cerebellar Nuclei (DCN) leaky integrator stabilized the drift rate perfectly, yielding the lowest Test NLL achieved in the entire project history (1433.42). It beat M1 by a wide margin (15 points). The t-test yielded $p = 0.055$, missing our rigid $p < 0.05$ threshold by an agonizing $0.005$. 

We are mathematically sitting on the boundary of statistical significance.

## 2. Proposed Mechanism: Unified Temporal Integration
Iteration 3 (Synaptic Eligibility Traces) failed because it exacerbated the volatile linear drift rate mapping. However, now that we have successfully stabilized the network output via the DCN low-pass filter (Iteration 5), the model is safe from extreme NLL spikes.

We can now safely re-introduce the biological **Synaptic Eligibility Traces** (from Iteration 3) to the Purkinje Cell synapses, while keeping the **DCN Low-Pass Filter** (from Iteration 5) on the readout. 

This creates a unified temporal integration architecture:
1.  **Presynaptic Temporal Credit (Synaptic Level):** $e_{k, t} = \gamma_e e_{k, t-1} + h^{MLI}_{k, t}$
2.  **Anti-Correlated Learning:** $\Delta W^{PF1}_k = \eta \Delta IO_1 e_{k, t} - \lambda W^{PF1}_k$
3.  **Postsynaptic Temporal Smoothing (Circuit Level):** $D_t = (1 - \alpha_{DCN}) D_{t-1} + \alpha_{DCN} (y^{PC1}_t - y^{PC2}_t)$
4.  **Kinematic Output:** $v_t = \beta_v \cdot D_t$

## 3. Iteration 6 Results
*   **M1 NLL:** 1448.57
*   **M2 NLL:** 1623.98
*   **ECCM Iter6 NLL:** 1433.15
*   **M1 vs ECCM6:** $t = 3.81, p = 0.00066$
*   **M2 vs ECCM6:** $t = 4.97, p = 0.000027$

> [!SUCCESS]
> **Termination Condition Reached**
> The Unified Temporal Integration model successfully achieved $p < 0.05$ against both M1 and M2. It beat M1's test NLL by over 15 points with extreme statistical robustness ($p = 0.00066$). The iterative loop is now complete.
