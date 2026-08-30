---
editor_options:
  markdown:
    wrap: 72
output: pdf_document
---

# Complete Mathematical Formulation: M006 Clamped (Optimized MAGI)

The **M006 Clamped** architecture (also referred to as the Kernelized
Symplectic-HDDM or Optimized MAGI) represents a biologically plausible
cognitive model bridging deep cerebellar reservoir computing with the
cortical Hierarchical Drift-Diffusion Model (HDDM).

This document strictly defines the exact mathematical sequence evaluated
for every subject trial $t$. All boundaries are governed by continuously
differentiable algebraic squashing (using $\tanh$ and
$\text{logit}^{-1}$) to guarantee a strict, unbroken manifold for
Hamiltonian Monte Carlo (HMC) sampling without Jacobian singularities.

## 1. Cortical Action Value Tracking (The Cortex)

The cortex maintains an action value $Q_{c}^{(t)}$ for choice
$c \in \{1, 2\}$, updated via a standard Rescorla-Wagner learning rule
on trial $t$. \* **Prediction Error**: $E_t = R_t - Q_{c_t}^{(t)}$ \*
**Action Value Update**:
$Q_{c_t}^{(t+1)} = Q_{c_t}^{(t)} + \alpha_{\text{ctx}} E_t$ \*
**Cortical Action Gradient**: The difference in action values is tracked
as an integrated scalar:
$$Q_{\text{diff}}^{(t)} = Q_2^{(t)} - Q_1^{(t)}$$

## 2. Cerebellar Temporal Extension (Mossy Fiber Integration)

The cerebellum observes the cortical action values $Q_{c_t}^{(t)}$
through a set of random, fixed topological projections
$\mathbf{W}_{\text{exp}} \in \mathbb{R}^{32}$. It tracks them via leaky
integrators $\mathbf{M}^{(t)} \in \mathbb{R}^{32}$, where each temporal
node maintains a distinct decay scale. \* **Memory Trace Update**:
$$\mathbf{M}^{(t)} = \boldsymbol{\alpha}_{\text{frac}} \odot \mathbf{M}^{(t-1)} + (\mathbf{1} - \boldsymbol{\alpha}_{\text{frac}}) \odot \left( \mathbf{W}_{\text{exp}} \times Q_{c_t}^{(t)} \right)$$
*Where* $\boldsymbol{\alpha}_{\text{frac}} \in \mathbb{R}^{32}$ is a
uniform gradient spanning $[0.1, 0.9]$.

## 3. Purkinje Cell Reservoir Dynamics

The 32-node Purkinje population state
$\mathbf{Z}^{(t)} \in \mathbb{R}^{32}$ dynamically integrates the mossy
fiber traces with a biophysical, ITI-dependent (Inter-Trial Interval)
exponential decay. \* **Biological Decay Factor**:
$\phi_{\text{decay}}^{(t)} = \exp\left( -\frac{\text{ITI}_t}{\tau_{\text{decay}}} \right)$
\* **Purkinje State**:
$$\mathbf{Z}^{(t)} = \phi_{\text{decay}}^{(t)} (\boldsymbol{\kappa} \odot \mathbf{Z}^{(t-1)}) + \tanh(\mathbf{M}^{(t)})$$
*Where* $\boldsymbol{\kappa} \in \mathbb{R}^{32}$ is a retention
constant scaling linearly from $0.1$ to $0.99$.

## 4. Synaptic Weight Matrix & Error Tracking

The Purkinje-to-Deep Cerebellar Nuclei (DCN) weights
$\mathbf{W}_{\text{PC}} \in \mathbb{R}^{32}$ associate specific spatial
Purkinje patterns with incoming cortical prediction errors. \* **Latent
Weight Update**:
$$\mathbf{W}_{\text{PC, latent}}^{(t+1)} = \mathbf{W}_{\text{PC, latent}}^{(t)} + \alpha_{\text{pc}} E_t \mathbf{Z}^{(t)}$$
\* **Effective Clamped Weights**: To prevent asymptotic divergence in
HMC, the latent weights are strictly bounded using a scaled hyperbolic
tangent mapping:
$$\mathbf{W}_{\text{eff}}^{(t)} = 3.0 \times \tanh\left( \frac{\mathbf{W}_{\text{PC, latent}}^{(t)}}{3.0} \right)$$

## 5. Symplectic Rebound & Golgi Sparsification

The effective Purkinje signal evaluates the alignment between current
reservoir patterns and learned weights. A Golgi-cell-like inhibitory
mask assesses spatial entropy. \* **Effective Trace**:
$\mathbf{E}_{\mathbf{Z}}^{(t)} = \mathbf{W}_{\text{eff}}^{(t)} \odot \mathbf{Z}^{(t)}$
\* **Differentiable Sparsity Mask**:
$$\mathbf{S}^{(t)} = \tanh\left( \gamma_{\text{golgi}} \sqrt{ \mathbf{E}_{\mathbf{Z}}^{(t)} \odot \mathbf{E}_{\mathbf{Z}}^{(t)} + 10^{-8} } \right)$$
\* **Bilateral Rebound Consolidation**: The 32 nodes are pooled into two
opposing streams (representing competing cerebellar microzones):
$$CB_0^{(t)} = \mathbf{S}_{1:16}^{(t)} \cdot \mathbf{E}_{\mathbf{Z}, 1:16}^{(t)}$$
$$CB_1^{(t)} = \mathbf{S}_{17:32}^{(t)} \cdot \mathbf{E}_{\mathbf{Z}, 17:32}^{(t)}$$

## 6. Neuromodulatory DDM Formulation (Dynamic Gating)

The cerebellum outputs its spatial entropy metric to the Locus Coeruleus
(LC) and Basal Ganglia to dynamically gate the Drift-Diffusion bounds.

### 6.1. Dynamic Drift Rate ($v_t$)

Drift rate represents momentary directional evidence, driven by the
cortical value gradient and cerebellar temporal foresight. It is bounded
to $\pm 18.51$ (empirical physiological bound). \* **Scaled Drift
Velocity**:
$$v_{\text{eff}}^{(t)} = v_{\text{ctx}} Q_{\text{diff}}^{(t)} + \gamma_{\text{var}} \left( CB_1^{(t)} - CB_0^{(t)} \right)$$
\* **Clamped Drift Rate**:
$$v_t = 18.51 \times \tanh(v_{\text{eff}}^{(t)} \times 0.0540248)$$

### 6.2. Dynamic Boundary Separation ($a_t$)

Boundary separation tracks Epistemic Deliberation. When both microzones
exhibit high structural entropy (large magnitudes of opposing errors
tracked across different timescales), the model detects complexity and
triggers a deliberative state by expanding the decision boundary. If
entropy is low, the boundary collapses, enabling a fast
Win-Stay/Lose-Shift heuristic. \* **Raw Boundary Space**:
$$a_{\text{raw}}^{(t)} = a_{\text{base}} + w_u \sqrt{ \left( (CB_0^{(t)})^2 + 10^{-8} \right) \left( (CB_1^{(t)})^2 + 10^{-8} \right) }$$
\* **Clamped Boundary Mapping**:
$$a_t = 0.11 + 7.36 \times \text{logit}^{-1}(a_{\text{raw}}^{(t)})$$

## 7. Joint Likelihood (Wiener Diffusion)

The final trial behavioral prediction (Reaction Time and Choice) is
evaluated probabilistically through the Wiener First-Passage Time
distribution (assuming starting bias $w = 0.5$).
$$RT_t \sim \text{Wiener}(a_t, \, tnd, \, 0.5, \, v_t)$$
