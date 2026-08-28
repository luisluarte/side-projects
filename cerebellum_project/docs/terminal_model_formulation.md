# MAGI Generative Architectures: Mathematical Formulation

This document formally specifies the mathematical logic for the two dominant reaction time architectures developed during the MAGI simulation: the pure analytical **Baseline Wald Model**, and the fully optimized biological **Terminal Decoupled Hybrid (Variant 11.2)**.

---

## 1. The Baseline Wald Model (Abstract Cortex)

The Baseline Model assumes reaction time is generated entirely by a pure cortical drift-diffusion process modeled as a first-passage time through an Inverse Gaussian (Wald) distribution. The cerebellum does not exist in this topology.

### A. Cortical Value Tracking
The cortex maintains an expected value $Q_{\text{ctx}}$ for each choice, updated via standard Rescorla-Wagner learning:
$$ \delta_{\text{ctx}}^{(t)} = \text{Reward}^{(t)} - Q_{\text{ctx}}^{(t)}[ch] $$
$$ Q_{\text{ctx}}^{(t+1)}[ch] = Q_{\text{ctx}}^{(t)}[ch] + \alpha_{\text{ctx}} \cdot \delta_{\text{ctx}}^{(t)} $$

### B. Stationary Drift and Boundary
The evidence accumulation drift rate $v^{(t)}$ is directly proportional to the tracked expected value. The decision boundary $a_{\text{base}}$ is rigid and stationary.
$$ v^{(t)} = \kappa_v \cdot \max(Q_{\text{ctx}}^{(t)}[ch], \epsilon) $$

### C. Generative Draw (Rigid Diffusion)
The reaction time is drawn from a Wald distribution where the shape parameter $\lambda$ is algebraically pinned to the square of the boundary (implying standard diffusion noise $\sigma \equiv 1$). The non-decision time $t_{\text{nd}}$ is a static scalar shift.
$$ \mu_{\text{wald}}^{(t)} = \frac{a_{\text{base}}}{v^{(t)}}, \qquad \lambda_{\text{wald}} = a_{\text{base}}^2 $$
$$ RT_{\text{sim}}^{(t)} \sim \text{Wald}\left(\mu_{\text{wald}}^{(t)}, \lambda_{\text{wald}}\right) + t_{\text{nd}} $$

---

## 2. The Terminal Decoupled Hybrid (Variant 11.2)

The Terminal architecture injects the Cerebello-Thalamo-Cortical loop as a parallel, high-dimensional, fast-context spectral reinforcement learning engine. It explicitly decouples the geometrical constraints of pure diffusion to allow biological scale-free memory ($1/f$) to shape the macroscopic reaction time sequence.

### A. The Fractional Granular Manifold (Spectral $1/f$ Memory)
The cerebellar granular layer ($N$ nodes) receives a multi-modal projection of the immediate physical and environmental history (Mossy Fibers, $\mathbf{W}_{MF}$).
To generate scale-free macroscopic sequence without $O(N^2)$ recurrent computation, the node time constants $\tau_i$ are logarithmically spaced, and the sensory projection matrix is spectrally scaled:
$$ \tau_i = 1 / \Lambda_i, \qquad \mathbf{W}_{MF, i} \sim \mathcal{N}\left(0, \tau_i^{-\gamma_{\text{spectral}}}\right) $$

The deterministic sensory state $\mathbf{u}_i^{(t)}$ is perturbed by intrinsic biological Poisson noise $\zeta$ and thresholded by a Sigmoidal Golgi Gate $\theta_t$ (which tracks absolute prediction error $I_t$). 
$$ \mathbf{u}_i^{(t)} = \mathbf{W}_{MF, i} \cdot \begin{bmatrix} \text{Choice}^{(t-1)} \\ \text{RT}^{(t-1)} \\ \text{Reward}^{(t-1)} \end{bmatrix} $$
$$ \theta_t = \frac{5.0}{1 + \exp(-\text{scale}_I \cdot (I_t - 0.5))} $$
The activity $h_i$ of the granular cells follows an AR(1) fractional momentum:
$$ \tilde{h}_i^{(t)} = \max\left( \mathbf{u}_i^{(t)} + \zeta - \theta_t, 0 \right) $$
$$ h_i^{(t)} = (1 - \Lambda_i) h_i^{(t-1)} + \Lambda_i \tilde{h}_i^{(t)} $$

### B. Purkinje Readout and LASSO Plasticity
The Purkinje cells linearly combine the granular manifold to produce a parallel expected value prediction, $Q_{\text{cb}}$. The climbing fiber plasticity rule is penalized by a sparse $L_1$ subgradient ($\lambda_{\text{lasso}}$) to crush redundant nodes and protect out-of-sample generalization (The Rademacher Gate).
$$ Q_{\text{cb}}^{(t)} = \sum_{i=1}^N W_{\text{cb}, i}^{(t)} \cdot h_i^{(t)} $$
$$ \mathbf{W}_{\text{cb}}^{(t+1)} = \mathbf{W}_{\text{cb}}^{(t)} + \alpha_{\text{cb}} \cdot \delta_{\text{ctx}}^{(t)} \cdot \mathbf{h}^{(t)} - \lambda_{\text{lasso}} \cdot \text{sgn}(\mathbf{W}_{\text{cb}}^{(t)}) $$

### C. Multiplicative DCN Gain Modulation
The cerebellum scales the rate of cortical evidence accumulation via excitatory thalamic projections. The cerebellar value is routed through a bounded hyperbolic tangent to act as a multiplicative gain percentage on the cortical base value.
$$ V_{\text{eff}}^{(t)} = Q_{\text{ctx}}^{(t)}[ch] \cdot \left(1.0 + \gamma_{\text{perturb}} \cdot \tanh(Q_{\text{cb}}^{(t)})\right) $$
$$ v^{(t)} = \kappa_v \cdot \max(V_{\text{eff}}^{(t)}, \epsilon) $$

### D. Decoupled Generative Draw & Boundary Collapse
To break the rigid $\mathcal{W}_1$ geometric barrier, the terminal model introduces two critical decoupling mechanics:
1.  **Dynamic Boundary Collapse:** The decision boundary $a^{(t)}$ dynamically shrinks (or expands) based on the cerebellar fatigue state, directly compressing the late-trial right tail.
2.  **Explicit Diffusion Scale:** The Wald shape parameter is severed from $a^2$ via an explicit diffusion noise coefficient $\sigma_{\text{diff}}$, and the deterministic non-decision time floor is replaced with a lognormal physical variance $\sigma_{\text{nd}}$.

$$ a^{(t)} = a_{\text{base}} \cdot \exp\left(\delta_{\text{cb}} \cdot \tanh(Q_{\text{cb}}^{(t)})\right) $$
$$ \mu_{\text{wald}}^{(t)} = \frac{a^{(t)}}{v^{(t)}}, \qquad \lambda_{\text{wald}} = \frac{\left(a^{(t)}\right)^2}{\sigma_{\text{diff}}^2} $$

The final simulated reaction time perfectly binds the pure continuous Wiener geometry with the macroscopic $1/f$ auto-correlation generated by the cerebellar manifold:
$$ RT_{\text{sim}}^{(t)} \sim \text{Wald}\left(\mu_{\text{wald}}^{(t)}, \lambda_{\text{wald}}\right) + \text{LogNormal}\left(\log(t_{\text{nd}}), \sigma_{\text{nd}}\right) $$
