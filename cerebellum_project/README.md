# Thermodynamic Cortico-Cerebellar Cognitive Architecture

This repository contains the mathematical formulation, C++ core engines, R optimization orchestrators, and benchmark visual proof suites for the bio-thermodynamic cortico-cerebellar architectures developed to model joint choice-RT dynamics and cognitive switching under the Navarro-Fuss (2009) Wiener First-Passage Time (WFPT) density.

---

## ??? Repository Structure

```
cerebellum_project/
+-- data/
¦   +-- raw/
¦   ¦   +-- behavioral_compilate.csv        # Empirical human multi-alternative choice & RT dataset
¦   +-- processed/
+-- docs/
¦   +-- CC_Baseline_Formulation.md          # Baseline: Cortical Q-Learning + Static Wald DDM
¦   +-- CC_WFPT_SWARM_005_Formulation.md    # M005: Opponent-Process + Gap-Junction Prior + Local LTD/LTP
¦   +-- CC_WFPT_SWARM_006_Formulation.md    # M006 (Champion): Continuous ITI Decay + Dynamic Doubt Boundary
+-- figures/
¦   +-- magi_5panel_distribution.png        # 5-panel distribution benchmarking (RT-RMSE, Brier, PR/ROC-AUC, NLL)
¦   +-- magi_iti_bifurcation.png            # Thermodynamic posterior predictive bifurcated by ITI quartile
¦   +-- magi_epistemic_boundary.png         # Empirical deliberation spikes matched to dynamic boundary expansion
¦   +-- magi_supremacy_forest.png           # Subject-level delta likelihood forest plot
+-- results/
¦   +-- subject_metrics_real.csv            # Subject-level optimization metrics across benchmarked models
¦   +-- trial_metrics_real_006.csv          # Trial-by-trial state spaces and internal boundary/conflict dynamics
+-- src/
¦   +-- models/
¦   ¦   +-- magi_all_models.cpp             # Unified C++ simulation and NLL engine for Base, M005, M006
¦   ¦   +-- magi_swarm_001.cpp             # Standalone baseline C++ engine
¦   ¦   +-- magi_swarm_005.cpp             # Standalone M005 discrete-time opponent C++ engine
¦   ¦   +-- magi_swarm_006.cpp             # Standalone M006 continuous-time champion C++ engine
¦   ¦   +-- navarro_fuss_core.cpp           # Navarro-Fuss fast WFPT joint density core
¦   +-- experiments/
¦       +-- run_benchmark_local.R           # Local multi-subject CMA-ES benchmarking orchestrator
¦       +-- run_swarm_005.R                 # Parallel CMA-ES optimizer for M005
¦       +-- run_swarm_006.R                 # Parallel CMA-ES optimizer for M006
¦       +-- plot_real.py                    # Visualization script rendering figures from benchmark results
+-- legacy/                                 # Preserved archive of all exploratory iterations, stages, and reports
```

---

## ?? Champion Architecture Summary

| Model | Structural Hypothesis | Key Mechanism | Likelihood ($\mathcal{L}_{DDM}$) | Cohen\'s $d$ vs M005 |
|---|---|---|---|---|
| **$M_{base}$** | Canonical Cortical | Q-Learning drift + Static Wald boundary | $108.5 \pm 4.0$ | — |
| **$M_{005}$** | Discrete Thermodynamic | Opponent-processing + Ising prior + Albus-Marr LTD/LTP | $104.5 \pm 3.8$ | Baseline ($d = 0.711$) |
| **$M_{006}$** *(Champion)* | Continuous Physical Time | Continuous $ITI_t$ decay + Dynamic Epistemic Boundary ($a^{(t)}$) | $\mathbf{100.5 \pm 3.5}$ | **$d = 2.18$** ($p < 10^{-11}$) |

### Core Mathematical Formulations
1. **Continuous-Time Eligibility Trace**:
   $$Z_i(t) = e^{-ITI_t / \tau_{decay}} \cdot \kappa_i Z_i(t-1) + \mathbf{h}_i(t)$$
2. **Dynamic Epistemic Decision Boundary**:
   $$a^{(t)} = a_{base} + w_u \cdot U_{epistemic}^{(t)}, \quad \text{where } U_{epistemic}^{(t)} = |cb_0^{(t)}| \cdot |cb_1^{(t)}|$$
3. **Local Albus-Marr Synaptic Plasticity**:
   $$W_{PC, i}^{(t+1)} = W_{PC, i}^{(t)} + \alpha_{PC} \cdot Z_i^{(t)} \cdot \mathcal{E}_{local, c}$$

---

## ?? Execution and Reproduction

### 1. Run Local Benchmark (C++ & R)
```R
Rscript src/experiments/run_benchmark_local.R
```
This compiles `src/models/magi_all_models.cpp` via Rcpp, executes CMA-ES parameter optimization against the human empirical dataset (`data/raw/behavioral_compilate.csv`), and exports trial and subject metrics to `results/`.

### 2. Generate Visual Manifolds
```bash
python src/experiments/plot_real.py
```
This reads the outputs in `results/` and updates the figure suite in `figures/`.
