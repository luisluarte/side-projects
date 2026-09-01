# Thermodynamic Cortico-Cerebellar Cognitive Architecture

This repository contains the mathematical formulation, C++ core engines, R optimization orchestrators, and benchmark visual proof suites for the bio-thermodynamic cortico-cerebellar architectures developed to model joint choice-RT dynamics and cognitive switching under the Navarro-Fuss (2009) Wiener First-Passage Time (WFPT) density.

---

## 🏗️ Repository Structure

`
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
¦   +-- stan/
¦   ¦   +-- Baseline_Optimized_GQ.stan      # Mathematically optimal, numerically bounded Q-learning baseline
¦   ¦   +-- m012_candidate.stan             # Final Continuous-Time Cerebellar Reservoir Architecture
¦   +-- experiments/
¦       +-- run_benchmark_local.R           # Local multi-subject CMA-ES benchmarking orchestrator
¦       +-- run_swarm_005.R                 # Parallel CMA-ES optimizer for M005
¦       +-- run_swarm_006.R                 # Parallel CMA-ES optimizer for M006
¦       +-- plot_real.py                    # Visualization script rendering figures from benchmark results
+-- legacy/                                 # Preserved archive of all exploratory iterations, stages, and reports
`

---

## 🏆 Champion Architecture Summary (M012 HMC Proof)

While earlier heuristic optimizers (CMA-ES) favored M006, full Bayesian Hamiltonian Monte Carlo (HMC) sampling revealed **M012** to be the definitive bio-thermodynamic champion. M012 utilizes a 4-node continuous time-decaying reservoir to dynamically track trial-by-trial predictive certainty and injects it into a generalized geometric boundary.

| Model | Structural Hypothesis | Key Mechanism | ELPD (LOO-CV) | Diff to Baseline |
|---|---|---|---|---|
| **V-OPT** | Canonical Cortical | Q-Learning drift + Softmax bounded boundary | -3330.6 (SE: 78.5) | — |
| **M012** | Continuous Reservoir | Continuous ITI decay + Cerebellar Epistemic Boundary | **-3058.9 (SE: 79.7)** | **Δ = +271.7 (SE: 23.4)** |

### Core Mathematical Formulations
1. **Continuous-Time Eligibility Trace**:
   Z_i(t) = exp(-ITI_t / tau_decay) * kappa_i * Z_i(t-1) + h_i(t)
2. **Dynamic Epistemic Decision Boundary**:
   a(t) = a_base + w_u * U_epistemic(t), where U_epistemic(t) = |cb_0(t)| * |cb_1(t)|
3. **Local Albus-Marr Synaptic Plasticity**:
   W_PC_i(t+1) = W_PC_i(t) + alpha_PC * Z_i(t) * E_local_c

---

## 🚀 Execution and Reproduction

### 1. Run the Definitive M012 vs Baseline Comparison (Stan HMC)
`ash
Rscript run_final_showdown_vopt.R
`
This script runs the fully rigorous Hamiltonian Monte Carlo parameter estimation and Pareto-Smoothed Importance Sampling (PSIS) LOO cross-validation. It fits both models independently from uninformative zero initializations, calculates their exact likelihood geometries, and produces the final \loo_compare()\ matrix to prove M012's absolute dominance.

### 2. Generate Final Posterior Predictive Statistics (ROC/PR-AUC)
`ash
Rscript eval_m012_metrics.R
`
Evaluates the full posterior median estimates and mathematically derives the RT-RMSE, Matthew's Correlation Coefficient (MCC), and strictly proper normalized confusion matrices for both M012 and V-OPT.

### 3. Run Local Benchmark (C++ & R - Legacy CMA-ES)
`ash
Rscript src/experiments/run_benchmark_local.R
`
Executes CMA-ES parameter optimization against the human empirical dataset and exports trial and subject metrics.

### 4. Generate Visual Manifolds
`ash
python src/experiments/plot_real.py
`
Reads the outputs in \esults/\ and updates the figure suite in \igures/\.
