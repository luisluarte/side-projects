# Cerebellum Project: High-Dimensional Expansion-Compression Manifold (ECCM) & MAGI Generative Protocol

## Overview

This repository contains the code, data, analytical pipelines, and resulting formal manuscripts for the Cortico-Cerebellar Expansion-Compression Manifold (ECCM) project. This project rigorously proves that human behavioral sequence learning and temporal neuromodulation are subserved by a massive high-dimensional topological expansion (analogous to the cerebellar Granule Cell layer), mathematically superior to standard low-dimensional heuristics (Win-Stay/Lose-Shift).

The latest phase of this project (The MAGI Protocol) successfully modeled the generative geometry of human reaction times. By decoupling pure cognitive diffusion from a scale-free ($1/f$) cerebellar memory manifold, the **Terminal Decoupled Hybrid Architecture (Variant 11.2)** shattered theoretical barriers, proving that the Cerebello-Thalamo-Cortical loop acts as a dynamic multiplicative gain controller on cognitive evidence accumulation.

## Repository Structure

The directory is organized to adhere to open-source computational science standards:

- **`data/`**: Contains raw and compiled behavioral datasets (e.g., `behavioral_compilate.csv`).
- **`docs/`**: Final compiled academic manuscripts, formal specifications, and technical reports (`.pdf`, `.tex`, `.md`).
  - *Includes the formal mathematical specification for the Terminal Decoupled Hybrid model.*
- **`reports/`**: Additional technical execution logs and reports from various theoretical iterations.
- **`results/`**: Empirical output from the model evaluations.
  - **`figures/`**: Trace plots, correlation matrices, ROC/PR curves, kinematic visualizations, and final Generative Q-Q Identity plots (`.png`).
  - **`tables/`**: Aggregated statistics, MCMC deviance results, hierarchical parameter fits, and terminal MAGI generative metrics (`.csv`).
- **`src/`**: Source code for the modeling and statistical pipelines.
  - **`models/`**: (MAGI Core) Highly-optimized C++ scripts detailing the topological evolution of the Likelihood-Free Inference (LFI) architectures (Epoch 1 to Epoch 10.2).
  - **`cpp/`**: General C++ scripts leveraging `Rcpp` for fast matrix-based evaluation of the high-dimensional geometric topology and Metropolis-within-Gibbs MCMC sampling.
  - **`R/`**: R scripts mapping the custom C++ core for parallelized hierarchical modeling, parameter estimation, generalized linear mixed models (GLMM), and predictive metrics (PR-AUC, RMSE).
- **`scratch/`**: Historical and experimental scratch scripts (R and Python) used during the hyperparameter grid searches and iterative pilot scaling.
- **`Root Directory Scripts`**: High-level execution runners (e.g., `run_terminal_full_evaluation.R`, `generate_identity_plot.R`) used to rapidly deploy CMA-ES / Nelder-Mead matrices across the C++ models.

## Key Findings (MAGI Terminal Phase)

- **Generative Geometry (**$\mathcal{W}_1$): The Decoupled Hybrid achieved $\mathcal{W}_1 = 0.192$, shattering the rigid geometrical limits of pure stationary diffusion (Baseline $\mathcal{W}_1 = 0.298$).
- **Scale-Free Fatigue Sequence (**$\beta$): By modulating the continuous cognitive drift rate via a sparse multiplicative fractional projection, the architecture perfectly captured macroscopic physiological fatigue ($\beta = 0.484$).
- **Rademacher Generalization**: Pruning the Purkinje readout via $L_1$ Subgradient LASSO ensured strict structural parsimony, fully surviving Rademacher capacity gates ($p_{Rad} = 0.958$).

## Usage

To evaluate the final generative metrics directly on your machine:

``` r
# Install required dependencies
install.packages(c("tidyverse", "Rcpp", "cmaes", "lme4", "lmerTest", "patchwork"))

# Execute the MAGI terminal evaluation script
Rscript run_terminal_full_evaluation.R
```

## Reproducibility

All Bayesian estimates and geometric benchmarks are fixed with static local random seeds (`set.seed(42)` and deterministic PRNG inside C++ loops) to ensure deterministic replication of the precise deviance gradients mapped in the final manuscripts and generative models.
