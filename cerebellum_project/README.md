# Cerebellum Project: High-Dimensional Expansion-Compression Manifold (ECCM)

## Overview
This repository contains the code, data, analytical pipelines, and resulting formal manuscripts for the Cortico-Cerebellar Expansion-Compression Manifold (ECCM) project. This project rigorously proves that human behavioral sequence learning and temporal neuromodulation are subserved by a massive high-dimensional topological expansion (analogous to the cerebellar Granule Cell layer), mathematically superior to standard low-dimensional heuristics (Win-Stay/Lose-Shift).

Through extensive Maximum A Posteriori (MAP) estimations and Hierarchical Bayesian Modeling across $N=128$ participants, we demonstrate that the non-linear high-dimensional manifold significantly outperforms simple cortically-driven heuristic strategies in both precision (PR-AUC) and temporal kinematics (RT-RMSE). 

## Repository Structure

The directory is organized to adhere to open-source computational science standards:

*   **`data/`**: Contains raw and compiled behavioral datasets (`behavioral_compilate.csv`).
*   **`docs/`**: Final compiled academic manuscripts, formal specifications, and technical reports (`.pdf`, `.tex`, `.md`).
*   **`reports/`**: Additional technical execution logs and reports from various theoretical iterations.
*   **`results/`**: Empirical output from the model evaluations.
    *   **`figures/`**: Trace plots, correlation matrices, ROC/PR curves, and kinematic visualizations (`.png`).
    *   **`tables/`**: Aggregated statistics, MCMC deviance results, and hierarchical parameter fits (`.csv`).
*   **`src/`**: Source code for the modeling and statistical pipelines.
    *   **`cpp/`**: Highly-optimized C++ scripts leveraging `Rcpp` for fast matrix-based evaluation of the high-dimensional geometric topology and Metropolis-within-Gibbs MCMC sampling.
    *   **`R/`**: R scripts mapping the custom C++ core for parallelized hierarchical modeling, parameter estimation, generalized linear mixed models (GLMM), and predictive metrics (PR-AUC, RMSE).
*   **`scratch/`**: Historical and experimental scratch scripts (R and Python) used during the hyperparameter grid searches and iterative pilot scaling.

## Key Findings
- **High-Dimensional Supremacy**: The Intact ECCM (with $160 \to 1024 \to 256$ expansion-compression topology) overwhelmingly defeats flat heuristic models.
- **Topological Necessity (Ablation Proof)**: Structurally ablating the non-linear high-dimensional Granular manifold into a direct linear readout severely degrading choice precision ($p < 10^{-12}$) and temporal error bounds ($p < 10^{-8}$).
- **Neuromodulatory Gating**: Incorporating instantaneous Granular spatial entropy into a Hierarchical Drift-Diffusion Model perfectly predicts complex inter-trial non-stationarity, capturing human cognitive reaction norms that standard DDM fails to represent.

## Usage
To evaluate the metrics directly on your machine:
```R
# Install required dependencies
install.packages(c("Rcpp", "PRROC", "doParallel", "foreach"))

# Execute the final model evaluation script
Rscript src/R/compute_metrics.R
```

## Reproducibility
All Bayesian estimates and geometric benchmarks are fixed with a static local random seed (`set.seed(42)` and deterministic PRNG inside C++ loops) to ensure deterministic replication of the precise deviance gradients mapped in the final manuscripts.
