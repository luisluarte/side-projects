# ==============================================================================
# EXACT-R: Master Script for Cross-Generalization & Manifold Analysis
# ==============================================================================
library(dplyr)

# Set working directory to project folder
if (requireNamespace("this.path", quietly = TRUE)) {
  setwd(this.path::here())
}

# Source sub-modules
source("data_generators.R")
source("evaluation_metrics.R")
source("cross_validation_generalization.R")
source("optimization_manifold.R")

cat("==============================================================================\n")
cat("Starting Advanced Cross-Generalization & Optimization Manifold Analysis\n")
cat("==============================================================================\n")

# Step 1: Run 4x4 Time-Series Cross-Generalization Benchmark
cv_results <- run_cross_generalization_benchmark(
  sweep_results_file = "optimization_sweep_results.csv",
  T_test = 800
)

# Step 2: Derive Optimization Manifold on Optimal Pre-Training Family
optimal_protocol <- cv_results$Best_Protocol
cat(sprintf("\nDeriving Optimization Manifold for Selected Optimal Family: %s\n", optimal_protocol))

manifold_results <- derive_optimization_manifold(
  sweep_file = "optimization_sweep_results.csv",
  sens_file = "sensitivity_analysis_results.csv",
  target_protocol = optimal_protocol
)

cat("\n==============================================================================\n")
cat("ADVANCED ANALYSIS COMPLETE\n")
cat("==============================================================================\n")
cat(sprintf("1. Optimal Pre-Training Family for Max Generalization: %s\n", optimal_protocol))
cat(sprintf("2. Out-of-Family Generalization Index Score        : %.4f\n", cv_results$Generalization_Index[optimal_protocol]))
cat(sprintf("3. Optimization Manifold Linear Coupling Fit (R^2)   : %.4f\n", manifold_results$Linear_Model %>% summary() %>% .$r.squared))
cat("==============================================================================\n")
