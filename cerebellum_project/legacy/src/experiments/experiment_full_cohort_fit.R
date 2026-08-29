# Experiment: Full Cohort Bayesian Parameter Estimation
# This script runs the mass-parallel Metropolis-within-Gibbs sampler across all 128 subjects

source("src/fitting_procedures/bayesian_fitting_orchestrator.R")

dataset_path <- "C:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv"
out_csv_path <- "docs/full_cohort_results.csv"

# Execute the orchestration
# For a full run, set iters = 150. For testing, iters = 10.
results <- run_full_cohort_mcmc(dataset_path, out_csv_path, iters = 10)

cat("\n--- Model Comparison Summary ---\n")
print(t.test(results$Deviance_Intact, results$Deviance_WSLS, paired=TRUE, alternative="less"))
print(t.test(results$Deviance_Intact, results$Deviance_Lesion, paired=TRUE, alternative="less"))
