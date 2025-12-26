# ==============================================================================
# CONVERGENCE DIAGNOSTICS: GEOMETRIC AND STATISTICAL AUDIT
# ==============================================================================
# Categorical Formalization: Verification of the Stationary Limit Object
# Logic: Checks if the path morphisms from all 4 chains have converged to a
# single, well-behaved posterior manifold.
# ==============================================================================

pacman::p_load(
    tidyverse, cmdstanr, posterior, bayesplot, patchwork, scales, this.path
)

setwd(this.path::here())

# 1. LOAD FIT OBJECT ----
# retrieves the limit object of the hamiltonian flow.
fit <- readRDS("../results/fit_full_volatility_final.rds")

# 2. SAMPLER DIAGNOSTICS (E-BFMI, Divergences, Treedepth) ----
# Categorical Logic: Measuring the 'Efficiency Functor'
# E-BFMI > 0.3 confirms that energy transitions are faithful to the manifold.
sampler_diags <- fit$sampler_diagnostics()

calc_ebfmi <- function(energy_vec) {
    sum(diff(energy_vec)^2) / (length(energy_vec) * var(energy_vec))
}

ebfmi_vals <- apply(sampler_diags[, , "energy__"], 2, calc_ebfmi)
div_counts <- apply(sampler_diags[, , "divergent__"], 2, sum)
max_td <- apply(sampler_diags[, , "treedepth__"], 2, max)

message("--- NUTS Sampler Audit ---")
for (i in 1:length(ebfmi_vals)) {
    message(paste0(
        "Chain ", i,
        " | E-BFMI: ", round(ebfmi_vals[i], 3),
        " | Div: ", div_counts[i],
        " | Max TD: ", max_td[i]
    ))
}

# 3. GLOBAL CONVERGENCE STATS (R-hat and ESS) ----
# Formalization: The R-hat represents the Rank-Isomorphism of the chains.
# ESS (Effective Sample Size) measures the informational resolution of the sample.

fit_summary <- fit$summary(
    variables = c("mu_beta", "mu_kappa", "mu_log_tau", "sigma_beta_trait", "sigma_beta_session"),
    .rhat = posterior::rhat,
    .ess_bulk = posterior::ess_bulk,
    .ess_tail = posterior::ess_tail
)

cat("\n--- Convergence Summary (Cognitive Parameters) ---\n")
print(fit_summary)

# 4. CHAIN-WISE DISCREPANCY CHECK (FOR R-HAT > 1.1) ----
# Categorical Logic: Identifying the 'Island' occupied by each Chain Morphism.
# If R-hat is high, this table reveals which chain is the outlier.
cat("\n--- Chain-Wise Parameter Medians (Identifying the 'Island') ---\n")
target_vars <- c("mu_beta[1,1]", "mu_log_tau[1,1]")
draws_all <- fit$draws(variables = target_vars)

chain_medians <- draws_all %>%
    summarise_draws(median, .chain = TRUE) %>%
    select(chain, variable, median) %>%
    pivot_wider(names_from = variable, values_from = median)

print(chain_medians)

# Logic: Automatic Identification of Parameters that fail the 'limit object' criteria.
problematic_params <- fit$summary() %>%
    filter(rhat > 1.05 | ess_bulk < 400) %>%
    arrange(desc(rhat))

if (nrow(problematic_params) > 0) {
    cat("\n--- WARNING: Parameters failing convergence criteria ---\n")
    cat("A high R-hat (e.g., > 1.1) means the chains are NOT in the same mode.\n")
    print(head(problematic_params, 20))
} else {
    cat("\nSUCCESS: All parameters satisfy R-hat < 1.05 and ESS > 400.\n")
}

# 5. VISUALIZATION: THE FUZZY CATERPILLAR TEST ----
# Categorical Intuition: Chains should be indistinguishable fibers.
draws_array <- fit$draws(variables = target_vars)

p_trace <- mcmc_trace(draws_array) +
    theme_minimal() +
    labs(title = "Chain Overlap (The Caterpillar Test)")

# 6. VISUALIZATION: POSTERIOR UNIFICATION ----
p_dens <- mcmc_dens_overlay(draws_array) +
    theme_minimal() +
    labs(title = "Unified Marginal Posterior")

# 7. JOINT MANIFOLD: BETA-TAU IDENTIFIABILITY ----
draws_df <- as_draws_df(draws_array) %>%
    mutate(mu_tau = exp(`mu_log_tau[1,1]`))

p_joint <- ggplot(draws_df, aes(x = `mu_beta[1,1]`, y = mu_tau)) +
    geom_bin2d(bins = 40) +
    scale_fill_viridis_c(option = "plasma") +
    geom_density_2d(color = "white", alpha = 0.3) +
    theme_minimal() +
    labs(
        title = "Joint Posterior: Vigor vs. Noise",
        x = "Motivation (mu_beta)", y = "Decision Noise (mu_tau)"
    )

# 8. ASSEMBLY AND EXPORT ----
diag_plot <- (p_trace / p_dens) | p_joint
ggsave("../results/diagnostics/final_convergence_audit.pdf", diag_plot, width = 14, height = 10)

message("\nDiagnostics exported to ../results/diagnostics/final_convergence_audit.pdf")
