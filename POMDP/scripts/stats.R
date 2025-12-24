# ==============================================================================
# POSTERIOR CORRELATION & DIVERGENCE DIAGNOSTIC
# ==============================================================================
# Categorical Formalization: Manifold Separation and Ergodic Failure
# Logic: Identifies if a divergent chain is trapped in a local non-convexity.
# ==============================================================================

pacman::p_load(
    tidyverse, posterior, bayesplot, patchwork, scales, this.path
)

setwd(this.path::here())

# 1. LOAD FIT AND EXTRACT TARGETS ----
fit <- readRDS("../results/fit_full_volatility_final.rds")

# We target [1,1] (Naive/Baseline) as the primary diagnostic anchor.
target_idx <- "[1,1]"
vars_to_plot <- c(
    paste0("mu_beta", target_idx),
    paste0("mu_log_tau", target_idx)
)

# Preserve the 'Chain' dimension to diagnose the divergence
draws_array <- fit$draws(variables = vars_to_plot)

# Create a chain-aware data frame for diagnostics
draws_df <- as_draws_df(draws_array) %>%
    mutate(
        mu_tau = exp(.data[[paste0("mu_log_tau", target_idx)]]),
        Chain = as.factor(.chain)
    )

# 2. CHAIN-AWARE JOINT POSTERIOR (IDENTIFYING ISLANDS) ----
# Categorical Intuition: Mapping the 'Island' problem.
# If the divergent chain is a different color in a different region,
# the manifold is multi-modal or poorly identified.
p_joint <- ggplot(draws_df, aes(x = .data[[paste0("mu_beta", target_idx)]], y = mu_tau, color = Chain)) +
    geom_point(alpha = 0.2, size = 0.5) +
    geom_density_2d(alpha = 0.5) +
    scale_color_viridis_d() +
    theme_minimal() +
    labs(
        title = "Chain-Aware Joint Posterior",
        subtitle = "Check if the divergent chain (color) is isolated from the others.",
        x = "Motivation (mu_beta)",
        y = "Decision Noise (mu_tau)"
    ) +
    guides(color = guide_legend(override.aes = list(alpha = 1, size = 2)))

# 3. TRACE OVERLAP (THE PATH MORPHISM) ----
color_scheme_set("viridis")
p_trace <- mcmc_trace(
    draws_array,
    pars = vars_to_plot,
    facet_args = list(ncol = 1, strip.position = "left")
) +
    theme_minimal() +
    labs(
        title = "Chain Trajectories (Divergence Check)",
        subtitle = "A divergent chain will appear as a 'wandering' line outside the main caterpillar."
    )

# 4. MARGINAL DENSITIES BY CHAIN ----
p_dens <- mcmc_dens_overlay(draws_array, pars = vars_to_plot) +
    theme_minimal() +
    labs(title = "Overlaid Marginal Densities")

# 5. ASSEMBLY ----
final_diag <- (p_joint | (p_dens / p_trace)) +
    plot_layout(widths = c(2, 1))

# Export ----
dir.create("../results/diagnostics", showWarnings = FALSE)
ggsave("../results/diagnostics/beta_tau_divergence_check.pdf", final_diag, width = 14, height = 8)

# 6. NUMERICAL DIVERGENCE DIAGNOSTICS ----
# We extract the 'sampler_diagnostics' to see if the chain hit the treedepth limit.
sampler_diags <- fit$sampler_diagnostics()
div_counts <- apply(sampler_diags[, , "divergent__"], 2, sum)
max_td <- apply(sampler_diags[, , "treedepth__"], 2, max)

message("--- Sampler Diagnostics per Chain ---")
for (i in 1:length(div_counts)) {
    message(paste("Chain", i, "- Divergences:", div_counts[i], "| Max Treedepth:", max_td[i]))
}

# 7. CORRELATION (CLEAN CHAINS ONLY) ----
# We calculate correlation excluding the divergent chain to see the 'true' manifold.
clean_draws <- draws_df %>% filter(Chain != which.max(div_counts))
cor_val <- cor(clean_draws[[paste0("mu_beta", target_idx)]], clean_draws$mu_tau)
message(paste("\nCorrelation (excluding divergent chain):", round(cor_val, 3)))
