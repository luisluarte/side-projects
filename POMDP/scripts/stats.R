# ==============================================================================
# GATING-LOSS INTERACTION TEST: OREXIN ANTAGONISM (FTCS)
# ==============================================================================
# Categorical Logic: Measuring the 'Curvature' of the Drug Effect.
# Interaction (I) = [Drug_High - Veh_High] - [Drug_Base - Veh_Base]
# ==============================================================================

pacman::p_load(tidyverse, cmdstanr, posterior, bayesplot, patchwork, scales, ggdist, bayestestR)

# 1. LOAD DATA ----
fit <- readRDS("../results/fit_full_volatility_final.rds")

# 2. EXTRACT DRAWS ----
# Target: mu_kappa[drug, context]
draws_df <- as_draws_df(fit$draws(variables = "mu_kappa"))

# 3. CALCULATE INTERACTION MORPHISM ----
# Drug 1 = Vehicle, Drug 2 = FTCS
# Context 1 = Baseline, Context 3 = High Uncertainty
interaction_df <- draws_df %>%
    mutate(
        # Drug effects (Morphisms) per context
        delta_base = `mu_kappa[2,1]` - `mu_kappa[1,1]`, # Usually negative
        delta_high = `mu_kappa[2,3]` - `mu_kappa[1,3]`, # Usually positive

        # The Interaction: Difference of Differences
        # This measures the "swing" or "gating loss"
        interaction_effect = delta_high - delta_base
    )

# 4. STATISTICAL INFERENCE FOR GATING ----
# Prior for the interaction based on mu_kappa ~ N(0.1, 0.5)
# Var(Interaction) = Var(d23) + Var(d13) + Var(d21) + Var(d11) = 4 * 0.5^2 = 1.0
# SD = 1.0
prior_interaction <- distribution_normal(nrow(interaction_df), mean = 0, sd = 1.0)

summary_stats <- interaction_df %>%
    summarise(
        median_interaction = median(interaction_effect),
        q05 = quantile(interaction_effect, 0.05),
        q95 = quantile(interaction_effect, 0.95),
        pd = max(sum(interaction_effect > 0), sum(interaction_effect < 0)) / n(),
        BF10 = as.numeric(bayesfactor_parameters(interaction_effect, prior = prior_interaction, null = 0))
    )

print("--- Gating-Loss Interaction Analysis (High vs Baseline) ---")
print(summary_stats)

# 5. VISUALIZATION OF THE CROSSOVER ----
# We plot the two effects side-by-side and the interaction distribution
p1 <- interaction_df %>%
    select(delta_base, delta_high) %>%
    pivot_longer(everything(), names_to = "Context", values_to = "Effect") %>%
    mutate(Context = factor(Context,
        levels = c("delta_base", "delta_high"),
        labels = c("Baseline Context", "High Uncertainty")
    )) %>%
    ggplot(aes(x = Effect, y = Context, fill = Context)) +
    geom_vline(xintercept = 0, linetype = "dashed") +
    stat_halfeye() +
    scale_fill_manual(values = c("#440154", "#fde725")) +
    theme_minimal() +
    labs(title = "Context-Dependent Drug Effects", x = "Effect Size (FTCS - Vehicle)")

p2 <- ggplot(interaction_df, aes(x = interaction_effect)) +
    geom_vline(xintercept = 0, linetype = "dotted") +
    stat_halfeye(fill = "gray70", .width = c(0.89, 0.95)) +
    theme_minimal() +
    labs(
        title = "Interaction Morphism (Gating Test)",
        subtitle = paste0("Evidence for Gating Loss (BF10): ", round(summary_stats$BF10, 2)),
        x = "Interaction Magnitude (ΔHigh - ΔBase)"
    )

final_plot <- p1 / p2 + plot_annotation(title = "Orexin Gating of Information Seeking")
ggsave("../results/orexin_gating_test.png", final_plot, width = 10, height = 8)
