# libs ----
pacman::p_load(
    tidyverse,
    cmdstanr,
    posterior,
    tidybayes,
    ggdist,
    patchwork,
    GGally
)

# set source as path ----
setwd(this.path::here())

# model description ----

## cognitive architecture ----

# the model is a hierarchical partially observable markov decision process (POMDP)
# this model captures how animals infer hidden environmental state to make decisions
# in a setting of motivated behavior (lickometer), and how TCS alters this cofnitive process

# belief updating and entropy
# the model assumes that animals do not know the true "experimental context", that is,
# the reward probability of the spouts is unknown. However, they do maintain a "belief state",
# which is simply a probability vector over possible contexts (low, mid, high entropy).
# bayesian updating: after receiving a reward or non-reward, the animal updates its belief using bayes' rule

# learning rates eta_pos eta_neg: we fit separate parameters for how much weight an animal gives
# to positive feedback (licking is rewarded) vs. negative feedback (no reward)

# belief diffusion: between sessions, beliefs decay slightly toward a uniform distribution,
# which is how the model represent not knowing the context or "forgetting" the context

# entropy (H): computed form the belief state, high entropy means the animal is confused/exploring;
# low entropy means the animal is confident/exploiting

## value function ----

# the probability of taking an action is proportional to its value (Q), evaluated
# via softmax function.
# Q-values are constructed from two main components:
# extrinsic values: the dot product of the current belief state and the optimal policy values (Q*).
# Q* is pre-computed via value iteration based on the objective mechanics of the lickometer.
# intrinsic value:
# kappa (information seeking): scales the entropy H. Determines if the animal is driven
# to lick when uncertain.
# phi (persistence): a bonus added if the animal repeats its last action
# side (spatial bias): inherent preference for spout1 over spout2
# beta and beta_slope (impulsivity/waiting): the baseline value of waiting, and how that
# value decays the longer the animal has waited.

## hierarchical structure ---

# to separate the acute drug effects from underlying animal traits, parameters
# are structures hierarchically:
# population level (mu): the base effect of a cognitive context, plus the acute delta
# effect of TCS.
# trait level (animal random effects): stable, animal-specific offsets. modeled
# via non-centered parametrization (z-scores * sigma_trat) to prevent divergences
# state level (session random effects): fluctuations for a specific animal a specific day
# Flattened into a global 1D vector to prevent out-of-bounds indexing error and optimize CPU cache
# during HMC sampling

## computational optimizations ----

# reduce-sum: log-likelihood is parallelized across animals
# vectorized session: session are indexed globally
# entropy memoization: entropy is only recomputed when the belief state actually changes
# that is, after feedback

# 1. LOAD MODEL & DATA ---------------------------------------------------------
message("Loading posterior samples...")
fit <- readRDS("../results/fit_optimal_final_v2.rds")

# 2. HMC HEALTH DIAGNOSTICS ----------------------------------------------------
message("\n--- HMC Diagnostics ---")
diag_summary <- fit$diagnostic_summary()
print(diag_summary)

# Check for R-hat (Convergence) and ESS (Effective Sample Size)
fit_summary <- fit$summary()
bad_rhat <- fit_summary %>% filter(rhat > 1.05)
if (nrow(bad_rhat) > 0) {
    warning("Some parameters have R-hat > 1.05. Model may not have fully converged.")
    print(bad_rhat %>% select(variable, rhat, ess_bulk))
} else {
    message("All R-hat values look excellent (< 1.05).")
}

# 3. TRAIT CORRELATION ANALYSIS (The "Funnel/Ridge" Check) ---------------------
# 3. TRAIT CORRELATION ANALYSIS (The "Funnel/Ridge" Check) ---------------------
message("\n--- Analyzing Baseline Trait Correlations ---")

# Extract the individual animal trait offsets (medians)
traits_df <- fit$summary(
    variables = c("r_animal_beta", "r_animal_kappa", "r_animal_phi"),
    median
) %>%
    mutate(
        # Extract EXACTLY what is inside the brackets (e.g., "1,1" or "1")
        # This guarantees every single row has a unique identifier
        unique_index = str_extract(variable, "(?<=\\[).*(?=\\])"),

        # Extract the parameter name
        param = str_extract(variable, "(beta|kappa|phi)")
    ) %>%
    # Keep only the columns we need to spread
    select(unique_index, param, median) %>%
    # Pivot wider (modern equivalent of spread)
    pivot_wider(names_from = param, values_from = median)

print("Median Baseline Traits per Index:")
print(traits_df)

# Plot the correlation matrix (ignoring the index column for the math)
p_corr <- ggpairs(
    traits_df %>% select(-unique_index),
    title = "Posterior Correlation of Baseline Traits"
)
print(p_corr)
ggsave("../results/trait_correlations.png", p_corr, width = 6, height = 6)

# 4. DRUG EFFECT HYPOTHESIS TESTING --------------------------------------------
message("\n--- Drug vs. Vehicle Deltas (Within-Context) ---")

# The parameters we care about: the explicit differences in each context
delta_params <- c(
    "drug_delta_beta", "drug_delta_kappa", "drug_delta_phi",
    "drug_delta_side", "drug_delta_beta_slope",
    "drug_delta_eta_pos", "drug_delta_eta_neg"
)

# Extract draws
draws_df <- as_draws_df(fit$draws(variables = delta_params))

# Calculate Statistics: Median, 95% HDI, and Probability of Direction (Pd)
# Pd = the percentage of the posterior distribution that has the same sign as the median
results_df <- draws_df %>%
    select(-.chain, -.iteration, -.draw) %>%
    pivot_longer(everything(), names_to = "variable", values_to = "value") %>%
    mutate(
        # Parse the variable name and context index
        param = str_remove(variable, "drug_delta_"),
        param = str_remove(param, "\\[\\d+\\]"),
        context_id = as.integer(str_extract(variable, "\\d+"))
    ) %>%
    group_by(param, context_id) %>%
    summarise(
        Median = median(value),
        Lower_95 = quantile(value, 0.025),
        Upper_95 = quantile(value, 0.975),
        # Bayesian P-value equivalent (Probability the effect is > 0)
        Prob_Greater_Zero = mean(value > 0),
        .groups = "drop"
    ) %>%
    arrange(context_id, param)

print("Target Hypothesis Tests (Drug Deltas):")
print(results_df, n = Inf)

write_csv(results_df, "../results/drug_delta_statistics.csv")

# 5. VISUALIZATION OF DRUG DELTAS ----------------------------------------------
# Forest plot for the three contexts
p_forest <- results_df %>%
    mutate(Context = paste("Context", context_id)) %>%
    ggplot(aes(x = Median, y = param, color = Context)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "black", alpha = 0.6) +
    geom_pointrange(aes(xmin = Lower_95, xmax = Upper_95), position = position_dodge(width = 0.5)) +
    facet_wrap(~Context, ncol = 1) +
    ggpubr::theme_pubr() +
    theme(legend.position = "none")
p_forest

print(p_forest)
ggsave("../results/drug_deltas_forest.png", p_forest, width = 8, height = 10)

message("\nAnalysis complete. Results and plots saved to ../results/")

# absolute values ----
# 1. Extract the pre-calculated absolute values directly from the mu_ matrices
absolute_draws <- fit %>%
    gather_draws(
        mu_phi[drug_id, ctx_id],
        mu_side[drug_id, ctx_id],
        mu_eta_pos[drug_id, ctx_id],
        mu_eta_neg[drug_id, ctx_id],
        mu_beta_slope[drug_id, ctx_id]
    ) %>%
    # Keep only Vehicle (ID 2) and Active Drug (ID 3)
    filter(drug_id %in% c(2, 3)) %>%
    mutate(
        Condition = ifelse(drug_id == 2, "Vehicle", "TCS"),
        # Clean up the variable names for the plot (e.g., "mu_phi" -> "phi")
        parameter = str_replace(.variable, "mu_", "")
    )

# 2. Calculate Summary Statistics
absolute_summary <- absolute_draws %>%
    group_by(parameter, Condition, ctx_id) %>%
    summarise(
        Median = median(.value),
        Lower_95 = quantile(.value, 0.025),
        Upper_95 = quantile(.value, 0.975),
        .groups = "drop"
    )
write_csv(x = absolute_summary, file = "../results/parameter_summary.csv")

# 3. Visualization: Side-by-Side Posterior Densities
p_absolute <- absolute_draws %>%
    ggplot(aes(x = Condition, y = .value, color = Condition, fill = Condition)) +
    stat_halfeye(
        alpha = 0.7,
        .width = c(0.89, 0.95),
        point_interval = median_hdci,
        position = position_dodge(width = 0.6)
    ) +
    ggpubr::theme_pubr() +
    facet_wrap(~ ctx_id * parameter, scales = "free")
p_absolute
