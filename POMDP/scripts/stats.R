pacman::p_load(tidyverse, cmdstanr, posterior, bayestestR, ggdist)
setwd(this.path::here())


# 1. LOAD DATA & FIT ----
d <- readRDS("../data/processed/discrete_data.rds")
fit <- readRDS("../results/fit_full_volatility_final.rds")

# 2. EXTRACT EMPIRICAL WAIT TIMES ----
# Calculate the duration of contiguous 'Wait' actions in the Idle state
empirical_waits <- d %>%
    mutate(
        A_code = case_when(A == "a_W" ~ 1, TRUE ~ 0),
        S_code = case_when(S == "S_I" ~ 1, TRUE ~ 0)
    ) %>%
    filter(S_code == 1) %>% # Only look at Idle state
    group_by(ID, n_sesion) %>%
    mutate(run_id = data.table::rleid(A_code)) %>%
    group_by(ID, n_sesion, run_id) %>%
    summarise(
        is_wait = first(A_code) == 1,
        duration_steps = n(),
        .groups = "drop"
    ) %>%
    filter(is_wait) %>%
    pull(duration_steps)

# 3. SIMULATE SYNTHETIC WAIT TIMES ----
# Extract posterior means for beta and tau
draws <- fit$draws(variables = c("mu_beta", "mu_kappa"), format = "df")
mean_beta <- mean(draws$`mu_beta[1,1]`) # Baseline Beta
mean_tau <- 1 # Baseline Tau
Q_wait <- mean_beta * 0.025 # Scaled Utility
Q_poke <- 0.7 # Approx Q* for poking (Physics Engine constant)

# Calculate probability of waiting P(W) vs Paking P(P)
# P(W) = exp(Q_w/tau) / (exp(Q_w/tau) + exp(Q_p/tau))
p_wait <- exp(Q_wait / mean_tau) / (exp(Q_wait / mean_tau) + exp(Q_poke / mean_tau))

# Simulate 10,000 bouts (Geometric Distribution)
# Duration ~ Geometric(1 - p_wait)
synthetic_waits <- rgeom(10000, prob = (1 - p_wait)) + 1

# 4. VISUAL COMPARISON ----
ppc_data <- bind_rows(
    tibble(Duration = empirical_waits, Type = "Empirical Data"),
    tibble(Duration = synthetic_waits, Type = "Posterior Prediction")
)

p_ppc <- ggplot(ppc_data, aes(x = Duration, fill = Type)) +
    geom_density(alpha = 0.5) +
    scale_x_log10(labels = scales::comma) +
    theme_minimal() +
    scale_fill_manual(values = c("black", "cyan")) +
    labs(
        title = "Posterior Predictive Check: Wait Time Distribution",
        subtitle = paste0("Model P(Wait) = ", round(p_wait, 3)),
        x = "Wait Duration (Steps of 25ms)",
        y = "Density (Log Scale)"
    )
p_ppc

ggsave("../results/figures/ppc_wait_times.png", p_ppc, width = 8, height = 6)

pacman::p_load(tidyverse, data.table, survival, survminer, fitdistrplus)

wait_bouts <- d %>%
    mutate(
        # Identify the target state-action pair
        is_idle_wait = (S == "S_I" & A == "a_W")
    ) %>%
    # Group by session to ensure runs don't bleed across boundaries
    group_by(ID, n_sesion) %>%
    mutate(
        # Unique ID for each contiguous run
        run_id = rleid(is_idle_wait)
    ) %>%
    group_by(ID, n_sesion, run_id) %>%
    summarise(
        is_target_bout = first(is_idle_wait),
        duration_steps = n(),
        .groups = "drop"
    ) %>%
    filter(is_target_bout) %>%
    mutate(
        duration_sec = duration_steps * 0.025 # Convert 25ms steps to seconds
    )

# 3. VISUAL DIAGNOSTIC: LOG-SURVIVAL PLOT ----
# Logic: If Beta is constant, P(Wait > t) = exp(-lambda * t).
# Therefore, log(P(Wait > t)) should be a STRAIGHT LINE with slope -lambda.
# Curvature indicates that Beta changes as the animal waits.

# Fit Kaplan-Meier survival curve
fit_km <- survfit(Surv(duration_sec) ~ 1, data = wait_bouts)

p_check <- ggsurvplot(
    fit_km,
    fun = "log", # Log-Transformation of Survival Probability
    conf.int = TRUE,
    ggtheme = theme_minimal()
)
p_check

# Save the plot
dir.create("../results/diagnostics", showWarnings = FALSE)
ggsave("../results/diagnostics/wait_time_stationarity.png", print(p_check$plot), width = 8, height = 6)


# 4. STATISTICAL MODEL COMPARISON (AIC) ----
# We fit three candidate distributions to the empirical data.
# Note: Sub-sampling 20,000 points for speed if dataset is massive.
set.seed(123)
sample_data <- sample(wait_bouts$duration_sec, min(nrow(wait_bouts), 20000))
# Ensure non-zero for Log-Normal fitting
sample_data <- sample_data[sample_data > 0]

message("Fitting distributions to empirical wait times...")
fit_exp <- fitdist(sample_data, "exp") # Constant Beta
fit_weibull <- fitdist(sample_data, "weibull") # Beta changes (Power Law)
fit_lnorm <- fitdist(sample_data, "lnorm") # Beta changes (Multiplicative)

# Compare Goodness-of-Fit
aic_results <- tibble(
    Model = c("Exponential (Constant Beta)", "Weibull (Dynamic Beta)", "Log-Normal (Dynamic Beta)"),
    AIC = c(fit_exp$aic, fit_weibull$aic, fit_lnorm$aic)
) %>%
    arrange(AIC)

cat("\n--- DISTRIBUTION FIT COMPARISON ---\n")
print(aic_results)

# Interpretation Helper
best_model <- aic_results$Model[1]
cat("\nVERDICT: The data is best described by the", best_model, ".\n")
if (best_model != "Exponential (Constant Beta)") {
    cat("This confirms that Beta is NON-STATIONARY. The animal's probability of re-engaging\n")
    cat("changes the longer it waits (likely due to inertia or satiety recovery).\n")
} else {
    cat("This supports the current model. Beta appears to be effectively constant.\n")
}
