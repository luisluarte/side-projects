# libs ----
pacman::p_load(
    tidyverse,
    cmdstanr,
    posterior
)

# set source as path ----
setwd(this.path::here())

# load model ----
fit_path <- "../results/fit_optimal_final.rds"
fit <- readRDS(fit_path)

# global diagnostics ----
sampler_diags <- fit$sampler_diagnostics(
    format = "df"
)

## divergent transitions ----
total_transitions <- nrow(sampler_diags)
divergences <- sum(sampler_diags$divergent__)
div_pct <- (divergences / total_transitions) * 100
div_pct

## maximum treedepth ----
max_treedepth_hits <- sum(sampler_diags$treedepth__ >= 10)
tree_pct <- (max_treedepth_hits / total_transitions) * 100
tree_pct


## E-BFMI ----
ebfmi_vals <- fit$diagnostic_summary()$ebfmi
low_ebfmi <- sum(ebfmi_vals < 0.3)
low_ebfmi

# parameter level diagnostics ----
summ <- fit$summary()

## potential scale reduction factor ----
rhat_failures <- summ %>%
    filter(rhat > 1.01) %>%
    select(variable, rhat)
rhat_failures

## bulk effective sample size ----
ess_bulk_failures <- summ %>%
    filter(ess_bulk < 400) %>%
    select(variable, ess_bulk)
ess_bulk_failures

## tail effective sample size ----
ess_tail_failures <- summ %>%
    filter(ess_tail < 400) %>%
    select(variable, ess_tail)
ess_tail_failures

## monte carlo standard error ratio ----
mcse_failures <- summ %>%
    mutate(
        mcse_mean_est = sd / sqrt(ess_bulk),
        mcse_ratio = mcse_mean_est / sd
    ) %>%
    filter(mcse_ratio > 0.1) %>%
    select(variable, mcse_mean_est, sd, mcse_ratio)
mcse_failures
