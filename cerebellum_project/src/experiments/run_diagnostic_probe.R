pacman::p_load(tidyverse, cmdstanr, bayesplot)
color_scheme_set("darkgray")

cat("MAGI Core Diagnostic Probe Initiated...\n")
repo_root <- "."
dat_raw <- read_csv(file.path(repo_root, "data/raw/behavioral_compilate.csv"), show_col_types=FALSE)
dat_raw$Reward <- dat_raw$F
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>%
    group_by(participant_id) %>%
    dplyr::mutate(RT = (ttr-ttp)/1000, Boundary = ifelse(Resp==2, 1, 2), ITI = (ttp - lag(ttr))/1000) %>%
    ungroup() %>% filter(RT>0.1, RT<3.0, !is.na(Resp), !is.na(Reward)) %>%
    mutate(ITI = ifelse(is.na(ITI) | ITI < 0, median(ITI, na.rm=TRUE), ITI), 
           participant_idx = as.integer(as.factor(participant_id)))

# 1. Empirical Restriction (N=30)
dat_clean <- dat_clean %>% filter(participant_idx <= 30)
N_subj <- 30

min_rt_df <- dat_clean %>% group_by(participant_idx) %>% summarise(min_rt = min(RT)) %>% arrange(participant_idx)

# 2. Host-Level Anatomical Matrix Injection (N(0,1))
set.seed(42)
W_exp_matrix <- matrix(rnorm(N_subj * 32, 0, 1), nrow=N_subj, ncol=32)

stan_data <- list(
  min_rt = min_rt_df$min_rt,
  N_trials = nrow(dat_clean),
  N_subj = N_subj,
  subj = dat_clean$participant_idx,
  resp = dat_clean$Boundary,
  reward = dat_clean$Reward,
  rt = dat_clean$RT,
  iti = dat_clean$ITI,
  W_exp = W_exp_matrix
)

cat("Compiling Continuous Manifold M006 V2...\n")
mod_clamp_v2 <- cmdstan_model(file.path(repo_root, "src/stan/m006_clamped_v2.stan"))

cat("Executing 100 Warmup / 50 Sampling Probe...\n")
fit <- mod_clamp_v2$sample(data = stan_data, chains = 4, parallel_chains = 4, 
                           iter_warmup = 100, iter_sampling = 50, refresh = 25,
                           adapt_delta = 0.8)

# 3. Symplectic Diagnostic Fail-Safe
sampler_diags <- fit$sampler_diagnostics(format="df")

min_stepsize <- min(sampler_diags$stepsize__)
max_treedepth_hits <- sum(sampler_diags$treedepth__ >= 10)
divergent_trans <- sum(sampler_diags$divergent__ == 1)
total_samples <- nrow(sampler_diags)

cat(sprintf("\n--- DIAGNOSTIC READOUT ---\n"))
cat(sprintf("Min Step-Size: %.2e\n", min_stepsize))
cat(sprintf("Max Treedepth Hits: %d / %d\n", max_treedepth_hits, total_samples))
cat(sprintf("Divergent Transitions: %d (%.2f%%)\n", divergent_trans, (divergent_trans/total_samples)*100))

failed <- FALSE
if (min_stepsize < 1e-5) { cat("ALERT: Step-Size Collapse Detected (< 1e-5). Funnel geometry.\n"); failed <- TRUE }
if (max_treedepth_hits > 0.10 * total_samples) { cat("ALERT: Severe Treedepth Saturation.\n"); failed <- TRUE }
if (divergent_trans > 0.05 * total_samples) { cat("ALERT: Divergence Threshold Exceeded (> 5%).\n"); failed <- TRUE }

if (failed || TRUE) { # Forcing plot generation for MAGI Core reporting if requested, but let's strictly follow logic:
  if(failed) {
    cat("GEOMETRIC BOTTLENECK DETECTED. HALTING FIT AND GENERATING NEAL'S FUNNEL DIAGNOSTICS.\n")
    # Plot sigmas against offsets to find the funnel
    p <- mcmc_pairs(fit$draws(c("sigma[1]", "z[1,1]", "sigma[2]", "z[2,1]", "sigma[3]", "z[3,1]")), 
                    np = bayesplot::nuts_params(fit))
    ggsave(file.path(repo_root, "figures/neals_funnel_diagnostic.png"), plot = p, width=12, height=12)
    cat("Saved pairs plot to figures/neals_funnel_diagnostic.png\n")
    quit(status=1)
  } else {
    cat("All geometric diagnostics passed. The symplectic manifold is stable.\n")
    quit(status=0)
  }
}
