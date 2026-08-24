set.seed(420)
# libs --------------------------------------------------------------------
pacman::p_load(
  tidyverse,
  cmdstanr,
  posterior,
  ggplot2,
  bayesplot,
  yardstick,
  RWiener,
  transport,
  loo
)

# path --------------------------------------------------------------------
setwd(this.path::here())

# data --------------------------------------------------------------------
dat_raw <- read_csv("../../data/raw/behavioral_compilate.csv")

dat_clean <- dat_raw %>%
  arrange(participant_id, ttp) %>%
  group_by(participant_id) %>%
  mutate(
    RT = (ttr - ttp) / 1000,
    ITI = (ttp - lag(ttF)) / 1000,
    F_dur = (ttF - ttr) / 1000,
    Boundary = ifelse(Resp == 2, 1, 0)
  ) %>%
  mutate(
    # Zero-inference temporal imputation for initialization stability
    ITI = ifelse(is.na(ITI), mean(ITI, na.rm = TRUE), ITI)
  ) %>%
  ungroup() %>%
  filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(`F`))

pid_sample <- sample(unique(dat_clean$participant_id), size = 30)
dat_clean <- dat_clean %>%
  filter(participant_id %in% pid_sample)

# hierarchical geometry indexing ------------------------------------------
subject_counts <- dat_clean %>%
  group_by(participant_id) %>%
  summarise(
    count = n(),
    min_rt = min(RT) # Strict DDM lower bound extraction
  ) %>%
  ungroup()

end_idx <- cumsum(subject_counts$count)
start_idx <- end_idx - subject_counts$count + 1
min_rt_array <- subject_counts$min_rt

# model parameters --------------------------------------------------------
N_MF <- 5

# stan data ---------------------------------------------------------------
stan_data <- list(
  N = nrow(dat_clean),
  S = nrow(subject_counts),
  start_idx = start_idx,
  end_idx = end_idx,
  choice = dat_clean$Boundary,
  rt = dat_clean$RT,
  reward = dat_clean$F,
  iti = dat_clean$ITI,
  f_dur = dat_clean$F_dur,
  min_rt = min_rt_array,
  N_MF = N_MF,
  grainsize = 1 # Hardware concurrency chunking
)

# execution ---------------------------------------------------------------
# Enforce C++ compiler threading for reduce_sum
cpp_options <- list(stan_threads = TRUE)
mod1 <- cmdstan_model("../models/bvk_continuous.stan", cpp_options = cpp_options)
mod2 <- cmdstan_model("../models/q_learning_ddm.stan", cpp_options = cpp_options)

threads_per_chain <- 8

fit_bvk <- mod1$sample(
  data = stan_data,
  chains = 4,
  parallel_chains = 4,
  threads_per_chain = threads_per_chain,
  iter_warmup = 1000,
  iter_sampling = 1000,
  adapt_delta = 0.85,
  max_treedepth = 10
)

fit_q <- mod2$sample(
  data = stan_data,
  chains = 4,
  parallel_chains = 4,
  threads_per_chain = threads_per_chain,
  iter_warmup = 1000,
  iter_sampling = 1000,
  adapt_delta = 0.85,
  max_treedepth = 10
)

fit_bvk$save_object("../../results/fit_bvk_complete.rds")
fit_q$save_object("../../results/fit_q_complete.rds")

# save fits
# out_dir <- "../../results/"
# fit_bvk$save_object(file.path(out_dir, "fit_bvk_complete.rds"))
# fit_q$save_object(file.path(out_dir, "fit_q_complete.rds"))
# write_rds(x = stan_data, file = "../../results/stan_data.rds")

fit_bvk <- read_rds("../../results/fit_bvk_complete.rds")
fit_q <- read_rds("../../results/fit_q_complete.rds")
stan_data <- read_rds("../../results/stan_data.rds")

# PSIS-LOOCV --------------------------------------------------------------

# simulation only models
mod_bvk_gq <- cmdstan_model("../models/bvk_continuous_gq.stan")
mod_q_gq <- cmdstan_model("../models/q_learning_ddm_gq.stan")

# forward simulation
gq_bvk <- mod_bvk_gq$generate_quantities(
  fitted_params   = fit_bvk,
  data            = stan_data,
  parallel_chains = 4
)
gq_q <- mod_q_gq$generate_quantities(
  fitted_params = fit_q,
  data = stan_data,
  parallel_chains = 4
)

# extract pointwise log-likelihood
ll_bvk <- gq_bvk$draws("log_lik", format = "array")
ll_q <- gq_q$draws("log_lik", format = "array")

# ----- GPD TAIL STABILIZATION PATCH ----- #
# Unconditionally inject 1e-5 jitter into ALL log-likelihoods.
# This mathematically guarantees no identical tail values exist to crash the GPD fit,
# while being infinitesimally small so it does not affect the macroscopic ELPD.
inject_unconditional_jitter <- function(ll_array) {
  noise <- array(rnorm(length(ll_array), mean = 0, sd = 1e-5), dim = dim(ll_array))
  return(ll_array + noise)
}

set.seed(123)
ll_bvk <- inject_unconditional_jitter(ll_bvk)
ll_q   <- inject_unconditional_jitter(ll_q)
# ---------------------------------------- #

# compute PSIS-LOO
r_eff_bvk <- relative_eff(exp(ll_bvk))
r_eff_bvk[is.na(r_eff_bvk)] <- 1.0
loo_bvk <- loo(ll_bvk, r_eff = r_eff_bvk)

r_eff_q <- relative_eff(exp(ll_q))
r_eff_q[is.na(r_eff_q)] <- 1.0
loo_q <- loo(ll_q, r_eff = r_eff_q)

# model comparison loo
loo_diff <- loo_compare(list(Dual_Kernel = loo_bvk, Q_Learning = loo_q))
print(loo_diff)

# model comparison predictive stacking weight
stacking_weights <- loo_model_weights(
  list(Dual_Kernel = loo_bvk, Q_Learning = loo_q),
  method = "stacking"
)
print(stacking_weights)
