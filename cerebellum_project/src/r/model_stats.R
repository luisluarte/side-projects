# setup -------------------------------------------------------------------
local_lib <- Sys.getenv("R_LIBS_USER")

if (!dir.exists(local_lib)) {
  dir.create(local_lib, recursive = TRUE)
}
.libPaths(c(local_lib, .libPaths()))

# libs --------------------------------------------------------------------
cat("LIBS\n")
if (!require("pacman", character.only = TRUE)) {
  install.packages("pacman",
                   lib = local_lib)
  library("pacman", character.only = TRUE)
}
pacman::p_load(
  tidyverse,
  cmdstanr,
  posterior,
  this.path,
  loo,
  caret
)

if (!dir.exists(cmdstan_path())) {
  install_cmdstan()
} else {
  message("cmdstan installed at: ", cmdstan_path())
}

setwd(here())

# data --------------------------------------------------------------------
cat("LOAD DATA")
dat_raw <- read_rds("../../data/processed/behavioral_compilate.rds")
cat("LOADING FITTED MODELS")
vopt <- read_rds("../../results/fit_vopt.rds")
m012 <- read_rds("../../results/fit_m012.rds")


## extract parameters ------------------------------------------------------
draws_vopt <- as_draws_df(vopt$draws())
draws_m012 <- as_draws_df(m012$draws())
get_median <- function(fit, prefix, n_subj) {
  unlist(sapply(1:n_subj, function(i) median(fit[[paste0(prefix, "[", i, "]")]])))
}
N_subj <- length(unique(dat_raw$participant_id))

# V-OPT Parameters
v_abase   <- get_median(draws_vopt, "a_base_raw", N_subj)
v_vctx    <- get_median(draws_vopt, "v_ctx", N_subj)
v_wbias   <- get_median(draws_vopt, "w_bias_raw", N_subj)
v_aw      <- get_median(draws_vopt, "aw", N_subj)
v_al      <- get_median(draws_vopt, "al", N_subj)
v_wctx    <- get_median(draws_vopt, "w_ctx", N_subj)
v_betamis <- get_median(draws_vopt, "beta_mismatch", N_subj)
# M012 Parameters
m_abase   <- get_median(draws_m012, "a_base_raw", N_subj)
m_vctx    <- get_median(draws_m012, "v_ctx", N_subj)
m_wbias   <- get_median(draws_m012, "w_bias_raw", N_subj)
m_aw      <- get_median(draws_m012, "aw", N_subj)
m_al      <- get_median(draws_m012, "al", N_subj)
m_tau     <- get_median(draws_m012, "tau", N_subj) # or tau_decay depending on what you named it
m_alphaPC <- get_median(draws_m012, "alpha_pc", N_subj)
m_wctx    <- get_median(draws_m012, "w_ctx", N_subj)
m_wcb     <- get_median(draws_m012, "w_cb", N_subj)
m_betamis <- get_median(draws_m012, "beta_mismatch", N_subj)
m_g       <- get_median(draws_m012, "golgi_scale", N_subj)

# M012 Constants
m_frac     <- 0.1 + 0.8 * (0:3 / 3.0)
m_inv_frac <- 1.0 - m_frac
m_kappa    <- 0.1 + 0.89 * (0:3 / 3.0)







# stats -------------------------------------------------------------------

## LOO ---------------------------------------------------------------------
cat("COMPUTE LOO")
loo_vopt <- vopt$loo()
loo_m012 <- m012$loo()

print(loo_vopt)
print(loo_m012)

cat("ELPD")
model_comp <- loo_compare(
  list(
    VOPT = loo_vopt,
    M012 = loo_m012
  )
)
print(model_comp)

## confusion matrices ------------------------------------------------------
true_labels <- factor(
  ifelse(
    dat_raw$stay_switch == 1, "switch", "stay"
  ),
  levels = c("stay", "switch")
)









