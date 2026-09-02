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
    lib = local_lib
  )
  library("pacman", character.only = TRUE)
}
pacman::p_load(
  tidyverse,
  cmdstanr,
  posterior,
  this.path,
  loo,
  caret,
  yardstick
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

## preds -------------------------------------------------------------------
pred_matrix_vopt <- as_draws_matrix(vopt$draws("pred_sw"))
pred_matrix_m012 <- as_draws_matrix(m012$draws("pred_sw"))

# median probs
vopt_preds <- vopt$summary("pred_sw", median)
m012_preds <- m012$summary("pred_sw", median)
dat_raw$pred_sw_vopt <- vopt_preds$median
dat_raw$pred_sw_m012 <- m012_preds$median

# clean skipped trials
dat_clean <- dat_raw %>%
  mutate(
    pred_sw_vopt = ifelse(pred_sw_vopt == -1.0, NA, pred_sw_vopt),
    pred_sw_m012 = ifelse(pred_sw_m012 == -1.0, NA, pred_sw_m012)
  ) %>%
  filter(!is.na(stay_switch) &
    !is.na(pred_sw_vopt) &
    !is.na(pred_sw_m012)) %>%
  mutate(
    truth = factor(stay_switch, levels = c("switch", "stay"))
  )

prauc_vopt <- pr_auc(dat_clean, truth = truth, pred_sw_vopt)
prauc_m012 <- pr_auc(dat_clean, truth = truth, pred_sw_m012)

cat("VOPR PRAUC\n")
print(prauc_vopt)
cat("M012 PRAUC\n")
print(prauc_m012)


# stats -------------------------------------------------------------------

## LOO ---------------------------------------------------------------------
cat("COMPUTE LOO")
loo_vopt <- vopt$loo()
loo_m012 <- m012$loo()

cat("ELPD")
model_comp <- loo_compare(
  list(
    VOPT = loo_vopt,
    M012 = loo_m012
  )
)
cat("ELPD")
print(model_comp)

## confusion matrices ------------------------------------------------------
