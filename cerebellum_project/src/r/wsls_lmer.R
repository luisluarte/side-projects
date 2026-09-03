local_lib <- Sys.getenv('R_LIBS_USER')
if (!dir.exists(local_lib)) dir.create(local_lib, recursive = TRUE)
.libPaths(c(local_lib, .libPaths()))

if (!require('pacman')) install.packages('pacman', lib=local_lib, repos='http://cran.us.r-project.org')
pacman::p_load(cmdstanr, dplyr, readr, lme4, lmerTest)

cat('Loading Data and Model...\n')
dat <- read_rds('data/processed/behavioral_compilate.rds')
fit_wsls <- readRDS('results/fit_wsls_spatial.rds')

cat('Extracting Log-Likelihood...\n')
log_lik_median <- fit_wsls[['summary']]('log_lik', 'median')[['median']]
dat[['log_lik']] <- log_lik_median

cat('Cleaning Data...\n')
dat_clean <- dat %>% filter(Resp %in% c(1, 2) & rt > 0)

cat('Fitting LMM...\n')
model_lmm <- lmer(log_lik ~ nt + (1 | participant_id), data = dat_clean)

cat('\n--- LMM RESULTS ---\n')
print(summary(model_lmm))
