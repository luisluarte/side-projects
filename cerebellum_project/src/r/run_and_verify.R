library(dplyr)
library(yardstick)
library(loo)
library(readr)

cat('Running Verification Script...\n')

dat_clean <- read_rds('../../data/processed/behavioral_sample.rds')
fit_vopt <- read_rds('../../results/fit_vopt.rds')
fit_m012 <- read_rds('../../results/fit_m012.rds')

pred_sw_vopt <- fit_vopt$summary('pred_sw', 'median') %>% pull(median)
pred_sw_m012 <- fit_m012$summary('pred_sw', 'median') %>% pull(median)

dat_clean$pred_sw_vopt <- pred_sw_vopt
dat_clean$pred_sw_m012 <- pred_sw_m012

dat_clean <- dat_clean %>%
  filter(!is.na(pred_sw_vopt), !is.na(pred_sw_m012)) %>%
  mutate(truth = factor(stay_switch, levels = c('switch', 'stay')))

prauc_vopt <- pr_auc(dat_clean, truth = truth, pred_sw_vopt)$.estimate
prauc_m012 <- pr_auc(dat_clean, truth = truth, pred_sw_m012)$.estimate

loo_vopt <- fit_vopt$loo()
loo_m012 <- fit_m012$loo()
comp <- loo_compare(loo_vopt, loo_m012)

cat('VERIFICATION_START\n')
cat('VOPT_PRAUC:', prauc_vopt, '\n')
cat('M012_PRAUC:', prauc_m012, '\n')
print(comp)
cat('VERIFICATION_END\n')
