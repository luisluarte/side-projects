library(Rcpp)
library(loo)

sourceCpp("src/fitting_procedures/mcmc_sampler.cpp")
sourceCpp("src/fitting_procedures/extract_pointwise_ll.cpp")

dat_all <- read.csv("C:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv")
set.seed(42)
participants <- sample(unique(dat_all[['participant_id']]), 30)

iters <- 10

# phi_23: a, t_nd, beta_v, eta_LTP, eta_LTD, w_cb, lambda_shift, gamma_suppress, theta_cb, kappa
init_phi_23 <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5, log(1.0), log(0.5), 0.0, log(0.5))
init_phi_6 <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5, log(1.0), log(0.5))

res <- list()
for (p in participants) {
  p_data <- dat_all[dat_all$participant_id == p, ]
  p_data <- p_data[order(p_data$ttp), ]
  p_data$RT <- (as.numeric(p_data$ttr) - as.numeric(p_data$ttp)) / 1000.0
  p_data$delta_t <- c(NA, diff(as.numeric(p_data$ttp)) / 1000.0) 
  p_data$delta_t[is.na(p_data$delta_t)] <- 2.0
  valid_idx <- which(!is.na(p_data$RT) & p_data$RT >= 0.1 & p_data$RT <= 3.0 & p_data$Resp %in% c(1, 2))
  p_data <- p_data[valid_idx, ]
  if (nrow(p_data) < 20) next
  
  ch_23 <- run_mcmc_subject(23, iters, init_phi_23, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  ll_23 <- extract_all_pointwise_ll(23, ch_23, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  
  ch_6 <- run_mcmc_subject(6, iters, init_phi_6, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  ll_6 <- extract_all_pointwise_ll(6, ch_6, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  
  r_eff_23 <- relative_eff(exp(ll_23), chain_id = rep(1, iters))
  loo_23 <- loo(ll_23, r_eff=r_eff_23)
  
  r_eff_6 <- relative_eff(exp(ll_6), chain_id = rep(1, iters))
  loo_6 <- loo(ll_6, r_eff=r_eff_6)
  
  res[[as.character(p)]] <- list(elpd_23 = loo_23$estimates["elpd_loo", "Estimate"], elpd_6 = loo_6$estimates["elpd_loo", "Estimate"])
  cat(sprintf("Participant %s done: BVK=%.2f, ECCM=%.2f\n", p, loo_23$estimates["elpd_loo", "Estimate"], loo_6$estimates["elpd_loo", "Estimate"]))
}

elpd_23_total <- sum(sapply(res, function(x) x$elpd_23))
elpd_6_total <- sum(sapply(res, function(x) x$elpd_6))

cat(sprintf("\n=== FINAL RESULTS ===\n"))
cat(sprintf("Baseline (M6) elpd: %.2f\n", elpd_6_total))
cat(sprintf("BVK (M23) elpd: %.2f (Delta: %.2f)\n", elpd_23_total, elpd_23_total - elpd_6_total))
