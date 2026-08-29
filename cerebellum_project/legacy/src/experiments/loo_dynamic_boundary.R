library(Rcpp)
library(loo)
library(doParallel)
library(foreach)

sourceCpp("src/fitting_procedures/mcmc_sampler.cpp")
sourceCpp("src/fitting_procedures/extract_pointwise_ll.cpp")

dat_all <- read.csv("C:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv")
set.seed(42)
participants <- sample(unique(dat_all[['participant_id']]), 30)

iters <- 10

# phi_10: a, t_nd, beta_v, eta_LTP, eta_LTD, w_cb, lambda_shift, gamma_v, gamma_a
init_phi_10 <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5, log(1.0), log(1.0), log(3.0))
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
  
  ch_10 <- run_mcmc_subject(10, iters, init_phi_10, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  ll_10 <- extract_all_pointwise_ll(10, ch_10, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  
  ch_6 <- run_mcmc_subject(6, iters, init_phi_6, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  ll_6 <- extract_all_pointwise_ll(6, ch_6, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  
  r_eff_10 <- relative_eff(exp(ll_10), chain_id = rep(1, iters))
  loo_10 <- loo(ll_10, r_eff=r_eff_10)
  
  r_eff_6 <- relative_eff(exp(ll_6), chain_id = rep(1, iters))
  loo_6 <- loo(ll_6, r_eff=r_eff_6)
  
  res[[as.character(p)]] <- list(elpd_10 = loo_10$estimates["elpd_loo", "Estimate"], elpd_6 = loo_6$estimates["elpd_loo", "Estimate"])
}

elpd_10_total <- sum(sapply(res, function(x) x$elpd_10))
elpd_6_total <- sum(sapply(res, function(x) x$elpd_6))

sink("output_dyn_bound.txt")
cat(sprintf("Baseline (M6) elpd: %.2f\n", elpd_6_total))
cat(sprintf("Dynamic Boundary (M10) elpd: %.2f (Delta: %.2f)\n", elpd_10_total, elpd_10_total - elpd_6_total))
sink()
