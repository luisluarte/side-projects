library(Rcpp)
library(loo)
library(doParallel)
library(foreach)

sourceCpp("src/fitting_procedures/mcmc_sampler.cpp")
sourceCpp("src/models/eccm_smooth_graph.cpp")
sourceCpp("src/fitting_procedures/extract_pointwise_ll.cpp")

dat_all <- read.csv("C:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv")
set.seed(42)
participants <- sample(unique(dat_all[['participant_id']]), 30)

iters <- 10

run_mcmc_smooth_graph <- function(iters, init_phi, resp, out, rt, delta_t) {
  chain <- matrix(0, nrow=iters, ncol=length(init_phi))
  curr_phi <- init_phi
  curr_ll <- -eval_eccm_smooth_graph(curr_phi, resp, out, rt, delta_t)
  for (i in 1:iters) {
    prop_phi <- curr_phi + rnorm(length(curr_phi), 0, 0.05)
    prop_ll <- -eval_eccm_smooth_graph(prop_phi, resp, out, rt, delta_t)
    if (runif(1) < exp(prop_ll - curr_ll)) { curr_phi <- prop_phi; curr_ll <- prop_ll }
    chain[i, ] <- curr_phi
  }
  return(chain)
}

init_phi_smooth <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.1/0.9), log(0.88/0.12), 0.5, log(1.0), log(0.5), log(2.0))
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
  
  ch_21 <- run_mcmc_smooth_graph(iters, init_phi_smooth, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  ll_21 <- extract_all_pointwise_ll(21, ch_21, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  
  ch_6 <- run_mcmc_subject(6, iters, init_phi_6, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  ll_6 <- extract_all_pointwise_ll(6, ch_6, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  
  r_eff_21 <- relative_eff(exp(ll_21), chain_id = rep(1, iters))
  loo_21 <- loo(ll_21, r_eff=r_eff_21)
  
  r_eff_6 <- relative_eff(exp(ll_6), chain_id = rep(1, iters))
  loo_6 <- loo(ll_6, r_eff=r_eff_6)
  
  res[[as.character(p)]] <- list(elpd_21 = loo_21$estimates["elpd_loo", "Estimate"], elpd_6 = loo_6$estimates["elpd_loo", "Estimate"])
}

elpd_21_total <- sum(sapply(res, function(x) x$elpd_21))
elpd_6_total <- sum(sapply(res, function(x) x$elpd_6))

sink("output_smooth_loo.txt")
cat(sprintf("Baseline (M6) elpd: %.2f\n", elpd_6_total))
cat(sprintf("Smooth Graph (M21) elpd: %.2f (Delta: %.2f)\n", elpd_21_total, elpd_21_total - elpd_6_total))
sink()
