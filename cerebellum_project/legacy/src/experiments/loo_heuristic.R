library(Rcpp)
library(loo)
library(doParallel)
library(foreach)

sourceCpp("src/fitting_procedures/mcmc_sampler.cpp")
sourceCpp("src/models/wsls.cpp")
sourceCpp("src/models/qlearning_ddm.cpp")
sourceCpp("src/fitting_procedures/extract_pointwise_ll.cpp")

dat_all <- read.csv("C:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv")
set.seed(42)
participants <- sample(unique(dat_all[['participant_id']]), 30)

iters <- 10
init_wsls <- c(log(2.0), log(0.3/0.7), log(3.0))
init_ql <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5))
init_phi_6 <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5, log(1.0), log(0.5))

run_mcmc_ql <- function(iters, init_phi, resp, out, rt) {
  chain <- matrix(0, nrow=iters, ncol=length(init_phi))
  curr_phi <- init_phi
  curr_ll <- eval_ql_ddm(curr_phi, resp, out, rt)
  for (i in 1:iters) {
    prop_phi <- curr_phi + rnorm(length(curr_phi), 0, 0.05)
    prop_ll <- eval_ql_ddm(prop_phi, resp, out, rt)
    if (runif(1) < exp(curr_ll - prop_ll)) { curr_phi <- prop_phi; curr_ll <- prop_ll }
    chain[i, ] <- curr_phi
  }
  return(chain)
}
run_mcmc_wsls <- function(iters, init_phi, resp, out, rt) {
  chain <- matrix(0, nrow=iters, ncol=length(init_phi))
  curr_phi <- init_phi
  curr_ll <- eval_wsls(curr_phi, resp, out, rt)
  for (i in 1:iters) {
    prop_phi <- curr_phi + rnorm(length(curr_phi), 0, 0.05)
    prop_ll <- eval_wsls(prop_phi, resp, out, rt)
    if (runif(1) < exp(curr_ll - prop_ll)) { curr_phi <- prop_phi; curr_ll <- prop_ll }
    chain[i, ] <- curr_phi
  }
  return(chain)
}

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
  
  ch_6 <- run_mcmc_subject(6, iters, init_phi_6, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  ch_wsls <- run_mcmc_wsls(iters, init_wsls, p_data$Resp, p_data$F, p_data$RT)
  ch_ql <- run_mcmc_ql(iters, init_ql, p_data$Resp, p_data$F, p_data$RT)
  
  ll_6 <- extract_all_pointwise_ll(6, ch_6, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  ll_wsls <- extract_ll_wsls_ql(ch_wsls, ch_ql, p_data$Resp, p_data$F, p_data$RT)
  ll_ql <- extract_ll_ql(ch_ql, p_data$Resp, p_data$F, p_data$RT)
  
  res[[as.character(p)]] <- list(ll_6=ll_6, ll_wsls=ll_wsls, ll_ql=ll_ql)
}

res <- res[!sapply(res, is.null)]

loo_6 <- 0; loo_wsls <- 0; loo_ql <- 0
for (r in res) {
  loo_6 <- loo_6 + suppressWarnings(loo(r$ll_6))$estimates["elpd_loo", "Estimate"]
  loo_wsls <- loo_wsls + suppressWarnings(loo(r$ll_wsls))$estimates["elpd_loo", "Estimate"]
  loo_ql <- loo_ql + suppressWarnings(loo(r$ll_ql))$estimates["elpd_loo", "Estimate"]
}

cat(sprintf("Baseline (M6) elpd: %.2f\n", loo_6))
cat(sprintf("WSLS (DDM) elpd: %.2f (Delta: %.2f)\n", loo_wsls, loo_wsls - loo_6))
cat(sprintf("Q-Learning (DDM) elpd: %.2f (Delta: %.2f)\n", loo_ql, loo_ql - loo_6))
