pacman::p_load(tidyverse, Rcpp, optimx)

Rcpp::sourceCpp("src/models/qlearning_ddm.cpp")

dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types = FALSE)
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>% group_by(participant_id) %>% mutate(RT = (ttr - ttp) / 1000, ITI = (ttp - lag(ttF)) / 1000, Boundary = ifelse(Resp == 2, 1, 0)) %>% mutate(ITI = ifelse(is.na(ITI), mean(ITI, na.rm = TRUE), ITI)) %>% ungroup() %>% filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(`F`))

dat_clean <- dat_clean %>% mutate(participant_idx = as.integer(as.factor(participant_id)))
S <- length(unique(dat_clean$participant_idx))
lambda <- 1.0 

fit_subject_ql <- function(s_idx) {
  d <- dat_clean %>% filter(participant_idx == s_idx)
  resp <- d$Boundary + 1
  out <- d$`F`
  rt <- d$RT
  
  obj_ql <- function(phi) {
    dev <- eval_ql_ddm(phi, resp, out, rt)
    penalty <- lambda * sum(abs(phi))
    return(dev + penalty)
  }
  
  init_phi <- c(0, 0, 0, 0, 0)
  res <- optim(init_phi, obj_ql, method="L-BFGS-B", lower=rep(-5, 5), upper=rep(5, 5))
  
  phi_hat <- res$par
  ll_mat <- extract_ll_ql(matrix(phi_hat, nrow=1), resp, out, rt)
  return(as.numeric(ll_mat[1,]))
}

fit_subject_dyn <- function(s_idx) {
  d <- dat_clean %>% filter(participant_idx == s_idx)
  resp <- d$Boundary + 1
  out <- d$`F`
  rt <- d$RT
  
  obj_dyn <- function(phi) {
    dev <- eval_ql_ddm_dynamic(phi, resp, out, rt)
    penalty <- lambda * sum(abs(phi))
    return(dev + penalty)
  }
  
  init_phi <- c(0, 0, 0, 0, 0, 0, 0)
  res <- optim(init_phi, obj_dyn, method="L-BFGS-B", lower=rep(-5, 7), upper=rep(5, 7))
  
  phi_hat <- res$par
  ll <- extract_ll_ql_dynamic_point(phi_hat, resp, out, rt)
  return(ll)
}

cat("Fitting Models...\n")
ll_ql <- c()
ll_dyn <- c()
for(s in 1:S) {
  ll_ql <- c(ll_ql, fit_subject_ql(s))
  ll_dyn <- c(ll_dyn, fit_subject_dyn(s))
}

# The user requested a paired t-test on the pointwise log likelihoods.
# We apply the structural penalty correction first.
N <- length(ll_ql)
k1 <- 5 * S
k2 <- 7 * S

# Raw Likelihoods
LR_raw <- ll_dyn - ll_ql
# AIC Adjusted Likelihoods
LR_adj <- LR_raw - (k2 - k1)/N

# Paired T-Test (equivalent to one-sample t-test on differences)
t_test_res <- t.test(LR_adj, alternative="greater")

cat("\n==========================================\n")
cat("FREQUENTIST STAGE: PAIRED T-TEST ON LIKELIHOODS\n")
cat("==========================================\n")
cat("Total Observations (Trials):", N, "\n")
cat("Total Static QL Parameters:", k1, "\n")
cat("Total Dynamic QL Parameters:", k2, "\n\n")

cat("Static QL Mean LL: ", mean(ll_ql), " (Sum:", sum(ll_ql), ")\n")
cat("Dynamic QL Mean LL:", mean(ll_dyn), " (Sum:", sum(ll_dyn), ")\n\n")

cat("Paired T-Test Results (Adjusted for structural complexity):\n")
cat("t-statistic:", t_test_res$statistic, "\n")
cat("df:", t_test_res$parameter, "\n")
cat("p-value:", t_test_res$p.value, "\n")
cat("Mean of differences:", t_test_res$estimate, "\n")
cat("95% CI: [", t_test_res$conf.int[1], ", ", t_test_res$conf.int[2], "]\n")
