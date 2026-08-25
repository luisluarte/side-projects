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
  return(list(ll = as.numeric(ll_mat[1,]), phi = phi_hat))
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
  return(list(ll = ll, phi = phi_hat))
}

cat("Fitting Q-Learning Baseline...\n")
ll_ql <- c()
for(s in 1:S) {
  cat(s, " ")
  ll_ql <- c(ll_ql, fit_subject_ql(s)$ll)
}
cat("\nFitting Dynamic Q-Learning (prev_RT)...\n")
ll_dyn <- c()
for(s in 1:S) {
  cat(s, " ")
  ll_dyn <- c(ll_dyn, fit_subject_dyn(s)$ll)
}
cat("\n")

N <- length(ll_ql)
k1 <- 5 * S
k2 <- 7 * S

LR_pointwise <- ll_dyn - ll_ql
sum_LR_adj <- sum(LR_pointwise) - (k2 - k1)
LR_pointwise_adj <- LR_pointwise - (k2 - k1)/N

mean_LR_adj <- mean(LR_pointwise_adj)
sd_LR_adj <- sd(LR_pointwise_adj)
Z <- (mean_LR_adj * sqrt(N)) / sd_LR_adj

cat("==========================================\n")
cat("FREQUENTIST VUONG CLOSENESS TEST\n")
cat("==========================================\n")
cat("Total N (trials):", N, "\n")
cat("Penalty (k2 - k1):", k2 - k1, "\n")
cat("Raw Sum LL QL:  ", sum(ll_ql), "\n")
cat("Raw Sum LL Dyn: ", sum(ll_dyn), "\n")
cat("Vuong Z-statistic (AIC adjusted):", Z, "\n")

if (Z > 1.96) {
    cat(">>> SUCCESS: Dynamic Q-Learning significantly beats Baseline (Z > 1.96).\n")
} else {
    cat(">>> FAILURE: Dynamic Q-Learning does not beat Baseline (Z <= 1.96).\n")
}
