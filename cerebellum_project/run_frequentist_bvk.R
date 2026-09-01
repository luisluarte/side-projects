pacman::p_load(tidyverse, Rcpp, optimx)

Rcpp::sourceCpp("src/models/qlearning_ddm.cpp")
Rcpp::sourceCpp("src/models/eccm_bvk.cpp")

dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types = FALSE)
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>% group_by(participant_id) %>% mutate(RT = (ttr - ttp) / 1000, ITI = (ttp - lag(ttF)) / 1000, Boundary = ifelse(Resp == 2, 1, 0)) %>% mutate(ITI = ifelse(is.na(ITI), mean(ITI, na.rm = TRUE), ITI)) %>% ungroup() %>% filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(`F`))

dat_clean <- dat_clean %>% mutate(participant_idx = as.integer(as.factor(participant_id)))
S <- length(unique(dat_clean$participant_idx))
lambda <- 1.0 

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
  
  init_phi <- rep(0, 7)
  res <- optim(init_phi, obj_dyn, method="L-BFGS-B", lower=rep(-5, 7), upper=rep(5, 7))
  
  ll <- extract_ll_ql_dynamic_point(res$par, resp, out, rt)
  return(ll)
}

fit_subject_bvk <- function(s_idx) {
  d <- dat_clean %>% filter(participant_idx == s_idx)
  resp <- d$Boundary + 1
  out <- d$`F`
  rt <- d$RT
  iti <- d$ITI
  
  obj_bvk <- function(phi) {
    dev <- eval_eccm_bvk(phi, resp, out, rt, iti)
    penalty <- lambda * sum(abs(phi))
    return(dev + penalty)
  }
  
  init_phi <- rep(0, 10)
  res <- optim(init_phi, obj_bvk, method="L-BFGS-B", lower=rep(-5, 10), upper=rep(5, 10))
  
  ll <- extract_ll_eccm_bvk(res$par, resp, out, rt, iti)
  return(ll)
}

cat("Fitting Models...\n")
ll_dyn <- c()
ll_bvk <- c()
for(s in 1:S) {
  ll_dyn <- c(ll_dyn, fit_subject_dyn(s))
  ll_bvk <- c(ll_bvk, fit_subject_bvk(s))
}

N <- length(ll_dyn)
k1 <- 7 * S
k2 <- 10 * S

LR_raw <- ll_bvk - ll_dyn
LR_adj <- LR_raw - (k2 - k1)/N
t_test_res <- t.test(LR_adj, alternative="greater")

cat("\n==========================================\n")
cat("FREQUENTIST STAGE: PAIRED T-TEST ON LIKELIHOODS\n")
cat("CEREDRIFT (Full-Gating) vs Dynamic Q-Learning\n")
cat("==========================================\n")
cat("Total Observations (Trials):", N, "\n")
cat("Total Dynamic QL Parameters:", k1, "\n")
cat("Total CEREDRIFT Parameters:", k2, "\n\n")

cat("Dynamic QL Mean LL:", mean(ll_dyn), " (Sum:", sum(ll_dyn), ")\n")
cat("CEREDRIFT Mean LL: ", mean(ll_bvk), " (Sum:", sum(ll_bvk), ")\n\n")

cat("Paired T-Test Results (Adjusted for structural complexity):\n")
cat("t-statistic:", t_test_res$statistic, "\n")
cat("df:", t_test_res$parameter, "\n")
cat("p-value:", t_test_res$p.value, "\n")
cat("Mean of differences:", t_test_res$estimate, "\n")
cat("95% CI: [", t_test_res$conf.int[1], ", ", t_test_res$conf.int[2], "]\n")
