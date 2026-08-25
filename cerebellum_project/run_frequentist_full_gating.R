pacman::p_load(tidyverse, Rcpp, optimx)

Rcpp::sourceCpp("src/models/qlearning_ddm.cpp")
Rcpp::sourceCpp("src/models/eccm_bvk_full_gating.cpp")

dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types = FALSE)
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>% group_by(participant_id) %>% mutate(RT = (ttr - ttp) / 1000, ITI = (ttp - lag(ttF)) / 1000, F_dur = (ttF - ttr) / 1000, Boundary = ifelse(Resp == 2, 1, 0)) %>% mutate(ITI = ifelse(is.na(ITI), mean(ITI, na.rm = TRUE), ITI), F_dur = ifelse(is.na(F_dur), 0.5, F_dur)) %>% ungroup() %>% filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(`F`))

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

fit_subject_fg <- function(s_idx) {
  d <- dat_clean %>% filter(participant_idx == s_idx)
  resp <- d$Boundary + 1
  out <- d$`F`
  rt <- d$RT
  iti <- d$ITI
  f_dur <- d$F_dur
  
  obj_fg <- function(phi) {
    dev <- eval_bvk_full_gating(phi, resp, out, rt, iti, f_dur)
    penalty <- lambda * sum(abs(phi))
    return(dev + penalty)
  }
  
  init_phi <- rep(0, 11)
  res <- optim(init_phi, obj_fg, method="L-BFGS-B", lower=rep(-5, 11), upper=rep(5, 11))
  
  ll <- extract_ll_bvk_full_gating(res$par, resp, out, rt, iti, f_dur)
  return(ll)
}

cat("Fitting Models...\n")
ll_dyn <- c()
ll_fg <- c()
for(s in 1:S) {
  ll_dyn <- c(ll_dyn, fit_subject_dyn(s))
  ll_fg <- c(ll_fg, fit_subject_fg(s))
}

N <- length(ll_dyn)
k1 <- 7 * S
k2 <- 11 * S

LR_raw <- ll_fg - ll_dyn
LR_adj <- LR_raw - (k2 - k1)/N
t_test_res <- t.test(LR_adj, alternative="greater")

cat("\n==========================================\n")
cat("FREQUENTIST STAGE: PAIRED T-TEST ON LIKELIHOODS\n")
cat("Full-Gating CEREDRIFT vs Dynamic Q-Learning\n")
cat("==========================================\n")
cat("Total Observations (Trials):", N, "\n")
cat("Total Dynamic QL Parameters:", k1, "\n")
cat("Total Full-Gating Parameters:", k2, "\n\n")

cat("Dynamic QL Mean LL: ", mean(ll_dyn), " (Sum:", sum(ll_dyn), ")\n")
cat("Full-Gating Mean LL:", mean(ll_fg), " (Sum:", sum(ll_fg), ")\n\n")

cat("Paired T-Test Results (Adjusted for structural complexity):\n")
cat("t-statistic:", t_test_res$statistic, "\n")
cat("df:", t_test_res$parameter, "\n")
cat("p-value:", t_test_res$p.value, "\n")
cat("Mean of differences:", t_test_res$estimate, "\n")
cat("95% CI: [", t_test_res$conf.int[1], ", ", t_test_res$conf.int[2], "]\n")
