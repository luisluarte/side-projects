pacman::p_load(tidyverse, Rcpp, optimx)
Rcpp::sourceCpp("src/models/qlearning_ddm.cpp")
dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types = FALSE)
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>% group_by(participant_id) %>% mutate(RT = (ttr - ttp) / 1000, ITI = (ttp - lag(ttF)) / 1000, Boundary = ifelse(Resp == 2, 1, 0)) %>% mutate(ITI = ifelse(is.na(ITI), mean(ITI, na.rm = TRUE), ITI)) %>% ungroup() %>% filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(`F`))
dat_clean <- dat_clean %>% mutate(participant_idx = as.integer(as.factor(participant_id)))
lambda <- 1.0
d <- dat_clean %>% filter(participant_idx == 1)
resp <- d$Boundary + 1
out <- d$`F`
rt <- d$RT
iti <- d$ITI

obj_dyn <- function(phi) {
  dev <- eval_ql_ddm_dynamic(phi, resp, out, rt, iti)
  penalty <- lambda * sum(abs(phi))
  return(dev + penalty)
}

init_phi <- c(0, 0, 0, 0, 0, 0, 0)
res <- optim(init_phi, obj_dyn, method="L-BFGS-B", lower=rep(-5, 7), upper=rep(5, 7))
print(res$par)
cat("\nFinal deviance:", res$value, "\n")
cat("\nLog Likelihood extract:\n")
ll <- extract_ll_ql_dynamic_point(res$par, resp, out, rt, iti)
print(head(ll))
print(sum(ll))
