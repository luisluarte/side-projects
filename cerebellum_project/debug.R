pacman::p_load(tidyverse, Rcpp, optimx)
Rcpp::sourceCpp("src/models/qlearning_ddm.cpp")
Rcpp::sourceCpp("src/models/eccm_bvk_full_gating_dist_fatigue_tnd.cpp")
dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types = FALSE)
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>% group_by(participant_id) %>% mutate(RT = (ttr - ttp) / 1000, ITI = (ttp - lag(ttF)) / 1000, F_dur = (ttF - ttr) / 1000, Boundary = ifelse(Resp == 2, 1, 0)) %>% mutate(ITI = ifelse(is.na(ITI), mean(ITI, na.rm = TRUE), ITI), F_dur = ifelse(is.na(F_dur), 0.5, F_dur)) %>% ungroup() %>% filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(`F`))
dat_clean <- dat_clean %>% mutate(participant_idx = as.integer(as.factor(participant_id)))
d <- dat_clean %>% filter(participant_idx == 1)
resp <- d$Boundary + 1
out <- d$`F`
rt <- d$RT
iti <- d$ITI
f_dur <- d$F_dur
best_N <- 20
tryCatch({
    res = eval_ql_ddm_dynamic_poly_tnd(rep(0, 11), resp, out, rt)
    print(res)
}, error = function(e) print(e))
tryCatch({
    res2 = eval_bvk_full_gating_dist_fatigue_tnd(rep(0, 14), resp, out, rt, iti, f_dur, best_N)
    print(res2)
}, error = function(e) print(e))
