pacman::p_load(tidyverse, Rcpp, cmaes, parallel)

repo_root <- "."
cpp_path <- normalizePath(file.path(repo_root, "src/models/magi_all_models.cpp"))
data_path <- normalizePath(file.path(repo_root, "data/raw/behavioral_compilate.csv"))

dat_raw <- read_csv(data_path, show_col_types=FALSE)
dat_raw$Reward <- dat_raw$F
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>%
    group_by(participant_id) %>%
    dplyr::mutate(RT = (ttr-ttp)/1000, Boundary = ifelse(Resp==2, 1, 0)) %>%
    ungroup() %>% filter(RT>0.1, RT<3.0, !is.na(Resp), !is.na(Reward)) %>%
    mutate(participant_idx = as.integer(as.factor(participant_id)))

S <- max(dat_clean$participant_idx)
d_list <- split(dat_clean, dat_clean$participant_idx)

cl <- parallel::makeCluster(12)
clusterExport(cl, c("cpp_path", "d_list"), envir=environment())
clusterEvalQ(cl, { library(Rcpp); library(cmaes); Rcpp::sourceCpp(cpp_path) })

fits <- parallel::parLapply(cl, 1:S, function(s_idx) {
    d <- d_list[[s_idx]]
    obj_base <- function(p) { 
        v <- get_nll_base(p, d$Boundary+1, d$Reward, d$RT)
        if(is.nan(v) || is.infinite(v)) 1e6 else v 
    }
    init_b <- c(log(1.5), log(0.25/(0.8-0.25)), log(1.5), log(0.5/(1-0.5)))
    res_b <- tryCatch(cma_es(init_b, obj_base, control=list(maxit=150, sigma=0.5)), error=function(e) list(par=init_b))
    p <- res_b$par
    list(s_idx=s_idx, pid=d$participant_id[1], a_base=exp(p[1]), tnd=0.8/(1+exp(-p[2])), v_ctx=exp(p[3]), alpha_ctx=1/(1+exp(-p[4])))
})
parallel::stopCluster(cl)

df_params <- data.frame(
    SubjectID = sapply(fits, function(x) x$s_idx),
    ParticipantID = sapply(fits, function(x) x$pid),
    a_base = sapply(fits, function(x) x$a_base),
    tnd = sapply(fits, function(x) x$tnd),
    v_ctx = sapply(fits, function(x) x$v_ctx),
    alpha_ctx = sapply(fits, function(x) x$alpha_ctx)
)
write_csv(df_params, "results/base_parameter_distributions.csv")
