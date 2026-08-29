pacman::p_load(tidyverse, Rcpp, cmaes)

# Locate repository root
repo_root <- ifelse(file.exists("src/models/magi_all_models.cpp"), ".", "../..")
cpp_path <- file.path(repo_root, "src/models/magi_all_models.cpp")
data_path <- file.path(repo_root, "data/raw/behavioral_compilate.csv")
results_dir <- file.path(repo_root, "results")
dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)

Rcpp::sourceCpp(cpp_path)

cat("Loading empirical data from:", data_path, "\n")
dat_raw <- read_csv(data_path, show_col_types=FALSE)
dat_raw$Reward <- dat_raw$F
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>%
    group_by(participant_id) %>%
    dplyr::mutate(RT = (ttr-ttp)/1000, Boundary = ifelse(Resp==2, 1, 0), ITI = (ttp - lag(ttr))/1000) %>%
    ungroup() %>% filter(RT>0.1, RT<3.0, !is.na(Resp), !is.na(Reward)) %>%
    mutate(ITI = ifelse(is.na(ITI) | ITI < 0, median(ITI, na.rm=TRUE), ITI),
           participant_idx = as.integer(as.factor(participant_id)))

# Benchmark subset of subjects for fast local evaluation
S <- 12
d_list <- split(dat_clean, dat_clean$participant_idx)[1:S]
hyper <- c(4.64e-4, 5e-4, 18)

cat(sprintf("Optimizing across %d subjects (Baseline, M005, M006)...\n", S))
res_list <- list()
trial_list <- list()

for (s_idx in 1:S) {
    d <- d_list[[s_idx]]
    cat(sprintf("Processing Subject %d/%d...\n", s_idx, S))
    
    # ---------------- BASE ----------------
    obj_base <- function(p) { v <- get_nll_base(p, d$Boundary+1, d$Reward, d$RT); if(is.nan(v)) 1e6 else v }
    res_b <- tryCatch(cma_es(rep(0,4), obj_base, control=list(maxit=15, sigma=0.5)), error=function(e) list(par=rep(0,4), value=NA))
    
    # ---------------- 005 ----------------
    obj_005 <- function(p) { v <- get_nll_005(p, hyper, d$Boundary+1, d$Reward, d$RT); if(is.nan(v)) 1e6 else v }
    res_5 <- tryCatch(cma_es(rep(0,7), obj_005, control=list(maxit=15, sigma=0.5)), error=function(e) list(par=rep(0,7), value=NA))
    
    # ---------------- 006 ----------------
    obj_006 <- function(p) { v <- get_nll_006(p, hyper, d$Boundary+1, d$Reward, d$RT, d$ITI); if(is.nan(v)) 1e6 else v }
    res_6 <- tryCatch(cma_es(rep(0,9), obj_006, control=list(maxit=15, sigma=0.5)), error=function(e) list(par=rep(0,9), value=NA))
    
    if(!is.na(res_b$value)) {
        sim_b <- sim_base(res_b$par, d$Boundary+1, d$Reward, d$RT)
        rt_pred_b <- sim_b$tnd + (sim_b$a / ifelse(abs(sim_b$v)<1e-3, 1e-3, abs(sim_b$v))) * tanh(sim_b$a * abs(sim_b$v) / 2)
        prob_b <- 1 / (1 + exp(-sim_b$a * sim_b$v))
        res_list[[length(res_list)+1]] <- data.frame(
            SubjectID=s_idx, Model="M_base", NLL=res_b$value,
            RT_RMSE = sqrt(mean((d$RT - rt_pred_b)^2)),
            Brier = mean((d$Boundary - prob_b)^2)
        )
    }
    
    if(!is.na(res_5$value)) {
        sim_5 <- sim_005(res_5$par, hyper, d$Boundary+1, d$Reward, d$RT)
        rt_pred_5 <- sim_5$tnd + (sim_5$a / ifelse(abs(sim_5$v)<1e-3, 1e-3, abs(sim_5$v))) * tanh(sim_5$a * abs(sim_5$v) / 2)
        prob_5 <- 1 / (1 + exp(-sim_5$a * sim_5$v))
        res_list[[length(res_list)+1]] <- data.frame(
            SubjectID=s_idx, Model="M_005", NLL=res_5$value,
            RT_RMSE = sqrt(mean((d$RT - rt_pred_5)^2)),
            Brier = mean((d$Boundary - prob_5)^2)
        )
    }
    
    if(!is.na(res_6$value)) {
        sim_6 <- sim_006(res_6$par, hyper, d$Boundary+1, d$Reward, d$RT, d$ITI)
        rt_pred_6 <- sim_6$tnd + (sim_6$a / ifelse(abs(sim_6$v)<1e-3, 1e-3, abs(sim_6$v))) * tanh(sim_6$a * abs(sim_6$v) / 2)
        prob_6 <- 1 / (1 + exp(-sim_6$a * sim_6$v))
        res_list[[length(res_list)+1]] <- data.frame(
            SubjectID=s_idx, Model="M_006", NLL=res_6$value,
            RT_RMSE = sqrt(mean((d$RT - rt_pred_6)^2)),
            Brier = mean((d$Boundary - prob_6)^2)
        )
        
        trial_list[[length(trial_list)+1]] <- data.frame(
            SubjectID=s_idx, Trial=1:nrow(d), ITI=d$ITI,
            Empirical_RT=d$RT, Predicted_RT=rt_pred_6,
            Boundary=sim_6$a, Conflict=sim_6$conflict,
            Pred_RT_005 = rt_pred_5
        )
    }
}

df_subj <- do.call(rbind, res_list)
write_csv(df_subj, file.path(results_dir, "subject_metrics_real.csv"))

df_trial <- do.call(rbind, trial_list)
write_csv(df_trial, file.path(results_dir, "trial_metrics_real_006.csv"))

cat("Optimization and simulation results saved to results/\n")
