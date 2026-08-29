pacman::p_load(tidyverse, Rcpp, cmaes, parallel)

# Set directories
repo_root <- ifelse(file.exists("src/models/magi_all_models.cpp"), ".", "../..")
cpp_path <- normalizePath(file.path(repo_root, "src/models/magi_all_models.cpp"))
data_path <- normalizePath(file.path(repo_root, "data/raw/behavioral_compilate.csv"))
results_dir <- file.path(repo_root, "results")
dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)

cat("Loading and preparing dataset...\n")
dat_raw <- read_csv(data_path, show_col_types=FALSE)
dat_raw$Reward <- dat_raw$F
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>%
    group_by(participant_id) %>%
    dplyr::mutate(RT = (ttr-ttp)/1000, 
                  Boundary = ifelse(Resp==2, 1, 0), 
                  ITI = (ttp - lag(ttr))/1000) %>%
    ungroup() %>% 
    filter(RT>0.1, RT<3.0, !is.na(Resp), !is.na(Reward)) %>%
    mutate(ITI = ifelse(is.na(ITI) | ITI < 0, median(ITI, na.rm=TRUE), ITI),
           participant_idx = as.integer(as.factor(participant_id)))

S <- max(dat_clean$participant_idx)
d_list <- split(dat_clean, dat_clean$participant_idx)

# Set hyperparameter vector for M006: [lambda_sparse, beta_ising, K_sa]
hyper <- c(4.64e-4, 5e-4, 18)

calc_expected_rt <- function(a, v, tnd) {
    sapply(1:length(v), function(i) {
        av <- abs(v[i])
        ai <- a[i]
        if (av < 1e-3) {
            tnd + (ai^2) / 2.0
        } else {
            tnd + (ai / av) * tanh(ai * av / 2.0)
        }
    })
}

num_cores <- min(parallel::detectCores() - 2, 12)
cl <- parallel::makeCluster(num_cores)

clusterExport(cl, c("cpp_path", "hyper", "d_list", "calc_expected_rt"), envir=environment())
clusterEvalQ(cl, {
    library(Rcpp)
    library(cmaes)
    Rcpp::sourceCpp(cpp_path)
})

fit_subject <- function(s_idx) {
    d <- d_list[[s_idx]]
    
    # 1. BASELINE FIT
    obj_base <- function(p) { 
        v <- get_nll_base(p, d$Boundary+1, d$Reward, d$RT)
        if(is.nan(v) || is.infinite(v)) 1e6 else v 
    }
    # Initial Base: a=1.5, tnd=0.25, v=1.5, alpha=0.5
    init_b <- c(log(1.5), log(0.25/(0.8-0.25)), log(1.5), log(0.5/(1-0.5)))
    res_b <- tryCatch(
        cma_es(init_b, obj_base, control=list(maxit=150, sigma=0.5)), 
        error=function(e) list(par=init_b, value=NA)
    )
    
    # 2. M006 FIT (BOUNDED & SA ACTIVE)
    obj_006 <- function(p) { 
        v <- get_nll_006(p, hyper, d$Boundary+1, d$Reward, d$RT, d$ITI)
        if(is.nan(v) || is.infinite(v)) 1e6 else v 
    }
    # a=1.5, tnd=0.25, v=1.5, alpha_ctx=0.5, alpha_pc=0.5, gamma=1.5, temp=1.0, tau=1.0, w_u=1.0
    init_6 <- c(log(1.5), log(0.25/(0.8-0.25)), log(1.5), log(0.5/(1-0.5)), log(0.5/(1-0.5)), log(1.5), log(1.0), log(1.0), log(1.0))
    res_6 <- tryCatch(
        cma_es(init_6, obj_006, control=list(maxit=200, sigma=0.5)), 
        error=function(e) list(par=init_6, value=NA)
    )
    
    sim_b <- sim_base(res_b$par, d$Boundary+1, d$Reward, d$RT)
    rt_pred_b <- calc_expected_rt(sim_b$a, sim_b$v, sim_b$tnd)
    prob_b <- 1 / (1 + exp(-sim_b$a * sim_b$v))
    
    sim_6 <- sim_006(res_6$par, hyper, d$Boundary+1, d$Reward, d$RT, d$ITI)
    rt_pred_6 <- calc_expected_rt(sim_6$a, sim_6$v, sim_6$tnd)
    prob_6 <- 1 / (1 + exp(-sim_6$a * sim_6$v))
    
    p6 <- res_6$par
    m006_params <- list(
        a_base = exp(p6[1]),
        tnd = 0.8 / (1.0 + exp(-p6[2])),
        v_ctx = exp(p6[3]),
        alpha_ctx = 1.0 / (1.0 + exp(-p6[4])),
        alpha_pc = 1.0 / (1.0 + exp(-p6[5])),
        gamma = exp(p6[6]),
        lambda_sa_temp = exp(p6[7]),
        tau_decay = exp(p6[8]),
        w_u = exp(p6[9])
    )
    
    list(
        s_idx = s_idx,
        pid = d$participant_id[1],
        n_trials = nrow(d),
        nll_base = res_b$value,
        nll_006 = res_6$value,
        rt_rmse_base = sqrt(mean((d$RT - rt_pred_b)^2, na.rm=TRUE)),
        rt_rmse_006 = sqrt(mean((d$RT - rt_pred_6)^2, na.rm=TRUE)),
        brier_base = mean((d$Boundary - prob_b)^2, na.rm=TRUE),
        brier_006 = mean((d$Boundary - prob_6)^2, na.rm=TRUE),
        m006_params = m006_params
    )
}

cat("Executing TRUE optimized cohort estimation (maxit=200, warm-started)...\n")
t_start <- Sys.time()
cohort_fits <- parallel::parLapply(cl, 1:S, fit_subject)
t_end <- Sys.time()
parallel::stopCluster(cl)

df_comp <- data.frame(
    SubjectID = sapply(cohort_fits, function(x) x$s_idx),
    ParticipantID = sapply(cohort_fits, function(x) x$pid),
    NTrials = sapply(cohort_fits, function(x) x$n_trials),
    NLL_Base = sapply(cohort_fits, function(x) x$nll_base),
    NLL_M006 = sapply(cohort_fits, function(x) x$nll_006),
    Delta_NLL = sapply(cohort_fits, function(x) x$nll_base - x$nll_006),
    RT_RMSE_Base = sapply(cohort_fits, function(x) x$rt_rmse_base),
    RT_RMSE_M006 = sapply(cohort_fits, function(x) x$rt_rmse_006),
    Brier_Base = sapply(cohort_fits, function(x) x$brier_base),
    Brier_M006 = sapply(cohort_fits, function(x) x$brier_006)
) %>%
mutate(
    AIC_Base = 2 * 4 + 2 * NLL_Base,
    AIC_M006 = 2 * 9 + 2 * NLL_M006,
    BIC_Base = log(NTrials) * 4 + 2 * NLL_Base,
    BIC_M006 = log(NTrials) * 9 + 2 * NLL_M006,
    Delta_AIC = AIC_Base - AIC_M006,
    Delta_BIC = BIC_Base - BIC_M006
)

df_params <- data.frame(
    SubjectID = sapply(cohort_fits, function(x) x$s_idx),
    ParticipantID = sapply(cohort_fits, function(x) x$pid),
    a_base = sapply(cohort_fits, function(x) x$m006_params$a_base),
    tnd = sapply(cohort_fits, function(x) x$m006_params$tnd),
    v_ctx = sapply(cohort_fits, function(x) x$m006_params$v_ctx),
    alpha_ctx = sapply(cohort_fits, function(x) x$m006_params$alpha_ctx),
    alpha_pc = sapply(cohort_fits, function(x) x$m006_params$alpha_pc),
    gamma = sapply(cohort_fits, function(x) x$m006_params$gamma),
    lambda_sa_temp = sapply(cohort_fits, function(x) x$m006_params$lambda_sa_temp),
    tau_decay = sapply(cohort_fits, function(x) x$m006_params$tau_decay),
    w_u = sapply(cohort_fits, function(x) x$m006_params$w_u)
)

write_csv(df_comp, file.path(results_dir, "cohort_comparison_metrics.csv"))
write_csv(df_params, file.path(results_dir, "m006_parameter_distributions.csv"))

wilcox_res <- wilcox.test(df_comp$NLL_M006, df_comp$NLL_Base, paired=TRUE, alternative="less")
t_nll <- t.test(df_comp$NLL_Base, df_comp$NLL_M006, paired=TRUE)
cohen_d <- mean(df_comp$Delta_NLL) / sd(df_comp$Delta_NLL)

cat(sprintf("Likelihood Dominance: M006 vs Base (Paired Wilcoxon V = %.1f, p = %.4e)\n", wilcox_res$statistic, wilcox_res$p.value))
cat(sprintf("Mean NLL Baseline: %.2f\n", mean(df_comp$NLL_Base)))
cat(sprintf("Mean NLL M006: %.2f\n", mean(df_comp$NLL_M006)))
cat(sprintf("Mean Delta NLL: %.2f (Cohen's d = %.3f, t = %.2f, p = %.4e)\n", mean(df_comp$Delta_NLL), cohen_d, t_nll$statistic, t_nll$p.value))
cat(sprintf("Subjects Favored: %d / %d (%.1f%%)\n", sum(df_comp$Delta_NLL > 0), S, 100 * mean(df_comp$Delta_NLL > 0)))

