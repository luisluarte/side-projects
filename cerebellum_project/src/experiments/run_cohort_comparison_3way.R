pacman::p_load(tidyverse, Rcpp, cmaes, parallel)

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

hyper <- c(4.64e-4, 5e-4, 18)

num_cores <- min(parallel::detectCores() - 2, 12)
cl <- parallel::makeCluster(num_cores)

clusterExport(cl, c("cpp_path", "hyper", "d_list"), envir=environment())
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
    init_b <- c(log(1.5), log(0.25/(0.8-0.25)), log(1.5), log(0.5/(1-0.5)))
    res_b <- tryCatch(
        cma_es(init_b, obj_base, control=list(maxit=150, sigma=0.5)), 
        error=function(e) list(par=init_b, value=NA)
    )
    
    init_6 <- c(log(1.5), log(0.25/(0.8-0.25)), log(1.5), log(0.5/(1-0.5)), log(0.5/(1-0.5)), log(1.5), log(1.0), log(1.0), log(1.0))

    # 2. M006 UNCLAMPED FIT
    obj_006_unc <- function(p) { 
        v <- get_nll_006_unclamped(p, hyper, d$Boundary+1, d$Reward, d$RT, d$ITI)
        if(is.nan(v) || is.infinite(v)) 1e6 else v 
    }
    res_6_unc <- tryCatch(
        cma_es(init_6, obj_006_unc, control=list(maxit=200, sigma=0.5)), 
        error=function(e) list(par=init_6, value=NA)
    )

    # 3. M006 CLAMPED FIT
    obj_006_clamp <- function(p) { 
        v <- get_nll_006(p, hyper, d$Boundary+1, d$Reward, d$RT, d$ITI)
        if(is.nan(v) || is.infinite(v)) 1e6 else v 
    }
    res_6_clamp <- tryCatch(
        cma_es(init_6, obj_006_clamp, control=list(maxit=200, sigma=0.5)), 
        error=function(e) list(par=init_6, value=NA)
    )
    
    list(
        s_idx = s_idx,
        pid = d$participant_id[1],
        n_trials = nrow(d),
        nll_base = res_b$value,
        nll_unc = res_6_unc$value,
        nll_clamp = res_6_clamp$value
    )
}

cat("Executing 3-Way TRUE optimized cohort estimation (maxit=200, warm-started)...\n")
t_start <- Sys.time()
cohort_fits <- parallel::parLapply(cl, 1:S, fit_subject)
t_end <- Sys.time()
parallel::stopCluster(cl)

df_comp <- data.frame(
    SubjectID = sapply(cohort_fits, function(x) x$s_idx),
    ParticipantID = sapply(cohort_fits, function(x) x$pid),
    NTrials = sapply(cohort_fits, function(x) x$n_trials),
    NLL_Base = sapply(cohort_fits, function(x) x$nll_base),
    NLL_M006_Unclamped = sapply(cohort_fits, function(x) x$nll_unc),
    NLL_M006_Clamped = sapply(cohort_fits, function(x) x$nll_clamp)
) %>%
mutate(
    AIC_Base = 2 * 4 + 2 * NLL_Base,
    AIC_Unc = 2 * 9 + 2 * NLL_M006_Unclamped,
    AIC_Clamp = 2 * 9 + 2 * NLL_M006_Clamped,
    Delta_NLL_Unc = NLL_Base - NLL_M006_Unclamped,
    Delta_NLL_Clamp = NLL_Base - NLL_M006_Clamped,
    Delta_NLL_Clamp_vs_Unc = NLL_M006_Unclamped - NLL_M006_Clamped
)

write_csv(df_comp, file.path(results_dir, "cohort_comparison_metrics_3way.csv"))

cat("\n--- 3-WAY COMPARISON ---\n")
cat(sprintf("Mean NLL Baseline: %.2f\n", mean(df_comp$NLL_Base, na.rm=T)))
cat(sprintf("Mean NLL M006 Unclamped: %.2f\n", mean(df_comp$NLL_M006_Unclamped, na.rm=T)))
cat(sprintf("Mean NLL M006 Clamped: %.2f\n", mean(df_comp$NLL_M006_Clamped, na.rm=T)))
cat("\n--- PAIRWISE STATS ---\n")

t1 <- t.test(df_comp$NLL_Base, df_comp$NLL_M006_Clamped, paired=TRUE)
cat(sprintf("Clamp vs Base: Delta NLL = %.2f (t = %.2f, p = %.4e)\n", mean(df_comp$Delta_NLL_Clamp, na.rm=T), t1$statistic, t1$p.value))

t2 <- t.test(df_comp$NLL_M006_Unclamped, df_comp$NLL_M006_Clamped, paired=TRUE)
cat(sprintf("Clamp vs Unclamp: Delta NLL = %.2f (t = %.2f, p = %.4e)\n", mean(df_comp$Delta_NLL_Clamp_vs_Unc, na.rm=T), t2$statistic, t2$p.value))

t3 <- t.test(df_comp$NLL_Base, df_comp$NLL_M006_Unclamped, paired=TRUE)
cat(sprintf("Unclamp vs Base: Delta NLL = %.2f (t = %.2f, p = %.4e)\n", mean(df_comp$Delta_NLL_Unc, na.rm=T), t3$statistic, t3$p.value))

cat(sprintf("Subjects Favored (Clamp > Base): %d / %d\n", sum(df_comp$Delta_NLL_Clamp > 0, na.rm=T), S))
cat(sprintf("Subjects Favored (Clamp > Unclamp): %d / %d\n", sum(df_comp$Delta_NLL_Clamp_vs_Unc > 0, na.rm=T), S))

