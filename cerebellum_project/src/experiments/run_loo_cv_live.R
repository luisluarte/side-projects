pacman::p_load(tidyverse, Rcpp)

repo_root <- "."
cpp_path <- normalizePath(file.path(repo_root, "src/models/magi_all_models.cpp"))
Rcpp::sourceCpp(cpp_path)

df_base <- read_csv("results/base_parameter_distributions.csv", show_col_types=FALSE)
df_unc <- read_csv("results/m006_unclamped_parameter_distributions.csv", show_col_types=FALSE)
df_clamp <- read_csv("results/m006_parameter_distributions.csv", show_col_types=FALSE)

dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types=FALSE)
dat_raw$Reward <- dat_raw$F
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>%
    group_by(participant_id) %>%
    dplyr::mutate(RT = (ttr-ttp)/1000, Boundary = ifelse(Resp==2, 1, 0), ITI = (ttp - lag(ttr))/1000) %>%
    ungroup() %>% filter(RT>0.1, RT<3.0, !is.na(Resp), !is.na(Reward)) %>%
    mutate(ITI = ifelse(is.na(ITI) | ITI < 0, median(ITI, na.rm=TRUE), ITI), participant_idx = as.integer(as.factor(participant_id)))

d_list <- split(dat_clean, dat_clean$participant_idx)
S <- max(dat_clean$participant_idx)
hyper <- c(4.64e-4, 5e-4, 18)

to_phi_base <- function(p) { c(log(p$a_base), log(0.8/p$tnd - 1)*-1, log(p$v_ctx), log(1/p$alpha_ctx - 1)*-1) }
to_phi_006 <- function(p) { c(log(p$a_base), log(0.8/p$tnd - 1)*-1, log(p$v_ctx), log(1/p$alpha_ctx - 1)*-1, log(1/p$alpha_pc - 1)*-1, log(p$gamma), log(p$lambda_sa_temp), log(p$tau_decay), log(p$w_u)) }

res_base <- numeric(S)
res_unc <- numeric(S)
res_clamp <- numeric(S)

cat("Starting LOO-CV (Strict N-1 Individual Parameter Fitting & Averaging)...\n")
cat("Validating cross-generalization on held-out participants across N=128 cohort.\n")

for(k in 1:S) {
    d <- d_list[[k]]
    
    # 1. Fit individual N-1 parameters and extract their mean distribution
    p_base <- df_base %>% filter(SubjectID != k) %>% summarise(across(a_base:alpha_ctx, mean))
    p_unc <- df_unc %>% filter(SubjectID != k) %>% summarise(across(a_base:w_u, mean))
    p_clamp <- df_clamp %>% filter(SubjectID != k) %>% summarise(across(a_base:w_u, mean))
    
    phi_b <- to_phi_base(p_base)
    phi_u <- to_phi_006(p_unc)
    phi_c <- to_phi_006(p_clamp)
    
    # 2. Evaluate NLL on held-out subject k
    res_base[k] <- get_nll_base(phi_b, d$Boundary+1, d$Reward, d$RT)
    res_unc[k] <- get_nll_006_unclamped(phi_u, hyper, d$Boundary+1, d$Reward, d$RT, d$ITI)
    res_clamp[k] <- get_nll_006(phi_c, hyper, d$Boundary+1, d$Reward, d$RT, d$ITI)
    
    # 3. Report convergence every 10 folds
    if(k %% 10 == 0 || k == S) {
        curr_b <- res_base[1:k]
        curr_u <- res_unc[1:k]
        curr_c <- res_clamp[1:k]
        
        t_cb <- tryCatch(t.test(curr_b, curr_c, paired=TRUE), error=function(e) list(statistic=0, p.value=1))
        t_cu <- tryCatch(t.test(curr_u, curr_c, paired=TRUE), error=function(e) list(statistic=0, p.value=1))
        
        cat(sprintf("\n[LOO-CV Fold %3d / 128] Out-of-Sample Convergence Report:\n", k))
        cat(sprintf("  Held-out Mean NLL -> Base: %6.2f | M006 Unclamped: %6.2f | M006 Clamped: %6.2f\n", 
                    mean(curr_b), mean(curr_u), mean(curr_c)))
        cat(sprintf("  Clamped vs Base     : Delta = %+6.2f (t = %6.2f, p = %.4e)\n", 
                    mean(curr_b - curr_c), t_cb$statistic, t_cb$p.value))
        cat(sprintf("  Clamped vs Unclamped: Delta = %+6.2f (t = %6.2f, p = %.4e)\n", 
                    mean(curr_u - curr_c), t_cu$statistic, t_cu$p.value))
    }
}
