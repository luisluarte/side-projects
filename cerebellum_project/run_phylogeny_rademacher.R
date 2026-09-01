pacman::p_load(tidyverse, Rcpp, cmaes, parallel, TTR, ecr)

CORES <- parallel::detectCores()
hyper <- c(0.01, 1.00, 2.0)

Rcpp::sourceCpp("src/models/epoch1_landscape.cpp")
Rcpp::sourceCpp("src/models/epoch4_champion_lti.cpp")
Rcpp::sourceCpp("src/models/epoch9_qperturbed_wald.cpp")
Rcpp::sourceCpp("src/models/magi_terminal_decoupled_hybrid.cpp")

dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types = FALSE)
dat_raw$Reward <- dat_raw$F
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>% 
    group_by(participant_id) %>% 
    dplyr::mutate(RT = (ttr - ttp) / 1000, Boundary = ifelse(Resp == 2, 1, 0)) %>% 
    ungroup() %>% dplyr::filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(Reward)) %>%
    dplyr::mutate(participant_idx = as.integer(as.factor(participant_id)))

S <- max(dat_clean$participant_idx)
d_list <- split(dat_clean, dat_clean$participant_idx)

get_spectral_beta <- function(x) {
    if(length(x) < 5 || var(x, na.rm=TRUE) < 1e-6) return(0)
    s <- tryCatch(spectrum(x, plot=FALSE), error=function(e) NULL)
    if(is.null(s)) return(0)
    b <- tryCatch(coef(lm(log(s$spec) ~ log(s$freq)))[2], error=function(e) 0)
    if(is.na(b)) return(0)
    return(b)
}

eval_model_standard <- function(sim, d) {
    if(any(is.na(sim))) return(1e6)
    w1 <- mean(abs(sort(sim) - sort(d$RT)))
    emp_b <- get_spectral_beta(d$RT)
    sim_b <- get_spectral_beta(sim)
    db <- abs(emp_b - sim_b)
    if(is.na(db)) db <- 1.0
    
    if(nrow(d) > 10) {
        ema_emp <- EMA(d$RT, n=10)
        ema_sim <- EMA(sim, n=10)
        valid <- !is.na(ema_emp) & !is.na(ema_sim)
        if(sum(valid) > 0) phase <- mean((ema_emp[valid] - ema_sim[valid])^2) else phase <- 1.0
    } else phase <- 1.0
    
    cost <- w1 + phase + db
    if(is.na(cost) || is.infinite(cost)) return(1e6)
    return(cost)
}

optimize_sweep <- function(s_idx) {
    tryCatch({
        d <- d_list[[s_idx]]
        
        obj1 <- function(p) eval_model_standard(extract_baseline_5(p, d$Boundary+1, d$Reward, d$RT, FALSE), d)
        res1 <- cma_es(rep(0, 10), obj1, control=list(maxit=100, sigma=0.5))
        
        obj2 <- function(p) eval_model_standard(extract_epoch4_lti(p, d$Boundary+1, d$Reward, d$RT), d)
        res2 <- cma_es(rep(0, 10), obj2, control=list(maxit=100, sigma=0.5))
        
        obj3 <- function(p) eval_model_standard(extract_epoch9_qperturbed(p, hyper, d$Boundary+1, d$Reward, d$RT), d)
        res3 <- cma_es(rep(0, 8), obj3, control=list(maxit=100, sigma=0.5))
        
        obj4 <- function(p) eval_model_standard(extract_epoch10_2_hybrid(p, hyper, d$Boundary+1, d$Reward, d$RT), d)
        res4 <- cma_es(rep(0, 12), obj4, control=list(maxit=100, sigma=0.5))
        
        data.frame(SubjectID = s_idx, Cost_Reservoir = res1$value, Cost_LTI = res2$value, Cost_QPerturbed = res3$value, Cost_Hybrid = res4$value)
    }, error = function(e) {
        data.frame(SubjectID = s_idx, Cost_Reservoir = NA, Cost_LTI = NA, Cost_QPerturbed = NA, Cost_Hybrid = NA)
    })
}

cat("Executing 4-Model Phylogeny Sweep...\n")
sweep_res <- mclapply(1:S, optimize_sweep, mc.cores = CORES)
df_sweep <- bind_rows(sweep_res)
mean_costs <- colMeans(df_sweep %>% dplyr::select(starts_with("Cost_")), na.rm=TRUE)

cat("Executing Empirical Rademacher Bounding...\n")
rademacher_eval <- function(s_idx) {
    tryCatch({
        d <- d_list[[s_idx]]
        set.seed(s_idx)
        R_noise <- sample(c(-1, 1), nrow(d), replace=TRUE)
        
        obj_lti <- function(p) {
            sim <- extract_epoch4_lti(p, d$Boundary+1, d$Reward, d$RT)
            if(sd(sim, na.rm=T) < 1e-6) return(1.0)
            c_val <- cor(sim, R_noise, use="complete.obs")
            if(is.na(c_val)) return(1.0)
            return(-c_val)
        }
        r_lti <- cma_es(rep(0, 10), obj_lti, control=list(maxit=80, sigma=0.5))
        
        obj_hyb <- function(p) {
            sim <- extract_epoch10_2_hybrid(p, hyper, d$Boundary+1, d$Reward, d$RT)
            if(sd(sim, na.rm=T) < 1e-6) return(1.0)
            c_val <- cor(sim, R_noise, use="complete.obs")
            if(is.na(c_val)) return(1.0)
            return(-c_val)
        }
        r_hyb <- cma_es(rep(0, 12), obj_hyb, control=list(maxit=80, sigma=0.5))
        
        data.frame(SubjectID = s_idx, Rad_LTI = -r_lti$value, Rad_Hybrid = -r_hyb$value)
    }, error = function(e) {
        data.frame(SubjectID = s_idx, Rad_LTI = NA, Rad_Hybrid = NA)
    })
}

rad_res <- mclapply(1:S, rademacher_eval, mc.cores = CORES)
df_rad <- bind_rows(rad_res)

sink("results/tables/magi_phylogeny_rademacher_stats.txt")
cat("=== UNIFIED MULTI-OBJECTIVE PHYLOGENY SWEEP ===\n")
cat("Mean Cost Reservoir:", mean_costs["Cost_Reservoir"], "\n")
cat("Mean Cost LTI Cascade:", mean_costs["Cost_LTI"], "\n")
cat("Mean Cost Q-Perturbed:", mean_costs["Cost_QPerturbed"], "\n")
cat("Mean Cost Terminal Hybrid:", mean_costs["Cost_Hybrid"], "\n\n")
cat("=== EMPIRICAL RADEMACHER COMPLEXITY (R^S) ===\n")
cat("Mean Rademacher LTI:", mean(df_rad$Rad_LTI, na.rm=TRUE), "\n")
cat("Mean Rademacher Hybrid:", mean(df_rad$Rad_Hybrid, na.rm=TRUE), "\n")
cat("Rademacher Overfit Ratio:", mean(df_rad$Rad_Hybrid, na.rm=TRUE) / mean(df_rad$Rad_LTI, na.rm=TRUE), "\n")
sink()
cat("--- SWEEP COMPLETE ---\n")
