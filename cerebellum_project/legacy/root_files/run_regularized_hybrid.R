pacman::p_load(tidyverse, Rcpp, cmaes, parallel, TTR)

CORES <- parallel::detectCores()
hyper <- c(0.01, 1.00, 2.0)

Rcpp::sourceCpp("src/models/epoch10_multiplicative_wald.cpp") # dummy to ensure compiler is warm
Rcpp::sourceCpp("src/models/epoch4_champion_lti.cpp")
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

optimize_regularized <- function(s_idx) {
    tryCatch({
        d <- d_list[[s_idx]]
        
        lambda_reg <- 0.10  # Stringent penalty
        alpha_en <- 0.5
        
        obj_hyb_reg <- function(p) {
            sim <- extract_epoch10_2_hybrid(p, hyper, d$Boundary+1, d$Reward, d$RT)
            base_cost <- eval_model_standard(sim, d)
            
            p_cb <- p[5:12]
            l1 <- sum(abs(p_cb))
            l2 <- sum(p_cb^2)
            penalty <- lambda_reg * (alpha_en * l1 + (1 - alpha_en) * l2)
            
            return(base_cost + penalty)
        }
        
        res_reg <- cma_es(rep(0, 12), obj_hyb_reg, control=list(maxit=100, sigma=0.5))
        
        sim_final <- extract_epoch10_2_hybrid(res_reg$par, hyper, d$Boundary+1, d$Reward, d$RT)
        pure_cost <- eval_model_standard(sim_final, d)
        
        set.seed(s_idx)
        R_noise <- sample(c(-1, 1), nrow(d), replace=TRUE)
        
        obj_hyb_rad <- function(p) {
            sim <- extract_epoch10_2_hybrid(p, hyper, d$Boundary+1, d$Reward, d$RT)
            if(sd(sim, na.rm=TRUE) < 1e-6) return(1.0)
            c_val <- cor(sim, R_noise, use="complete.obs")
            if(is.na(c_val)) return(1.0)
            
            p_cb <- p[5:12]
            l1 <- sum(abs(p_cb))
            l2 <- sum(p_cb^2)
            penalty <- lambda_reg * (alpha_en * l1 + (1 - alpha_en) * l2)
            
            return(-c_val + penalty)
        }
        
        r_hyb_rad <- cma_es(rep(0, 12), obj_hyb_rad, control=list(maxit=80, sigma=0.5))
        sim_rad <- extract_epoch10_2_hybrid(r_hyb_rad$par, hyper, d$Boundary+1, d$Reward, d$RT)
        final_c_val <- cor(sim_rad, R_noise, use="complete.obs")
        
        data.frame(SubjectID = s_idx, Pure_Cost = pure_cost, Rad_Hybrid = final_c_val)
    }, error = function(e) {
        data.frame(SubjectID = s_idx, Pure_Cost = NA, Rad_Hybrid = NA)
    })
}

cat("Executing Elastic-Net Regularized Sweep...\n")
sweep_res <- mclapply(1:S, optimize_regularized, mc.cores = CORES)
df_reg <- bind_rows(sweep_res)

sink("results/tables/magi_regularized_stats.txt")
cat("=== ELASTIC-NET REGULARIZATION RESULTS ===\n")
cat("Mean Pure Cost (Reg. Hybrid):", mean(df_reg$Pure_Cost, na.rm=TRUE), "\n")
cat("Mean Rademacher (Reg. Hybrid):", mean(df_reg$Rad_Hybrid, na.rm=TRUE), "\n")
cat("Rademacher Overfit Ratio (vs LTI Anchor 0.157):", mean(df_reg$Rad_Hybrid, na.rm=TRUE) / 0.157, "\n")
sink()
cat("--- REGULARIZATION COMPLETE ---\n")
