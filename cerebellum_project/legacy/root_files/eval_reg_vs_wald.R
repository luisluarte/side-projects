pacman::p_load(tidyverse, Rcpp, cmaes, parallel, TTR)

CORES <- parallel::detectCores()
hyper <- c(0.01, 1.00, 2.0)

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

gen_4d <- function(s_idx) {
    tryCatch({
        d <- d_list[[s_idx]]
        
        obj_wald <- function(p) eval_model_standard(extract_baseline_wald_sim(p, d$Boundary+1, d$Reward, d$RT), d)
        res_wald <- cma_es(rep(0, 4), obj_wald, control=list(maxit=100, sigma=0.5))
        sim_wald <- extract_baseline_wald_sim(res_wald$par, d$Boundary+1, d$Reward, d$RT)
        
        lambda_reg <- 0.10
        alpha_en <- 0.5
        obj_hyb_reg <- function(p) {
            sim <- extract_epoch10_2_hybrid(p, hyper, d$Boundary+1, d$Reward, d$RT)
            base_cost <- eval_model_standard(sim, d)
            p_cb <- p[5:12]
            penalty <- lambda_reg * (alpha_en * sum(abs(p_cb)) + (1 - alpha_en) * sum(p_cb^2))
            return(base_cost + penalty)
        }
        res_hyb <- cma_es(rep(0, 12), obj_hyb_reg, control=list(maxit=100, sigma=0.5))
        sim_hyb <- extract_epoch10_2_hybrid(res_hyb$par, hyper, d$Boundary+1, d$Reward, d$RT)
        
        w1_wald <- mean(abs(sort(sim_wald) - sort(d$RT)))
        db_wald <- abs(get_spectral_beta(d$RT) - get_spectral_beta(sim_wald))
        ema_emp <- EMA(d$RT, n=10); ema_sim_wald <- EMA(sim_wald, n=10)
        valid <- !is.na(ema_emp) & !is.na(ema_sim_wald)
        ph_wald <- if(sum(valid) > 0) mean((ema_emp[valid] - ema_sim_wald[valid])^2) else 1.0
        
        w1_hyb <- mean(abs(sort(sim_hyb) - sort(d$RT)))
        db_hyb <- abs(get_spectral_beta(d$RT) - get_spectral_beta(sim_hyb))
        ema_sim_hyb <- EMA(sim_hyb, n=10)
        valid <- !is.na(ema_emp) & !is.na(ema_sim_hyb)
        ph_hyb <- if(sum(valid) > 0) mean((ema_emp[valid] - ema_sim_hyb[valid])^2) else 1.0
        
        data.frame(SubjectID = s_idx, W1_Wald = w1_wald, dB_Wald = db_wald, Ph_Wald = ph_wald, W1_Hyb = w1_hyb, dB_Hyb = db_hyb, Ph_Hyb = ph_hyb)
    }, error = function(e) {
        data.frame(SubjectID = s_idx, W1_Wald=NA, dB_Wald=NA, Ph_Wald=NA, W1_Hyb=NA, dB_Hyb=NA, Ph_Hyb=NA)
    })
}

cat("Extracting Individualized 3D Geometries...\n")
obj_res <- mclapply(1:S, gen_4d, mc.cores = CORES)
df_obj <- bind_rows(obj_res) %>% drop_na()

cat("Running Pareto Spectral Bootstrap (1000 Iterations)...\n")
boot_res <- mclapply(1:1000, function(b) {
    idx <- sample(1:nrow(df_obj), nrow(df_obj), replace=TRUE)
    boot_df <- df_obj[idx, ]
    
    obj_wald <- c(mean(boot_df$W1_Wald), mean(boot_df$dB_Wald), mean(boot_df$Ph_Wald))
    obj_hyb <- c(mean(boot_df$W1_Hyb), mean(boot_df$dB_Hyb), mean(boot_df$Ph_Hyb))
    
    Z_ref <- c(
        max(c(df_obj$W1_Wald, df_obj$W1_Hyb)) * 1.1,
        max(c(df_obj$dB_Wald, df_obj$dB_Hyb)) * 1.1,
        max(c(df_obj$Ph_Wald, df_obj$Ph_Hyb)) * 1.1
    )
    
    hv_wald <- (Z_ref[1] - obj_wald[1]) * (Z_ref[2] - obj_wald[2]) * (Z_ref[3] - obj_wald[3])
    hv_hyb <- (Z_ref[1] - obj_hyb[1]) * (Z_ref[2] - obj_hyb[2]) * (Z_ref[3] - obj_hyb[3])
    eps <- max(c(obj_hyb[1] - obj_wald[1], obj_hyb[2] - obj_wald[2], obj_hyb[3] - obj_wald[3]))
    
    data.frame(Iter = b, HV_Wald = hv_wald, HV_Hyb = hv_hyb, Delta_HV = hv_hyb - hv_wald, Epsilon = eps)
}, mc.cores = CORES)

boot_df <- bind_rows(boot_res)

sink("results/tables/magi_reg_vs_wald_bootstrap_stats.txt")
cat("=== REGULARIZED HYBRID vs. BASELINE WALD (3D PARETO BOOTSTRAP) ===\n")
cat("Global W1 Baseline Wald:", mean(df_obj$W1_Wald), "\n")
cat("Global W1 Reg. Hybrid:", mean(df_obj$W1_Hyb), "\n")
cat("P-Value of Supremacy (Wald < Hybrid):", mean(boot_df$Delta_HV <= 0), "\n")
cat("Mean Delta HV:", mean(boot_df$Delta_HV), "\n")
cat("Epsilon Indicator:", mean(boot_df$Epsilon), "\n")
sink()
cat("--- BOOTSTRAP COMPLETE ---\n")
