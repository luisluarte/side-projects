pacman::p_load(tidyverse, Rcpp, cmaes, parallel, TTR)

CORES <- parallel::detectCores()
hyper <- c(0.01, 1.00, 2.0)

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

gen_4d <- function(s_idx) {
    tryCatch({
        d <- d_list[[s_idx]]
        
        obj_lti <- function(p) eval_model_standard(extract_epoch4_lti(p, d$Boundary+1, d$Reward, d$RT), d)
        res_lti <- cma_es(rep(0, 10), obj_lti, control=list(maxit=100, sigma=0.5))
        sim_lti <- extract_epoch4_lti(res_lti$par, d$Boundary+1, d$Reward, d$RT)
        
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
        
        w1_lti <- mean(abs(sort(sim_lti) - sort(d$RT)))
        db_lti <- abs(get_spectral_beta(d$RT) - get_spectral_beta(sim_lti))
        ema_emp <- EMA(d$RT, n=10); ema_sim_lti <- EMA(sim_lti, n=10)
        valid <- !is.na(ema_emp) & !is.na(ema_sim_lti)
        ph_lti <- if(sum(valid) > 0) mean((ema_emp[valid] - ema_sim_lti[valid])^2) else 1.0
        
        w1_hyb <- mean(abs(sort(sim_hyb) - sort(d$RT)))
        db_hyb <- abs(get_spectral_beta(d$RT) - get_spectral_beta(sim_hyb))
        ema_sim_hyb <- EMA(sim_hyb, n=10)
        valid <- !is.na(ema_emp) & !is.na(ema_sim_hyb)
        ph_hyb <- if(sum(valid) > 0) mean((ema_emp[valid] - ema_sim_hyb[valid])^2) else 1.0
        
        data.frame(SubjectID = s_idx, W1_LTI = w1_lti, dB_LTI = db_lti, Ph_LTI = ph_lti, W1_Hyb = w1_hyb, dB_Hyb = db_hyb, Ph_Hyb = ph_hyb)
    }, error = function(e) {
        data.frame(SubjectID = s_idx, W1_LTI=NA, dB_LTI=NA, Ph_LTI=NA, W1_Hyb=NA, dB_Hyb=NA, Ph_Hyb=NA)
    })
}

cat("Extracting Individualized 3D Geometries...\n")
obj_res <- mclapply(1:S, gen_4d, mc.cores = CORES)
df_obj <- bind_rows(obj_res) %>% drop_na()

cat("Running Pareto Spectral Bootstrap (1000 Iterations)...\n")
boot_res <- mclapply(1:1000, function(b) {
    idx <- sample(1:nrow(df_obj), nrow(df_obj), replace=TRUE)
    boot_df <- df_obj[idx, ]
    
    obj_lti <- c(mean(boot_df$W1_LTI), mean(boot_df$dB_LTI), mean(boot_df$Ph_LTI))
    obj_hyb <- c(mean(boot_df$W1_Hyb), mean(boot_df$dB_Hyb), mean(boot_df$Ph_Hyb))
    
    Z_ref <- c(
        max(c(df_obj$W1_LTI, df_obj$W1_Hyb)) * 1.1,
        max(c(df_obj$dB_LTI, df_obj$dB_Hyb)) * 1.1,
        max(c(df_obj$Ph_LTI, df_obj$Ph_Hyb)) * 1.1
    )
    
    hv_lti <- (Z_ref[1] - obj_lti[1]) * (Z_ref[2] - obj_lti[2]) * (Z_ref[3] - obj_lti[3])
    hv_hyb <- (Z_ref[1] - obj_hyb[1]) * (Z_ref[2] - obj_hyb[2]) * (Z_ref[3] - obj_hyb[3])
    eps <- max(c(obj_hyb[1] - obj_lti[1], obj_hyb[2] - obj_lti[2], obj_hyb[3] - obj_lti[3]))
    
    data.frame(Iter = b, HV_LTI = hv_lti, HV_Hyb = hv_hyb, Delta_HV = hv_hyb - hv_lti, Epsilon = eps)
}, mc.cores = CORES)

boot_df <- bind_rows(boot_res)

sink("results/tables/magi_reg_vs_lti_bootstrap_stats.txt")
cat("=== REGULARIZED HYBRID vs. LTI CASCADE (3D PARETO BOOTSTRAP) ===\n")
cat("Global W1 LTI Cascade:", mean(df_obj$W1_LTI), "\n")
cat("Global W1 Reg. Hybrid:", mean(df_obj$W1_Hyb), "\n")
cat("P-Value of Supremacy (LTI < Hybrid):", mean(boot_df$Delta_HV <= 0), "\n")
cat("Mean Delta HV:", mean(boot_df$Delta_HV), "\n")
cat("Epsilon Indicator:", mean(boot_df$Epsilon), "\n")
sink()
cat("--- BOOTSTRAP COMPLETE ---\n")
