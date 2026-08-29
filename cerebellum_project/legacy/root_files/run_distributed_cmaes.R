pacman::p_load(tidyverse, Rcpp, cmaes, parallel, TTR)

cat("--- SYSTEM OVERRIDE: INITIATING BIOLOGICAL INDIVIDUALITY PROTOCOL ---\n")
CORES <- parallel::detectCores()
RAM <- system("awk '/MemTotal/ {printf \"%.1f\", $2/1024/1024}' /proc/meminfo", intern=TRUE)
cat("Dynamic Hardware Detection -> CORES:", CORES, "| RAM:", RAM, "GB\n")
cat("Fully saturating asynchronous cluster...\n")

Rcpp::sourceCpp("src/models/epoch9_qperturbed_wald.cpp") 
Rcpp::sourceCpp("src/models/magi_terminal_decoupled_hybrid.cpp") 
Rcpp::sourceCpp("extract_expectations.cpp")

dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types = FALSE)
dat_raw$Reward <- dat_raw$F
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>% 
    group_by(participant_id) %>% 
    mutate(RT = (ttr - ttp) / 1000, Boundary = ifelse(Resp == 2, 1, 0)) %>% 
    ungroup() %>% dplyr::filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(Reward)) %>%
    mutate(participant_idx = as.integer(as.factor(participant_id)))

S <- max(dat_clean$participant_idx)
d_sub <- dat_clean %>% dplyr::filter(participant_idx <= S)
d_list <- split(d_sub, d_sub$participant_idx)
hyper <- c(0.01, 1.00, 2.0)

cat(sprintf("Initiating S=%d Independent Asynchronous Optimization Matrices...\n", S))

run_subject_optimization <- function(s_idx) {
    d <- d_list[[s_idx]]
    
    obj_var <- function(phi) {
        # 1. Topological Constraint (W1)
        sim <- extract_epoch10_2_hybrid(phi, hyper, d$Boundary+1, d$Reward, d$RT)
        w1 <- mean(abs(sort(sim) - sort(d$RT)))
        
        # 2. Chronological Constraint (Phase Error)
        var_res <- get_hybrid_expect(phi, hyper, d$Boundary+1, d$Reward, d$RT)
        exp_t <- var_res[,2]
        
        if(nrow(d) > 10) {
            ema_emp <- EMA(d$RT, n=10)
            ema_term <- EMA(exp_t, n=10)
            valid <- !is.na(ema_emp) & !is.na(ema_term)
            if(sum(valid) > 0) {
                phase <- mean((ema_emp[valid] - ema_term[valid])^2)
            } else {
                phase <- 1.0
            }
        } else {
            phase <- 1.0
        }
        
        # Joint biological constraint objective
        return(w1 + phase)
    }
    
    # Run 500-iteration evolutionary search strictly for this brain's manifold
    res_var <- cma_es(rep(0, 12), obj_var, control=list(maxit=500, sigma=0.5))
    
    out <- c(SubjectID = s_idx, res_var$par, Cost = res_var$value)
    return(out)
}

start_time <- Sys.time()
subject_results <- mclapply(1:S, run_subject_optimization, mc.cores = CORES)
end_time <- Sys.time()

cat("Optimization completed in:", round(as.numeric(difftime(end_time, start_time, units="mins")), 2), "minutes.\n")

param_matrix <- as.data.frame(do.call(rbind, subject_results))
colnames(param_matrix) <- c("SubjectID", paste0("phi_", 1:12), "Joint_Cost")

write_csv(param_matrix, "results/tables/magi_subject_level_hybrid_S128_matrix.csv")
cat("SUCCESS: Subject-level S x 12 parameter matrix physically sealed.\n")
