pacman::p_load(tidyverse, Rcpp, cmaes, parallel, TTR, GGally, scales, grDevices, patchwork)

CORES <- parallel::detectCores()
hyper <- c(0.01, 1.00, 2.0)

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
d_list <- split(dat_clean, dat_clean$participant_idx)

hybrid_mat <- read_csv("results/tables/magi_subject_level_hybrid_S128_matrix.csv", show_col_types=FALSE)

cat("Re-fitting Individual Baseline Walds for 4D Projection...\n")
run_base_opt <- function(s_idx) {
    d <- d_list[[s_idx]]
    obj_base <- function(phi) {
        sim <- extract_baseline_wald_sim(phi, d$Boundary+1, d$Reward, d$RT)
        w1 <- mean(abs(sort(sim) - sort(d$RT)))
        var_res <- get_base_expect(phi, d$Boundary+1, d$Reward, d$RT)
        exp_t <- var_res[,2]
        if(nrow(d) > 10) {
            ema_emp <- EMA(d$RT, n=10)
            ema_term <- EMA(exp_t, n=10)
            valid <- !is.na(ema_emp) & !is.na(ema_term)
            if(sum(valid) > 0) phase <- mean((ema_emp[valid] - ema_term[valid])^2) else phase <- 1.0
        } else phase <- 1.0
        return(w1 + phase)
    }
    res <- cma_es(rep(0, 4), obj_base, control=list(maxit=300, sigma=0.5))
    return(c(SubjectID = s_idx, res$par, Cost = res$value))
}

base_res_list <- mclapply(1:S, run_base_opt, mc.cores = CORES)
base_mat <- as.data.frame(do.call(rbind, base_res_list))

cat("Generating 4D Objective Vectors...\n")
get_spectral_beta <- function(x) {
    s <- spectrum(x, plot=FALSE)
    coef(lm(log(s$spec) ~ log(s$freq)))[2]
}

calc_objectives <- function(df) {
    w1_base <- mean(abs(sort(df$Baseline_Wald_Sim_RT) - sort(df$Empirical_RT)))
    w1_term <- mean(abs(sort(df$Terminal_Hybrid_Sim_RT) - sort(df$Empirical_RT)))
    
    emp_beta <- get_spectral_beta(df$Empirical_RT)
    base_beta <- get_spectral_beta(df$Baseline_Wald_Expected_RT)
    term_beta <- get_spectral_beta(df$Terminal_Hybrid_Expected_RT)
    db_base <- abs(emp_beta - base_beta)
    db_term <- abs(emp_beta - term_beta)
    
    df_sw <- df %>% dplyr::filter(!is.na(Is_Switch))
    eps <- 1e-15
    p_b_v <- pmax(pmin(df_sw$P_Switch_Base, 1-eps), eps)
    p_t_v <- pmax(pmin(df_sw$P_Switch_Term, 1-eps), eps)
    ll_base <- -mean(df_sw$Is_Switch * log(p_b_v) + (1-df_sw$Is_Switch) * log(1-p_b_v))
    ll_term <- -mean(df_sw$Is_Switch * log(p_t_v) + (1-df_sw$Is_Switch) * log(1-p_t_v))
    
    df_ema <- df %>% group_by(SubjectID) %>% mutate(
        ema_emp = EMA(Empirical_RT, n=10),
        ema_base = EMA(Baseline_Wald_Expected_RT, n=10),
        ema_term = EMA(Terminal_Hybrid_Expected_RT, n=10)
    ) %>% ungroup() %>% dplyr::filter(!is.na(ema_emp))
    phase_base <- mean((df_ema$ema_emp - df_ema$ema_base)^2)
    phase_term <- mean((df_ema$ema_emp - df_ema$ema_term)^2)
    
    return(c(w1_base, db_base, ll_base, phase_base, w1_term, db_term, ll_term, phase_term))
}

gen_4d <- function(s_idx) {
    d <- d_list[[s_idx]]
    p_b <- as.numeric(base_mat[s_idx, 2:5])
    p_t <- as.numeric(hybrid_mat[s_idx, 2:13])
    
    sim_b <- extract_baseline_wald_sim(p_b, d$Boundary+1, d$Reward, d$RT)
    exp_b_res <- get_base_expect(p_b, d$Boundary+1, d$Reward, d$RT)
    sim_t <- extract_epoch10_2_hybrid(p_t, hyper, d$Boundary+1, d$Reward, d$RT)
    exp_t_res <- get_hybrid_expect(p_t, hyper, d$Boundary+1, d$Reward, d$RT)
    
    is_sw <- rep(NA, nrow(d))
    p_b_sw <- rep(0.5, nrow(d))
    p_t_sw <- rep(0.5, nrow(d))
    if(nrow(d) > 1) {
        for(t in 2:nrow(d)) {
            is_sw[t] <- ifelse(d$Boundary[t] != d$Boundary[t-1], 1, 0)
            p_b_sw[t] <- 1 - exp_b_res[t, d$Boundary[t-1]+1]
            p_t_sw[t] <- 1 - exp_t_res[t, d$Boundary[t-1]+1]
        }
    }
    
    df <- data.frame(
        SubjectID=s_idx, Trial=1:nrow(d), Empirical_RT=d$RT,
        Baseline_Wald_Sim_RT=sim_b, Baseline_Wald_Expected_RT=exp_b_res[,2],
        Terminal_Hybrid_Sim_RT=sim_t, Terminal_Hybrid_Expected_RT=exp_t_res[,2],
        Is_Switch=is_sw, P_Switch_Base=p_b_sw, P_Switch_Term=p_t_sw
    )
    
    res <- calc_objectives(df)
    return(data.frame(
        SubjectID = s_idx,
        Model = c("Baseline Wald", "Terminal Hybrid"),
        W1 = c(res[1], res[5]),
        DeltaBeta = c(res[2], res[6]),
        LogLoss = c(res[3], res[7]),
        Phase = c(res[4], res[8])
    ))
}

obj_list <- mclapply(1:S, gen_4d, mc.cores = CORES)
df_obj <- bind_rows(obj_list)

dir.create("results/figures", showWarnings=F)

# 1. Parallel Coordinates
cat("Plotting 1...\n")
p1 <- ggparcoord(df_obj, columns = 3:6, groupColumn = 2, alphaLines = 0.2) +
    theme_minimal() +
    scale_color_manual(values=c("Baseline Wald"="#E69F00", "Terminal Hybrid"="#56B4E9")) +
    labs(title="Global 4D Topological Projection (N=128)", x="Objective Space Constraint", y="Normalized Error Vector")
ggsave("results/figures/magi_parallel_coordinates.png", p1, width=8, height=6)

# 2. Bivariate Pareto Trade-Off
cat("Plotting 2...\n")
hulls <- df_obj %>% group_by(Model) %>% slice(chull(W1, Phase))
p2 <- ggplot(df_obj, aes(x=W1, y=Phase, color=Model)) +
    geom_point(alpha=0.6) +
    geom_polygon(data=hulls, aes(fill=Model), alpha=0.2) +
    theme_minimal() +
    scale_color_manual(values=c("Baseline Wald"="#E69F00", "Terminal Hybrid"="#56B4E9")) +
    scale_fill_manual(values=c("Baseline Wald"="#E69F00", "Terminal Hybrid"="#56B4E9")) +
    labs(title="Bivariate Pareto Topology (W1 vs Phase)", x="Topological Error (W1)", y="Chronological Error (Phase)")
ggsave("results/figures/magi_bivariate_pareto.png", p2, width=8, height=6)

# 3. Polar Topologies (Sub 42)
cat("Plotting 3...\n")
Z_ref <- c( W1 = max(df_obj$W1)*1.1, DeltaBeta = max(df_obj$DeltaBeta)*1.1, 
            LogLoss = max(df_obj$LogLoss)*1.1, Phase = max(df_obj$Phase)*1.1 )
df_radar_scaled <- df_obj %>% filter(SubjectID == 42) %>% mutate(
    W1 = W1 / Z_ref["W1"], DeltaBeta = DeltaBeta / Z_ref["DeltaBeta"],
    LogLoss = LogLoss / Z_ref["LogLoss"], Phase = Phase / Z_ref["Phase"]
)
df_polar <- df_radar_scaled %>% pivot_longer(cols=c(W1, DeltaBeta, LogLoss, Phase), names_to="Objective", values_to="Value")
# Close polygon logic for ggplot2 coord_polar
df_polar <- df_polar %>% mutate(Objective = factor(Objective, levels=c("W1", "DeltaBeta", "LogLoss", "Phase")))
p3 <- ggplot(df_polar, aes(x=Objective, y=Value, group=Model, color=Model, fill=Model)) +
    geom_polygon(alpha=0.3) +
    geom_point(size=2) +
    coord_polar() +
    theme_minimal() +
    scale_y_continuous(limits=c(0, NA)) +
    scale_color_manual(values=c("Baseline Wald"="#E69F00", "Terminal Hybrid"="#56B4E9")) +
    scale_fill_manual(values=c("Baseline Wald"="#E69F00", "Terminal Hybrid"="#56B4E9")) +
    labs(title="Subject-Level Error Footprint (Sub 42)", y="Normalized Error Magnitude")
ggsave("results/figures/magi_polar_footprint.png", p3, width=6, height=6)

# 4. ECDF of Epsilon-Indicator
cat("Plotting 4...\n")
eps_list <- lapply(1:S, function(s) {
    sub <- df_obj %>% filter(SubjectID == s)
    base_obj <- as.numeric(sub[sub$Model=="Baseline Wald", 3:6])
    term_obj <- as.numeric(sub[sub$Model=="Terminal Hybrid", 3:6])
    data.frame(SubjectID=s, Epsilon=max(term_obj - base_obj))
})
df_eps <- bind_rows(eps_list)

p4 <- ggplot(df_eps, aes(x=Epsilon)) +
    stat_ecdf(geom="step", color="#D55E00", size=1.2) +
    theme_minimal() +
    labs(title="Population Dominance Distribution (ECDF of Epsilon-indicator)",
         x="Translation Magnitude Required for Baseline Parity", y="Cumulative Proportion of Population")
ggsave("results/figures/magi_ecdf_epsilon.png", p4, width=8, height=6)

cat("--- VISUAL PROJECTIONS SEALED ---\n")
