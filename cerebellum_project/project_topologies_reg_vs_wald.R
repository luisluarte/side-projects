pacman::p_load(tidyverse, Rcpp, cmaes, parallel, TTR, GGally)

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

cat("Formatting Plot Topologies...\n")
df_plot <- df_obj %>% pivot_longer(cols=c(-SubjectID), names_to="Metric_Model", values_to="Value") %>%
    mutate(Model = ifelse(grepl("Wald", Metric_Model), "Baseline Wald", "Reg. Hybrid"),
           Objective = gsub("_.*", "", Metric_Model)) %>%
    dplyr::select(-Metric_Model) %>%
    pivot_wider(names_from="Objective", values_from="Value")

df_plot <- df_plot %>% rename(DeltaBeta = dB, Phase = Ph)

dir.create("results/figures", showWarnings=FALSE)

cat("Plotting 1 (Parallel Coordinates)...\n")
p1 <- ggparcoord(df_plot, columns = 3:5, groupColumn = "Model", alphaLines = 0.2) +
    theme_minimal() +
    scale_color_manual(values=c("Baseline Wald"="#E69F00", "Reg. Hybrid"="#56B4E9")) +
    labs(title="Global 3D Topological Projection (N=128)", x="Objective Space Constraint", y="Normalized Error Vector")
ggsave("results/figures/magi_parallel_coordinates_reg.png", p1, width=8, height=6)

cat("Plotting 2 (Bivariate Pareto)...\n")
hulls <- df_plot %>% group_by(Model) %>% slice(chull(W1, Phase))
p2 <- ggplot(df_plot, aes(x=W1, y=Phase, color=Model)) +
    geom_point(alpha=0.6) +
    geom_polygon(data=hulls, aes(fill=Model), alpha=0.2) +
    theme_minimal() +
    scale_color_manual(values=c("Baseline Wald"="#E69F00", "Reg. Hybrid"="#56B4E9")) +
    scale_fill_manual(values=c("Baseline Wald"="#E69F00", "Reg. Hybrid"="#56B4E9")) +
    labs(title="Bivariate Pareto Topology (W1 vs Phase)", x="Topological Error (W1)", y="Chronological Error (Phase)")
ggsave("results/figures/magi_bivariate_pareto_reg.png", p2, width=8, height=6)

cat("Plotting 3 (Polar Footprint Sub 42)...\n")
Z_ref <- c( W1 = max(df_plot$W1, na.rm=TRUE)*1.1, DeltaBeta = max(df_plot$DeltaBeta, na.rm=TRUE)*1.1, Phase = max(df_plot$Phase, na.rm=TRUE)*1.1 )
df_radar_scaled <- df_plot %>% filter(SubjectID == 42) %>% mutate(
    W1 = W1 / Z_ref["W1"], DeltaBeta = DeltaBeta / Z_ref["DeltaBeta"], Phase = Phase / Z_ref["Phase"]
)
df_polar <- df_radar_scaled %>% pivot_longer(cols=c(W1, DeltaBeta, Phase), names_to="Objective", values_to="Value")
df_polar <- df_polar %>% mutate(Objective = factor(Objective, levels=c("W1", "DeltaBeta", "Phase")))
p3 <- ggplot(df_polar, aes(x=Objective, y=Value, group=Model, color=Model, fill=Model)) +
    geom_polygon(alpha=0.3) +
    geom_point(size=2) +
    coord_polar() +
    theme_minimal() +
    scale_y_continuous(limits=c(0, NA)) +
    scale_color_manual(values=c("Baseline Wald"="#E69F00", "Reg. Hybrid"="#56B4E9")) +
    scale_fill_manual(values=c("Baseline Wald"="#E69F00", "Reg. Hybrid"="#56B4E9")) +
    labs(title="Subject-Level Error Footprint (Sub 42)", y="Normalized Error Magnitude")
ggsave("results/figures/magi_polar_footprint_reg.png", p3, width=6, height=6)

cat("Plotting 4 (ECDF Epsilon)...\n")
eps_list <- lapply(unique(df_plot$SubjectID), function(s) {
    sub <- df_plot %>% filter(SubjectID == s)
    if(nrow(sub) == 2) {
        base_obj <- as.numeric(sub[sub$Model=="Baseline Wald", c("W1", "DeltaBeta", "Phase")])
        term_obj <- as.numeric(sub[sub$Model=="Reg. Hybrid", c("W1", "DeltaBeta", "Phase")])
        data.frame(SubjectID=s, Epsilon=max(term_obj - base_obj))
    } else { NULL }
})
df_eps <- bind_rows(eps_list)

p4 <- ggplot(df_eps, aes(x=Epsilon)) +
    stat_ecdf(geom="step", color="#D55E00", linewidth=1.2) +
    theme_minimal() +
    labs(title="Population Dominance Distribution (ECDF of Epsilon-indicator)",
         x="Translation Magnitude Required for Baseline Parity", y="Cumulative Proportion of Population")
ggsave("results/figures/magi_ecdf_epsilon_reg.png", p4, width=8, height=6)

cat("--- PROJECTIONS COMPLETE ---\n")
