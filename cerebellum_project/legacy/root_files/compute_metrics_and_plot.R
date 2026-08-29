pacman::p_load(tidyverse, Rcpp, cmaes, parallel, yardstick, DescTools, patchwork)

CORES <- parallel::detectCores()
hyper <- c(0.01, 1.00, 2.0)

Rcpp::sourceCpp("extract_joint.cpp")

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

eval_model_obj <- function(sim, d) {
    if(any(is.na(sim))) return(1e6)
    w1 <- mean(abs(sort(sim) - sort(d$RT)))
    emp_b <- get_spectral_beta(d$RT)
    sim_b <- get_spectral_beta(sim)
    db <- abs(emp_b - sim_b)
    if(is.na(db)) db <- 1.0
    return(w1 + db)
}

run_subj <- function(s_idx) {
    d <- d_list[[s_idx]]
    
    # 1. Baseline Wald
    obj_w <- function(p) eval_model_obj(extract_base_joint(p, d$Boundary+1, d$Reward, d$RT)[,1], d)
    res_w <- cma_es(rep(0, 4), obj_w, control=list(maxit=100, sigma=0.5))
    out_w <- extract_base_joint(res_w$par, d$Boundary+1, d$Reward, d$RT)
    
    # 2. Reg Hybrid
    lam <- 0.10; alp <- 0.5
    obj_h <- function(p) {
        sim <- extract_hybrid_joint(p, hyper, d$Boundary+1, d$Reward, d$RT)[,1]
        bc <- eval_model_obj(sim, d)
        pcb <- p[5:12]
        return(bc + lam * (alp * sum(abs(pcb)) + (1-alp)*sum(pcb^2)))
    }
    res_h <- cma_es(rep(0, 12), obj_h, control=list(maxit=100, sigma=0.5))
    out_h <- extract_hybrid_joint(res_h$par, hyper, d$Boundary+1, d$Reward, d$RT)
    
    # out_w and out_h columns: 1 = rt_sim, 2 = rt_exp, 3 = P(Choice = 1)
    # Compute Metrics
    truth <- factor(d$Boundary, levels=c("1", "0"))
    
    calc_mets <- function(probs, rt_pred) {
        if(any(is.na(probs))) probs[is.na(probs)] <- 0.5
        df <- data.frame(truth=truth, p1=probs, p0=1-probs)
        df$pred_class <- factor(ifelse(df$p1 > 0.5, "1", "0"), levels=c("1", "0"))
        
        pr <- pr_auc(df, truth, p1)$.estimate
        roc <- roc_auc(df, truth, p1)$.estimate
        mcc <- mcc(df, truth, pred_class)$.estimate
        brier <- brier_class(df, truth, p1)$.estimate
        rmse <- sqrt(mean((rt_pred - d$RT)^2))
        return(c(PR_AUC=pr, ROC_AUC=roc, MCC=mcc, Brier=brier, RT_RMSE=rmse))
    }
    
    m_w <- calc_mets(out_w[,3], out_w[,1])
    m_h <- calc_mets(out_h[,3], out_h[,1])
    
    # We also return a downsampled subset of RTs for plotting (first 100 trials)
    tr_len <- min(100, nrow(d))
    plot_df <- data.frame(
        SubjectID = s_idx,
        Trial = 1:tr_len,
        Empirical_RT = d$RT[1:tr_len],
        Wald_RT = out_w[1:tr_len, 1],
        Hybrid_RT = out_h[1:tr_len, 1]
    )
    
    return(list(
        mets = data.frame(
            SubjectID = s_idx,
            Model = c("Baseline Wald", "Reg. Hybrid"),
            PR_AUC = c(m_w[1], m_h[1]),
            ROC_AUC = c(m_w[2], m_h[2]),
            MCC = c(m_w[3], m_h[3]),
            Brier = c(m_w[4], m_h[4]),
            RT_RMSE = c(m_w[5], m_h[5])
        ),
        plot_df = plot_df
    ))
}

cat("Running Metric Extraction & Optimization...\n")
res_all <- mclapply(1:S, run_subj, mc.cores = CORES)

met_list <- lapply(res_all, function(x) x$mets)
plot_list <- lapply(res_all, function(x) x$plot_df)
df_mets <- bind_rows(met_list)
df_plot <- bind_rows(plot_list)

# Save metrics summary
sink("results/tables/magi_classification_metrics.txt")
cat("=== CLASSIFICATION & PREDICTION METRICS (N=128) ===\n")
print(df_mets %>% group_by(Model) %>% summarize(
    Mean_PR_AUC = mean(PR_AUC, na.rm=T),
    Mean_ROC_AUC = mean(ROC_AUC, na.rm=T),
    Mean_MCC = mean(MCC, na.rm=T),
    Mean_Brier = mean(Brier, na.rm=T),
    Mean_RT_RMSE = mean(RT_RMSE, na.rm=T)
))
sink()

# Generate RT Plot (Sub 42 as representative)
dir.create("results/figures", showWarnings=F)
p_df <- df_plot %>% filter(SubjectID == 42)
p1 <- ggplot(p_df, aes(x=Trial)) +
    geom_line(aes(y=Empirical_RT), color="black", size=1, alpha=0.5) +
    geom_line(aes(y=Wald_RT), color="#E69F00", size=0.8) +
    theme_minimal() + labs(title="Baseline Wald vs Empirical RT", y="Reaction Time (s)")

p2 <- ggplot(p_df, aes(x=Trial)) +
    geom_line(aes(y=Empirical_RT), color="black", size=1, alpha=0.5) +
    geom_line(aes(y=Hybrid_RT), color="#56B4E9", size=0.8) +
    theme_minimal() + labs(title="Reg. Hybrid vs Empirical RT", y="Reaction Time (s)")

p_final <- p1 / p2
ggsave("results/figures/magi_rt_prediction_plot.png", p_final, width=10, height=6)

cat("--- COMPUTATION COMPLETE ---\n")
