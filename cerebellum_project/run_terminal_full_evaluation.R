pacman::p_load(tidyverse, Rcpp, patchwork, lme4, lmerTest)

Rcpp::sourceCpp("src/models/epoch9_qperturbed_wald.cpp") 
Rcpp::sourceCpp("src/models/magi_terminal_decoupled_hybrid.cpp") 

dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types = FALSE)
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>% 
    group_by(participant_id) %>% 
    mutate(RT = (ttr - ttp) / 1000, 
           Boundary = ifelse(Resp == 2, 1, 0),
           CumulativeFatigue = row_number()) %>% 
    ungroup() %>% filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(`F`))
dat_clean <- dat_clean %>% mutate(participant_idx = as.integer(as.factor(participant_id)))

S <- 30
d_sub <- dat_clean %>% filter(participant_idx <= S)
hyper <- c(0.01, 1.00, 2.0)

cat("1. Fitting Baseline across N=30...\n")
obj_base <- function(phi) {
    err <- 0
    for(s in 1:S) {
        d <- d_sub %>% filter(participant_idx == s)
        sim <- extract_baseline_wald_sim(phi, d$Boundary+1, d$`F`, d$RT)
        err <- err + mean(abs(sort(sim) - sort(d$RT)))
    }
    return(err / S)
}
res_base <- optim(rep(0, 4), obj_base, method="Nelder-Mead", control=list(maxit=100))

cat("2. Fitting Terminal Model across N=30...\n")
obj_var <- function(phi) {
    err <- 0
    for(s in 1:S) {
        d <- d_sub %>% filter(participant_idx == s)
        sim <- extract_epoch10_2_hybrid(phi, hyper, d$Boundary+1, d$`F`, d$RT)
        err <- err + mean(abs(sort(sim) - sort(d$RT)))
    }
    return(err / S)
}
res_var <- optim(rep(0, 12), obj_var, method="Nelder-Mead", control=list(maxit=100))

cat("3. Generating Global Distributions...\n")
rt_base_all <- numeric(0); rt_var_all <- numeric(0); rt_emp_all <- numeric(0)
lmer_data <- data.frame(); w1_data <- data.frame()

for(s in 1:S) {
    d <- d_sub %>% filter(participant_idx == s)
    sim_base <- extract_baseline_wald_sim(res_base$par, d$Boundary+1, d$`F`, d$RT)
    sim_var <- extract_epoch10_2_hybrid(res_var$par, hyper, d$Boundary+1, d$`F`, d$RT)
    
    rt_base_all <- c(rt_base_all, sim_base)
    rt_var_all <- c(rt_var_all, sim_var)
    rt_emp_all <- c(rt_emp_all, d$RT)
    
    w1_base <- mean(abs(sort(sim_base) - sort(d$RT)))
    w1_var  <- mean(abs(sort(sim_var) - sort(d$RT)))
    
    lmer_data <- bind_rows(lmer_data, data.frame(SubjectID=s, CumulativeFatigue=d$CumulativeFatigue, EvidenceDrift=sim_base, Variant="Baseline"))
    lmer_data <- bind_rows(lmer_data, data.frame(SubjectID=s, CumulativeFatigue=d$CumulativeFatigue, EvidenceDrift=sim_var, Variant="V11_2_Decoupled"))
    
    w1_data <- bind_rows(w1_data, data.frame(SubjectID=s, Seed=sample(1:100,1), W1=w1_base, Variant="Baseline"))
    w1_data <- bind_rows(w1_data, data.frame(SubjectID=s, Seed=sample(1:100,1), W1=w1_var, Variant="V11_2_Decoupled"))
}

cat("4. Rendering Density Plot...\n")
df_dens <- data.frame(
    RT = c(rt_emp_all, rt_base_all, rt_var_all),
    Source = factor(rep(c("Empirical", "Baseline (Wald)", "Terminal Hybrid (Decoupled)"), each=length(rt_emp_all)), 
                    levels=c("Empirical", "Baseline (Wald)", "Terminal Hybrid (Decoupled)"))
) %>% filter(RT < 2.5) 

p1 <- ggplot(df_dens, aes(x = RT, fill = Source, color = Source)) +
    geom_density(alpha = 0.4, linewidth=1) +
    scale_fill_manual(values=c("black", "firebrick", "dodgerblue")) +
    scale_color_manual(values=c("black", "firebrick", "dodgerblue")) +
    theme_minimal() +
    labs(title = "Empirical vs Predicted RT Distributions (Global N=30)",
         subtitle = "Baseline Wald vs. Terminal Decoupled Hybrid Architecture",
         x = "Reaction Time (s)", y = "Density") +
    theme(legend.position="bottom", text=element_text(size=14))

artifact_dir <- "C:/Users/DCCS5/.gemini/antigravity/brain/1d8f9958-fd49-4502-b57b-97a7887eb7ad"
ggsave(file.path(artifact_dir, "magi_terminal_rt_distributions_N30.png"), p1, width=9, height=6, dpi=300)

cat("5. Rendering Global Q-Q Identity Plot...\n")
df_qq <- data.frame(
    Empirical = rep(sort(rt_emp_all), 2),
    Predicted = c(sort(rt_base_all), sort(rt_var_all)),
    Model = factor(rep(c("Baseline (Wald)", "Terminal Hybrid (Decoupled)"), each=length(rt_emp_all)), 
                   levels=c("Baseline (Wald)", "Terminal Hybrid (Decoupled)"))
)

p2 <- ggplot(df_qq, aes(x = Empirical, y = Predicted, color = Model)) +
    geom_line(linewidth=1.5, alpha=0.8) +
    geom_abline(slope = 1, intercept = 0, color = "black", linetype = "dashed", linewidth=1.2) +
    scale_color_manual(values=c("firebrick", "dodgerblue")) +
    theme_minimal() +
    coord_fixed(ratio=1, xlim=c(0, 2.5), ylim=c(0, 2.5)) +
    labs(title = "Global Q-Q Identity Plot (N=30)",
         subtitle = "Unified cumulative trace across all participants' trials.",
         x = "Empirical Reaction Time (s)", y = "Predicted Reaction Time (s)") +
    theme(legend.position="bottom", text=element_text(size=14))

ggsave(file.path(artifact_dir, "magi_qq_identity_plot_N30.png"), p2, width=7, height=7, dpi=300)

cat("6. Executing LMM Verification (N=30)...\n")
lmer_data$Variant <- as.factor(lmer_data$Variant)
lmer_data$Variant <- relevel(lmer_data$Variant, ref="Baseline")
w1_data$Variant <- as.factor(w1_data$Variant)
w1_data$Variant <- relevel(w1_data$Variant, ref="Baseline")

cat("\n--- LMER Test A: Goodness-of-Fit (W1 Reduction) ---\n")
mod_A <- lmer(W1 ~ Variant + (1 | SubjectID) + (1 | Seed), data = w1_data)
print(summary(mod_A)$coefficients)

cat("\n--- LMER Test B: Sequential Fatigue Dynamics (Beta Enhancement) ---\n")
lmer_data$CumulativeFatigue_s <- scale(lmer_data$CumulativeFatigue)
mod_B <- lmer(EvidenceDrift ~ CumulativeFatigue_s * Variant + (1 + CumulativeFatigue_s | SubjectID), data = lmer_data, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=100000)))
print(summary(mod_B)$coefficients)

cat("\nPipeline Complete.\n")
