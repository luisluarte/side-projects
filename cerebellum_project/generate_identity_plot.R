pacman::p_load(tidyverse, Rcpp, patchwork)

Rcpp::sourceCpp("src/models/epoch9_qperturbed_wald.cpp") 
Rcpp::sourceCpp("src/models/magi_terminal_decoupled_hybrid.cpp") 

dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types = FALSE)
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>% 
    group_by(participant_id) %>% 
    mutate(RT = (ttr - ttp) / 1000, 
           Boundary = ifelse(Resp == 2, 1, 0)) %>% 
    ungroup() %>% filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(`F`))
dat_clean <- dat_clean %>% mutate(participant_idx = as.integer(as.factor(participant_id)))

d_sub <- dat_clean %>% filter(participant_idx <= 5)
hyper <- c(0.01, 1.00, 2.0)

obj_base <- function(phi) {
    rt_sim <- numeric(0); rt_emp <- numeric(0)
    for(s in 1:5) {
        d <- d_sub %>% filter(participant_idx == s)
        sim <- extract_baseline_wald_sim(phi, d$Boundary+1, d$`F`, d$RT)
        rt_sim <- c(rt_sim, sim); rt_emp <- c(rt_emp, d$RT)
    }
    return(mean(abs(sort(rt_sim) - sort(rt_emp))))
}
res_base <- optim(rep(0, 4), obj_base, method="Nelder-Mead", control=list(maxit=100))

obj_var <- function(phi) {
    rt_sim <- numeric(0); rt_emp <- numeric(0)
    for(s in 1:5) {
        d <- d_sub %>% filter(participant_idx == s)
        sim <- extract_epoch10_2_hybrid(phi, hyper, d$Boundary+1, d$`F`, d$RT)
        rt_sim <- c(rt_sim, sim); rt_emp <- c(rt_emp, d$RT)
    }
    return(mean(abs(sort(rt_sim) - sort(rt_emp))))
}
res_var <- optim(rep(0, 12), obj_var, method="Nelder-Mead", control=list(maxit=150))

rt_base_all <- numeric(0); rt_var_all <- numeric(0); rt_emp_all <- numeric(0)
for(s in 1:5) {
    d <- d_sub %>% filter(participant_idx == s)
    rt_base_all <- c(rt_base_all, sort(extract_baseline_wald_sim(res_base$par, d$Boundary+1, d$`F`, d$RT)))
    rt_var_all <- c(rt_var_all, sort(extract_epoch10_2_hybrid(res_var$par, hyper, d$Boundary+1, d$`F`, d$RT)))
    rt_emp_all <- c(rt_emp_all, sort(d$RT))
}

df_plot <- data.frame(
    Empirical = rep(rt_emp_all, 2),
    Predicted = c(rt_base_all, rt_var_all),
    Model = factor(rep(c("Baseline (Wald)", "Terminal Hybrid (Decoupled)"), each=length(rt_emp_all)), 
                   levels=c("Baseline (Wald)", "Terminal Hybrid (Decoupled)"))
)

p <- ggplot(df_plot, aes(x = Empirical, y = Predicted, color = Model)) +
    geom_point(alpha = 0.4, size=1.5) +
    geom_abline(slope = 1, intercept = 0, color = "black", linetype = "dashed", linewidth=1.2) +
    scale_color_manual(values=c("firebrick", "dodgerblue")) +
    theme_minimal() +
    coord_fixed(ratio=1, xlim=c(0, 2.5), ylim=c(0, 2.5)) +
    labs(title = "Q-Q Identity Plot: Empirical vs Predicted RT",
         subtitle = "The Decoupled Hybrid pulls the extreme right tail back to the identity (y=x) line.",
         x = "Empirical Reaction Time (s)", y = "Predicted Reaction Time (s)") +
    theme(legend.position="bottom", text=element_text(size=14))

artifact_dir <- "C:/Users/DCCS5/.gemini/antigravity/brain/1d8f9958-fd49-4502-b57b-97a7887eb7ad"
ggsave(file.path(artifact_dir, "magi_qq_identity_plot.png"), p, width=7, height=7, dpi=300)
cat("Plot generated successfully.\n")
