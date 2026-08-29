pacman::p_load(tidyverse, Rcpp, patchwork)

Rcpp::sourceCpp("src/models/epoch9_qperturbed_wald.cpp") # loads baseline function
Rcpp::sourceCpp("src/models/epoch10_2_wald_decoupled.cpp") # loads variant 11.2 function

# Save to the final name for the user
file.copy("src/models/epoch10_2_wald_decoupled.cpp", "src/models/magi_terminal_decoupled_hybrid.cpp", overwrite=TRUE)

dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types = FALSE)
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>% 
    group_by(participant_id) %>% 
    mutate(RT = (ttr - ttp) / 1000, 
           Boundary = ifelse(Resp == 2, 1, 0)) %>% 
    ungroup() %>% filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(`F`))
dat_clean <- dat_clean %>% mutate(participant_idx = as.integer(as.factor(participant_id)))

d_sub <- dat_clean %>% filter(participant_idx <= 5)
hyper <- c(0.01, 1.00, 2.0)

cat("Fitting Baseline...\n")
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

cat("Fitting Terminal Model...\n")
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
    rt_base_all <- c(rt_base_all, extract_baseline_wald_sim(res_base$par, d$Boundary+1, d$`F`, d$RT))
    rt_var_all <- c(rt_var_all, extract_epoch10_2_hybrid(res_var$par, hyper, d$Boundary+1, d$`F`, d$RT))
    rt_emp_all <- c(rt_emp_all, d$RT)
}

df_plot <- data.frame(
    RT = c(rt_emp_all, rt_base_all, rt_var_all),
    Source = factor(rep(c("Empirical", "Baseline (Wald)", "Terminal Hybrid (Decoupled)"), each=length(rt_emp_all)), 
                    levels=c("Empirical", "Baseline (Wald)", "Terminal Hybrid (Decoupled)"))
)

df_plot <- df_plot %>% filter(RT < 2.5) 

p <- ggplot(df_plot, aes(x = RT, fill = Source, color = Source)) +
    geom_density(alpha = 0.4, linewidth=1) +
    scale_fill_manual(values=c("black", "firebrick", "dodgerblue")) +
    scale_color_manual(values=c("black", "firebrick", "dodgerblue")) +
    theme_minimal() +
    labs(title = "Empirical vs Predicted RT Distributions",
         subtitle = "Baseline Wald vs. Terminal Decoupled Hybrid Architecture (N=5)",
         x = "Reaction Time (s)", y = "Density") +
    theme(legend.position="bottom", text=element_text(size=14))

artifact_dir <- "C:/Users/DCCS5/.gemini/antigravity/brain/1d8f9958-fd49-4502-b57b-97a7887eb7ad"
ggsave(file.path(artifact_dir, "magi_terminal_rt_distributions.png"), p, width=9, height=6, dpi=300)
cat("Plot generated successfully.\n")
