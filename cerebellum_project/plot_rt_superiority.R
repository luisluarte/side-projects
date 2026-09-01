pacman::p_load(tidyverse, cmdstanr, posterior, ggplot2, ggpubr, loo)
theme_set(theme_pubr() + theme(text = element_text(family = "sans"), axis.title = element_text(face="plain")))

cat("Loading fits and data...\n")
fit_bvk <- read_rds("results/fit_bvk_complete.rds")
fit_q <- read_rds("results/fit_q_complete.rds")

dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types = FALSE)
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>% group_by(participant_id) %>% mutate(RT = (ttr - ttp) / 1000, ITI = (ttp - lag(ttF)) / 1000, F_dur = (ttF - ttr) / 1000, Boundary = ifelse(Resp == 2, 1, 0)) %>% mutate(ITI = ifelse(is.na(ITI), mean(ITI, na.rm = TRUE), ITI)) %>% ungroup() %>% filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(`F`))
set.seed(420)
pid_sample <- sample(unique(dat_clean$participant_id), size = 30)
dat_clean <- dat_clean %>% filter(participant_id %in% pid_sample)
subject_counts <- dat_clean %>% group_by(participant_id) %>% summarise(count = n()) %>% mutate(end_idx = cumsum(count), start_idx = end_idx - count + 1)
min_rt_df <- dat_clean %>% group_by(participant_id) %>% summarise(min_rt = min(RT))
stan_data <- list(N = nrow(dat_clean), S = nrow(subject_counts), start_idx = subject_counts$start_idx, end_idx = subject_counts$end_idx, choice = dat_clean$Boundary, rt = dat_clean$RT, reward = dat_clean$`F`, iti = dat_clean$ITI, f_dur = dat_clean$F_dur, min_rt = min_rt_df$min_rt, N_MF = 5, grainsize = 1)

system("git checkout src/models/bvk_continuous_gq.stan src/models/q_learning_ddm_gq.stan")

mod_bvk_gq <- cmdstan_model("src/models/bvk_continuous_gq.stan")
mod_q_gq <- cmdstan_model("src/models/q_learning_ddm_gq.stan")
gq_bvk <- mod_bvk_gq$generate_quantities(fitted_params = fit_bvk, data = stan_data, parallel_chains = 4)
gq_q <- mod_q_gq$generate_quantities(fitted_params = fit_q, data = stan_data, parallel_chains = 4)

ll_bvk <- apply(gq_bvk$draws("log_lik", format = "array"), 3, mean)
ll_q <- apply(gq_q$draws("log_lik", format = "array"), 3, mean)

dat_clean$ll_bvk <- ll_bvk
dat_clean$ll_q <- ll_q
dat_clean$ll_diff <- dat_clean$ll_bvk - dat_clean$ll_q

dat_clean <- dat_clean %>% mutate(rt_tertile = ntile(RT, 3), rt_category = factor(rt_tertile, levels=c(1,2,3), labels=c("fast", "medium", "slow")))

model_lm <- lm(ll_diff ~ poly(RT, 2), data = dat_clean)

pdf("figures/5_rt_superiority.pdf", width = 14, height = 7)
p1 <- ggplot(dat_clean, aes(x = RT, y = ll_diff)) +
  geom_point(alpha = 0.1, color="gray50") +
  geom_smooth(method = "gam", formula = y ~ s(x, bs = "cs"), color = "#21908CFF", linewidth=1.5) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  coord_cartesian(ylim = c(quantile(dat_clean$ll_diff, 0.05), quantile(dat_clean$ll_diff, 0.95))) +
  labs(title = "predictive advantage across reaction times", subtitle = "positive y-axis indicates dual-kernel superiority", x = "reaction time (s)", y = "log-likelihood difference (dual - q)") +
  theme_pubr()

p2 <- ggboxplot(dat_clean, x = "rt_category", y = "ll_diff", fill = "rt_category", palette = "viridis", outlier.shape = NA) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  coord_cartesian(ylim = c(quantile(dat_clean$ll_diff, 0.05), quantile(dat_clean$ll_diff, 0.95))) +
  stat_compare_means(comparisons = list(c("fast", "medium"), c("medium", "slow"), c("fast", "slow")), label = "p.signif") +
  stat_compare_means(label.y = quantile(dat_clean$ll_diff, 0.95)) +
  labs(title = "log-likelihood difference by rt tertile", x = "rt category", y = "log-likelihood difference") +
  theme(legend.position = "none")

print(ggarrange(p1, p2, ncol = 2, nrow = 1))
dev.off()

cat("Stats summary:\n")
print(summary(model_lm))
