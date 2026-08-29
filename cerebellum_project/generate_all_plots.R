pacman::p_load(tidyverse, cmdstanr, posterior, pROC, PRROC, ggpubr, loo, matrixStats)

dir.create("results/figures", showWarnings = FALSE)

theme_set(theme_pubr() + theme(text = element_text(family = "sans"), axis.title = element_text(face = "bold"), strip.text = element_text(face="bold")))

dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types = FALSE)
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>% group_by(participant_id) %>% mutate(RT = (ttr - ttp) / 1000, ITI = (ttp - lag(ttF)) / 1000, F_dur = (ttF - ttr) / 1000, Boundary = ifelse(Resp == 2, 1, 0)) %>% mutate(ITI = ifelse(is.na(ITI), mean(ITI, na.rm = TRUE), ITI)) %>% ungroup() %>% filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(`F`))
set.seed(420)
pid_sample <- sample(unique(dat_clean$participant_id), size = 30)
dat_clean <- dat_clean %>% filter(participant_id %in% pid_sample)

cat("loading fits...\n")
fit_base <- read_rds("results/fit_q_complete.rds")
fit_gating <- read_rds("results/fit_full_gating_complete.rds")

# 2. Performance Plots (PSIS-LOO)
cat("computing LOO...\n")
stan_data <- read_rds("results/stan_data.rds")
mod_base_gq <- cmdstan_model("src/models/q_learning_ddm_gq.stan")
gq_base <- mod_base_gq$generate_quantities(fitted_params = fit_base, data = stan_data, parallel_chains = 4)
ll_base <- gq_base$draws("log_lik")
loo_base <- loo(ll_base, r_eff = relative_eff(exp(ll_base)))

mod_gating_gq <- cmdstan_model("src/models/bvk_full_gating_gq.stan")
gq_gating <- mod_gating_gq$generate_quantities(fitted_params = fit_gating, data = stan_data, parallel_chains = 4)
ll_gating <- gq_gating$draws("log_lik")
loo_gating <- loo(ll_gating, r_eff = relative_eff(exp(ll_gating)))

# 3. Likelihood Curvature
# CORRECT CALCULATION: log(mean(likelihood)) = log_sum_exp(log_lik) - log(S)
ll_mat_base <- as_draws_matrix(ll_base)
ll_mat_gating <- as_draws_matrix(ll_gating)

ll_base_mean <- colLogSumExps(ll_mat_base) - log(nrow(ll_mat_base))
ll_gating_mean <- colLogSumExps(ll_mat_gating) - log(nrow(ll_mat_gating))

subject_vec <- integer(stan_data$N)
for (i in 1:stan_data$S) {
  subject_vec[stan_data$start_idx[i]:stan_data$end_idx[i]] <- i
}

df_ll <- tibble(trial_idx = 1:length(ll_base_mean), base = ll_base_mean, gating = ll_gating_mean, subject = subject_vec)
df_ll <- df_ll %>% group_by(subject) %>% mutate(trial = row_number()) %>% ungroup()
df_ll_sum <- df_ll %>% group_by(trial) %>% summarise(mean_base = mean(base), mean_gating = mean(gating)) %>% pivot_longer(cols=c(mean_base, mean_gating), names_to="model", values_to="ll") %>% mutate(model = str_remove(model, "mean_"))

p3 <- ggplot(df_ll_sum, aes(x = trial, y = ll, color = model)) + geom_smooth(span=0.3, se=FALSE, size=1.5) + scale_color_viridis_d(option="viridis", end=0.8) + labs(title = "trial-by-trial log likelihood curvature", subtitle = "smoothed average across subjects (corrected)", x = "trial number", y = "log predictive density")
pdf("results/figures/3_likelihood_curvature.pdf", width = 8, height = 5)
print(p3)
dev.off()

cat("done generating likelihood plot!\n")
