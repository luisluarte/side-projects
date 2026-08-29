pacman::p_load(tidyverse, cmdstanr, posterior, ggplot2, bayesplot, yardstick, RWiener, transport, loo, ggpubr, pROC, PRROC)
theme_set(theme_pubr() + theme(text = element_text(family = "sans"), axis.title = element_text(face="plain")))

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

cat("Compiling and generating quantities...\n")
mod_bvk_gq <- cmdstan_model("src/models/bvk_continuous_gq.stan")
mod_q_gq <- cmdstan_model("src/models/q_learning_ddm_gq.stan")
gq_bvk <- mod_bvk_gq$generate_quantities(fitted_params = fit_bvk, data = stan_data, parallel_chains = 4)
gq_q <- mod_q_gq$generate_quantities(fitted_params = fit_q, data = stan_data, parallel_chains = 4)

ll_bvk <- gq_bvk$draws("log_lik", format = "array")
ll_q <- gq_q$draws("log_lik", format = "array")
pu_bvk <- apply(gq_bvk$draws("p_upper", format = "array"), 3, mean)
pu_q <- apply(gq_q$draws("p_upper", format = "array"), 3, mean)
ll_bvk_mean <- apply(ll_bvk, 3, mean)
ll_q_mean <- apply(ll_q, 3, mean)

cat("Computing LOO...\n")
inject_unconditional_jitter <- function(ll_array) { return(ll_array + array(rnorm(length(ll_array), 0, 1e-5), dim = dim(ll_array))) }
loo_bvk <- loo(inject_unconditional_jitter(ll_bvk))
loo_q <- loo(inject_unconditional_jitter(ll_q))
comp <- loo_compare(loo_bvk, loo_q)
weights <- loo_model_weights(list(dual_kernel = loo_bvk, q_learning = loo_q), method = "stacking")

pdf("figures/2_performance_loo.pdf", width = 11, height = 8.5)
df_comp <- as.data.frame(comp)
df_comp$model <- tolower(rownames(df_comp))
print(ggplot(df_comp, aes(x = model, y = elpd_diff, fill = model)) + geom_bar(stat="identity") + geom_errorbar(aes(ymin = elpd_diff - se_diff, ymax = elpd_diff + se_diff), width=0.2) + scale_fill_viridis_d() + labs(title="psis-loo elpd difference", y="elpd diff", x="model") + theme(legend.position="none"))
dev.off()

pdf("figures/2_performance_weights.pdf", width = 11, height = 8.5)
df_w <- data.frame(model = c("dual_kernel", "q_learning"), weight = as.numeric(weights))
print(ggplot(df_w, aes(x = model, y = weight, fill = model)) + geom_bar(stat="identity") + scale_fill_viridis_d() + labs(title="stacking weights", y="weight", x="model") + theme(legend.position="none"))
dev.off()

cat("Computing multi-metrics...\n")
dat_clean$p_upper_bvk <- pu_bvk
dat_clean$p_upper_q <- pu_q
dat_clean$ll_bvk <- ll_bvk_mean
dat_clean$ll_q <- ll_q_mean
dat_clean$switch <- c(0, ifelse(diff(dat_clean$Boundary) != 0, 1, 0))
dat_clean$switch[stan_data$start_idx] <- NA 

dat_clean <- dat_clean %>% group_by(participant_id) %>% mutate(trial_idx = row_number()) %>% ungroup()
pdf("figures/3_curvature.pdf", width = 11, height = 8.5)
curv_df <- dat_clean %>% select(trial_idx, ll_bvk, ll_q) %>% pivot_longer(cols=c(ll_bvk, ll_q), names_to="model", values_to="ll") %>% mutate(model = ifelse(model=="ll_bvk", "dual_kernel", "q_learning"))
print(ggplot(curv_df, aes(x=trial_idx, y=ll, color=model, fill=model)) + geom_smooth(method="gam") + scale_color_viridis_d() + scale_fill_viridis_d() + labs(title="likelihood curvature across trials", x="trial index", y="pointwise log-likelihood"))
dev.off()

switch_df <- dat_clean %>% filter(!is.na(switch))
pr_bvk <- pr.curve(scores.class0 = switch_df$p_upper_bvk[switch_df$switch==1], scores.class1 = switch_df$p_upper_bvk[switch_df$switch==0], curve=TRUE)
pr_q <- pr.curve(scores.class0 = switch_df$p_upper_q[switch_df$switch==1], scores.class1 = switch_df$p_upper_q[switch_df$switch==0], curve=TRUE)
roc_bvk <- roc(switch_df$switch, switch_df$p_upper_bvk)
roc_q <- roc(switch_df$switch, switch_df$p_upper_q)

# Calculate MCC safely
calculate_mcc <- function(truth, pred) {
  tp <- sum(truth == 1 & pred == 1)
  tn <- sum(truth == 0 & pred == 0)
  fp <- sum(truth == 0 & pred == 1)
  fn <- sum(truth == 1 & pred == 0)
  num <- (tp * tn) - (fp * fn)
  den <- sqrt(as.numeric(tp + fp) * as.numeric(tp + fn) * as.numeric(tn + fp) * as.numeric(tn + fn))
  if (den == 0) return(0)
  return(num / den)
}

mcc_bvk <- calculate_mcc(switch_df$switch, ifelse(switch_df$p_upper_bvk > 0.5, 1, 0))
mcc_q <- calculate_mcc(switch_df$switch, ifelse(switch_df$p_upper_q > 0.5, 1, 0))

pdf("figures/3_metrics.pdf", width = 11, height = 8.5)
met_df <- data.frame(model = rep(c("dual_kernel", "q_learning"), 3), metric = c("pr-auc (switch)", "pr-auc (switch)", "roc-auc", "roc-auc", "mcc", "mcc"), value = c(pr_bvk$auc.integral, pr_q$auc.integral, as.numeric(roc_bvk$auc), as.numeric(roc_q$auc), mcc_bvk, mcc_q))
print(ggplot(met_df, aes(x=metric, y=value, fill=model)) + geom_bar(stat="identity", position="dodge") + scale_fill_viridis_d() + labs(title="predictive metrics for switch trials", y="metric value"))
dev.off()

cat("PPC...\n")
pdf("figures/4_ppc_rt.pdf", width = 11, height = 8.5)
print(ggplot(dat_clean, aes(x=RT, fill=as.factor(switch))) + geom_density(alpha=0.5) + scale_fill_viridis_d(name="switch") + labs(title="empirical rt distribution (switch vs stay)", x="rt", y="density"))
dev.off()

pdf("figures/4_ppc_proportions.pdf", width = 11, height = 8.5)
prop_df <- data.frame(type = factor(c("empirical", "dual_kernel_pred", "q_learning_pred"), levels=c("empirical", "dual_kernel_pred", "q_learning_pred")), prop = c(mean(dat_clean$Boundary), mean(dat_clean$p_upper_bvk), mean(dat_clean$p_upper_q)))
print(ggplot(prop_df, aes(x=type, y=prop, fill=type)) + geom_bar(stat="identity") + scale_fill_viridis_d() + labs(title="overall choice proportions", y="p(choice=1)") + theme(legend.position="none"))
dev.off()

cat("Done!\n")
