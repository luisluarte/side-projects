pacman::p_load(tidyverse, cmdstanr, posterior, ggplot2, bayesplot, yardstick, loo, ggpubr, pROC, PRROC)
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

# WE DO NOT NEED TO GENERATE QUANTITIES FOR P_UPPER IF WE DON'T WANT TO MODIFY STAN. 
# BUT WE DO NEED LOG_LIK FOR PSIS-LOO. I will use the UNMODIFIED STAN files for log_lik.
git_checkout <- system("git checkout src/models/bvk_continuous_gq.stan src/models/q_learning_ddm_gq.stan")

mod_bvk_gq <- cmdstan_model("src/models/bvk_continuous_gq.stan")
mod_q_gq <- cmdstan_model("src/models/q_learning_ddm_gq.stan")
gq_bvk <- mod_bvk_gq$generate_quantities(fitted_params = fit_bvk, data = stan_data, parallel_chains = 4)
gq_q <- mod_q_gq$generate_quantities(fitted_params = fit_q, data = stan_data, parallel_chains = 4)

ll_bvk <- gq_bvk$draws("log_lik", format = "array")
ll_q <- gq_q$draws("log_lik", format = "array")

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

cat("Computing simulated metrics in R...\n")
# Extract posterior means
summ_bvk <- fit_bvk$summary()
summ_q <- fit_q$summary()
bvk_params <- summ_bvk$mean; names(bvk_params) <- summ_bvk$variable
q_params <- summ_q$mean; names(q_params) <- summ_q$variable

exact_mf_step <- function(dt, mf, tau_m, I_drive, N_MF) {
  mf_next <- numeric(N_MF)
  d <- mf - I_drive
  x <- dt / tau_m
  decay <- exp(-x)
  w <- numeric(N_MF)
  w[1] <- 1.0
  for (i in 2:N_MF) { w[i] <- w[i-1] * x / (i - 1.0) }
  for (k in 1:N_MF) {
    conv_sum <- 0.0
    for (j in 1:k) { conv_sum <- conv_sum + d[k - j + 1] * w[j] }
    mf_next[k] <- I_drive + conv_sum * decay
  }
  return(mf_next)
}

pu_bvk <- numeric(stan_data$N)
for (s in 1:stan_data$S) {
  start_t <- stan_data$start_idx[s]
  end_t <- stan_data$end_idx[s]
  
  alpha_ctx <- bvk_params[paste0("alpha_ctx[", s, "]")]
  tau_m <- bvk_params[paste0("tau_m[", s, "]")]
  eta_gc <- bvk_params[paste0("eta_gc[", s, "]")]
  lambda_gc <- bvk_params[paste0("lambda_gc[", s, "]")]
  theta_cb <- bvk_params[paste0("theta_cb[", s, "]")]
  kappa_ctx <- bvk_params[paste0("kappa_ctx[", s, "]")]
  gamma_suppress <- bvk_params[paste0("gamma_suppress[", s, "]")]
  a_s <- bvk_params[paste0("a[", s, "]")]
  
  Q_ctx <- c(0.5, 0.5)
  mf_state <- rep(0.0, 5)
  w_gc1 <- rep(0.0, 5)
  w_gc2 <- rep(0.0, 5)
  w_mli1 <- rep(0.0, 5)
  w_mli2 <- rep(0.0, 5)
  
  for (t in start_t:end_t) {
    iti <- stan_data$iti[t]
    if (iti > 0.01) {
      mf_state <- exact_mf_step(iti, mf_state, tau_m, 0.0, 5)
      decay_gc <- exp(-lambda_gc * iti)
      decay_mli <- exp(-(lambda_gc * 1.5) * iti)
      w_gc1 <- w_gc1 * decay_gc
      w_gc2 <- w_gc2 * decay_gc
      w_mli1 <- w_mli1 * decay_mli
      w_mli2 <- w_mli2 * decay_mli
    }
    
    Q_cb_1 <- sum(w_gc1 * mf_state) - sum(w_mli1 * mf_state)
    Q_cb_2 <- sum(w_gc2 * mf_state) - sum(w_mli2 * mf_state)
    delta_Q_ctx <- Q_ctx[2] - Q_ctx[1]
    delta_Q_cb <- Q_cb_2 - Q_cb_1
    w_bias <- 0.5 + 0.45 * tanh(theta_cb * delta_Q_cb)
    conflict <- 0.5 * (1.0 - tanh(10.0 * delta_Q_ctx) * tanh(10.0 * delta_Q_cb))
    v_base <- kappa_ctx * delta_Q_ctx
    v_eff <- v_base * exp(-gamma_suppress * conflict)
    if (abs(v_eff) < 1e-4) { v_eff <- ifelse(v_eff>=0, 1e-4, -1e-4) }
    
    p_up <- (1.0 - exp(-2.0 * v_eff * a_s * w_bias)) / (1.0 - exp(-2.0 * v_eff * a_s))
    pu_bvk[t] <- p_up
    
    ch <- stan_data$choice[t]
    rew <- stan_data$reward[t]
    RPE_ctx <- rew - Q_ctx[ch + 1]
    Q_ctx[ch + 1] <- Q_ctx[ch + 1] + alpha_ctx * RPE_ctx
    cb_pred <- ifelse(ch == 1, Q_cb_2, Q_cb_1)
    RPE_cb <- rew - cb_pred
    E_cb1 <- ifelse(ch == 0, RPE_cb, 0.0)
    E_cb2 <- ifelse(ch == 1, RPE_cb, 0.0)
    f_dur <- stan_data$f_dur[t]
    if (f_dur > 0.01) {
      mf_state <- exact_mf_step(f_dur, mf_state, tau_m, rew, 5)
      l_gc <- lambda_gc + 1e-8
      l_mli <- (lambda_gc * 1.5) + 1e-8
      dec_gc <- exp(-l_gc * f_dur)
      dec_mli <- exp(-l_mli * f_dur)
      int_gc <- (1.0 - dec_gc) / l_gc
      int_mli <- (1.0 - dec_mli) / l_mli
      w_gc1 <- w_gc1 * dec_gc + mf_state * (eta_gc * E_cb1 * int_gc)
      w_gc2 <- w_gc2 * dec_gc + mf_state * (eta_gc * E_cb2 * int_gc)
      w_mli1 <- w_mli1 * dec_mli + mf_state * (-eta_gc * E_cb1 * int_mli)
      w_mli2 <- w_mli2 * dec_mli + mf_state * (-eta_gc * E_cb2 * int_mli)
    }
  }
}

pu_q <- numeric(stan_data$N)
for (s in 1:stan_data$S) {
  start_t <- stan_data$start_idx[s]
  end_t <- stan_data$end_idx[s]
  alpha_ctx <- q_params[paste0("alpha_ctx[", s, "]")]
  kappa_ctx <- q_params[paste0("kappa_ctx[", s, "]")]
  a_s <- q_params[paste0("a[", s, "]")]
  w_bias <- q_params[paste0("w_bias[", s, "]")]
  Q_ctx <- c(0.5, 0.5)
  for (t in start_t:end_t) {
    delta_Q_ctx <- Q_ctx[2] - Q_ctx[1]
    drift_sign <- ifelse(delta_Q_ctx >= 0, 1.0, -1.0)
    v_eff <- drift_sign * sqrt((kappa_ctx * delta_Q_ctx)^2 + 1e-4)
    if (abs(v_eff) < 1e-4) { v_eff <- ifelse(v_eff>=0, 1e-4, -1e-4) }
    
    p_up <- (1.0 - exp(-2.0 * v_eff * a_s * w_bias)) / (1.0 - exp(-2.0 * v_eff * a_s))
    pu_q[t] <- p_up
    
    ch <- stan_data$choice[t]
    rew <- stan_data$reward[t]
    RPE <- rew - Q_ctx[ch + 1]
    Q_ctx[ch + 1] <- Q_ctx[ch + 1] + alpha_ctx * RPE
  }
}

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
