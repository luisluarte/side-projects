pacman::p_load(tidyverse, cmdstanr, posterior, pROC, PRROC)
fit_bvk <- read_rds("results/fit_bvk_complete.rds")
fit_q <- read_rds("results/fit_q_complete.rds")
summ_bvk <- fit_bvk$summary()
summ_q <- fit_q$summary()
bvk_params <- summ_bvk$mean; names(bvk_params) <- summ_bvk$variable
q_params <- summ_q$mean; names(q_params) <- summ_q$variable
dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types = FALSE)
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>% group_by(participant_id) %>% mutate(RT = (ttr - ttp) / 1000, ITI = (ttp - lag(ttF)) / 1000, F_dur = (ttF - ttr) / 1000, Boundary = ifelse(Resp == 2, 1, 0)) %>% mutate(ITI = ifelse(is.na(ITI), mean(ITI, na.rm = TRUE), ITI)) %>% ungroup() %>% filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(`F`))
set.seed(420)
pid_sample <- sample(unique(dat_clean$participant_id), size = 30)
dat_clean <- dat_clean %>% filter(participant_id %in% pid_sample)
subject_counts <- dat_clean %>% group_by(participant_id) %>% summarise(count = n()) %>% mutate(end_idx = cumsum(count), start_idx = end_idx - count + 1)

exact_mf_step <- function(dt, mf, tau_m, I_drive, N_MF) {
  mf_next <- numeric(N_MF)
  d <- mf - I_drive; x <- dt / tau_m; decay <- exp(-x)
  w <- numeric(N_MF); w[1] <- 1.0; for (i in 2:N_MF) { w[i] <- w[i-1] * x / (i - 1.0) }
  for (k in 1:N_MF) {
    conv_sum <- 0.0; for (j in 1:k) { conv_sum <- conv_sum + d[k - j + 1] * w[j] }
    mf_next[k] <- I_drive + conv_sum * decay
  }
  return(mf_next)
}
pu_bvk <- numeric(nrow(dat_clean))
for (s in 1:nrow(subject_counts)) {
  start_t <- subject_counts$start_idx[s]; end_t <- subject_counts$end_idx[s]
  alpha_ctx <- bvk_params[paste0("alpha_ctx[", s, "]")]; tau_m <- bvk_params[paste0("tau_m[", s, "]")]; eta_gc <- bvk_params[paste0("eta_gc[", s, "]")]; lambda_gc <- bvk_params[paste0("lambda_gc[", s, "]")]; theta_cb <- bvk_params[paste0("theta_cb[", s, "]")]; kappa_ctx <- bvk_params[paste0("kappa_ctx[", s, "]")]; gamma_suppress <- bvk_params[paste0("gamma_suppress[", s, "]")]; a_s <- bvk_params[paste0("a[", s, "]")]
  Q_ctx <- c(0.5, 0.5); mf_state <- rep(0.0, 5); w_gc1 <- rep(0.0, 5); w_gc2 <- rep(0.0, 5); w_mli1 <- rep(0.0, 5); w_mli2 <- rep(0.0, 5)
  for (t in start_t:end_t) {
    iti <- dat_clean$ITI[t]
    if (iti > 0.01) {
      mf_state <- exact_mf_step(iti, mf_state, tau_m, 0.0, 5); decay_gc <- exp(-lambda_gc * iti); decay_mli <- exp(-(lambda_gc * 1.5) * iti)
      w_gc1 <- w_gc1 * decay_gc; w_gc2 <- w_gc2 * decay_gc; w_mli1 <- w_mli1 * decay_mli; w_mli2 <- w_mli2 * decay_mli
    }
    Q_cb_1 <- sum(w_gc1 * mf_state) - sum(w_mli1 * mf_state); Q_cb_2 <- sum(w_gc2 * mf_state) - sum(w_mli2 * mf_state)
    delta_Q_ctx <- Q_ctx[2] - Q_ctx[1]; delta_Q_cb <- Q_cb_2 - Q_cb_1
    w_bias <- 0.5 + 0.45 * tanh(theta_cb * delta_Q_cb); conflict <- 0.5 * (1.0 - tanh(10.0 * delta_Q_ctx) * tanh(10.0 * delta_Q_cb))
    v_base <- kappa_ctx * delta_Q_ctx; v_eff <- v_base * exp(-gamma_suppress * conflict)
    if (abs(v_eff) < 1e-4) { v_eff <- ifelse(v_eff>=0, 1e-4, -1e-4) }
    pu_bvk[t] <- (1.0 - exp(-2.0 * v_eff * a_s * w_bias)) / (1.0 - exp(-2.0 * v_eff * a_s))
    ch <- dat_clean$Boundary[t]; rew <- dat_clean$`F`[t]
    Q_ctx[ch + 1] <- Q_ctx[ch + 1] + alpha_ctx * (rew - Q_ctx[ch + 1]); cb_pred <- ifelse(ch == 1, Q_cb_2, Q_cb_1)
    f_dur <- dat_clean$F_dur[t]
    if (f_dur > 0.01) {
      mf_state <- exact_mf_step(f_dur, mf_state, tau_m, rew, 5); l_gc <- lambda_gc + 1e-8; l_mli <- (lambda_gc * 1.5) + 1e-8
      dec_gc <- exp(-l_gc * f_dur); dec_mli <- exp(-l_mli * f_dur); int_gc <- (1.0 - dec_gc) / l_gc; int_mli <- (1.0 - dec_mli) / l_mli
      E_cb1 <- ifelse(ch == 0, rew - cb_pred, 0.0); E_cb2 <- ifelse(ch == 1, rew - cb_pred, 0.0)
      w_gc1 <- w_gc1 * dec_gc + mf_state * (eta_gc * E_cb1 * int_gc); w_gc2 <- w_gc2 * dec_gc + mf_state * (eta_gc * E_cb2 * int_gc)
      w_mli1 <- w_mli1 * dec_mli + mf_state * (-eta_gc * E_cb1 * int_mli); w_mli2 <- w_mli2 * dec_mli + mf_state * (-eta_gc * E_cb2 * int_mli)
    }
  }
}
pu_q <- numeric(nrow(dat_clean))
for (s in 1:nrow(subject_counts)) {
  start_t <- subject_counts$start_idx[s]; end_t <- subject_counts$end_idx[s]
  alpha_ctx <- q_params[paste0("alpha_ctx[", s, "]")]; kappa_ctx <- q_params[paste0("kappa_ctx[", s, "]")]; a_s <- q_params[paste0("a[", s, "]")]; w_bias <- q_params[paste0("w_bias[", s, "]")]
  Q_ctx <- c(0.5, 0.5)
  for (t in start_t:end_t) {
    delta_Q_ctx <- Q_ctx[2] - Q_ctx[1]; drift_sign <- ifelse(delta_Q_ctx >= 0, 1.0, -1.0)
    v_eff <- drift_sign * sqrt((kappa_ctx * delta_Q_ctx)^2 + 1e-4)
    if (abs(v_eff) < 1e-4) { v_eff <- ifelse(v_eff>=0, 1e-4, -1e-4) }
    pu_q[t] <- (1.0 - exp(-2.0 * v_eff * a_s * w_bias)) / (1.0 - exp(-2.0 * v_eff * a_s))
    ch <- dat_clean$Boundary[t]; rew <- dat_clean$`F`[t]
    Q_ctx[ch + 1] <- Q_ctx[ch + 1] + alpha_ctx * (rew - Q_ctx[ch + 1])
  }
}
dat_clean$p_upper_bvk <- pu_bvk
dat_clean$p_upper_q <- pu_q
dat_clean <- dat_clean %>% group_by(participant_id) %>% mutate(prev_Boundary = lag(Boundary), switch = ifelse(Boundary != prev_Boundary, 1, 0), p_switch_bvk = ifelse(prev_Boundary == 0, p_upper_bvk, 1.0 - p_upper_bvk), p_switch_q = ifelse(prev_Boundary == 0, p_upper_q, 1.0 - p_upper_q)) %>% ungroup()
switch_df <- dat_clean %>% filter(!is.na(switch))
calculate_mcc <- function(truth, pred) {
  tp <- sum(truth == 1 & pred == 1); tn <- sum(truth == 0 & pred == 0); fp <- sum(truth == 0 & pred == 1); fn <- sum(truth == 1 & pred == 0)
  num <- (tp * tn) - (fp * fn)
  den <- sqrt(as.numeric(tp + fp) * as.numeric(tp + fn) * as.numeric(tn + fp) * as.numeric(tn + fn))
  if (den == 0) return(0)
  return(num / den)
}
results <- list()
for(pid in unique(switch_df$participant_id)) {
  sub_df <- switch_df %>% filter(participant_id == pid)
  if(length(unique(sub_df$switch)) == 2) {
    roc_bvk <- roc(sub_df$switch, sub_df$p_switch_bvk, quiet = TRUE)$auc
    pr_bvk <- pr.curve(scores.class0 = sub_df$p_switch_bvk[sub_df$switch==1], scores.class1 = sub_df$p_switch_bvk[sub_df$switch==0])$auc.integral
    mcc_bvk <- calculate_mcc(sub_df$switch, ifelse(sub_df$p_switch_bvk > 0.5, 1, 0))
    roc_q <- roc(sub_df$switch, sub_df$p_switch_q, quiet = TRUE)$auc
    pr_q <- pr.curve(scores.class0 = sub_df$p_switch_q[sub_df$switch==1], scores.class1 = sub_df$p_switch_q[sub_df$switch==0])$auc.integral
    mcc_q <- calculate_mcc(sub_df$switch, ifelse(sub_df$p_switch_q > 0.5, 1, 0))
    results[[length(results) + 1]] <- data.frame(participant_id = pid, model = "dual_kernel", metric = c("roc-auc", "pr-auc", "mcc"), value = c(as.numeric(roc_bvk), pr_bvk, mcc_bvk))
    results[[length(results) + 1]] <- data.frame(participant_id = pid, model = "q_learning", metric = c("roc-auc", "pr-auc", "mcc"), value = c(as.numeric(roc_q), pr_q, mcc_q))
  }
}
final_df <- bind_rows(results)
sum_df <- final_df %>% group_by(metric, model) %>% summarize(mean_val = mean(value), .groups="drop") %>% pivot_wider(names_from = model, values_from = mean_val)
print(sum_df)
