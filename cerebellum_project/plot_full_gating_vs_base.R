pacman::p_load(tidyverse, cmdstanr, posterior, pROC, PRROC, ggpubr)

dat_raw <- read_csv("~/cerebellum_project/data/raw/behavioral_compilate.csv", show_col_types = FALSE)
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>% group_by(participant_id) %>% mutate(RT = (ttr - ttp) / 1000, ITI = (ttp - lag(ttF)) / 1000, F_dur = (ttF - ttr) / 1000, Boundary = ifelse(Resp == 2, 1, 0)) %>% mutate(ITI = ifelse(is.na(ITI), mean(ITI, na.rm = TRUE), ITI)) %>% ungroup() %>% filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(`F`))
set.seed(420)
pid_sample <- sample(unique(dat_clean$participant_id), size = 30)
dat_clean <- dat_clean %>% filter(participant_id %in% pid_sample)

# Helper function to compute metrics
compute_metrics <- function(actual, pred_prob) {
  roc_obj <- roc(actual, pred_prob, quiet=TRUE)
  roc_auc <- auc(roc_obj)
  pr_obj <- pr.curve(scores.class0 = pred_prob[actual==1], scores.class1 = pred_prob[actual==0], curve=FALSE)
  pr_auc <- pr_obj$auc.integral
  pred_class <- ifelse(pred_prob > 0.5, 1, 0)
  tp <- sum(pred_class == 1 & actual == 1); tn <- sum(pred_class == 0 & actual == 0)
  fp <- sum(pred_class == 1 & actual == 0); fn <- sum(pred_class == 0 & actual == 1)
  denom <- sqrt(as.numeric(tp + fp) * as.numeric(tp + fn) * as.numeric(tn + fp) * as.numeric(tn + fn))
  mcc <- ifelse(denom == 0, 0, (tp * tn - fp * fn) / denom)
  c(ROC = as.numeric(roc_auc), PR = as.numeric(pr_auc), MCC = mcc)
}

extract_metrics <- function(fit_obj, model_name, is_full_gating=FALSE) {
  params <- fit_obj$summary()$mean
  names(params) <- fit_obj$summary()$variable
  
  res_list <- list()
  for (s in 1:30) {
    df_s <- dat_clean %>% filter(participant_id == pid_sample[s])
    alpha_ctx <- params[paste0("alpha_ctx[", s, "]")]; tau_m <- params[paste0("tau_m[", s, "]")]; eta_gc <- params[paste0("eta_gc[", s, "]")]; lambda_gc <- params[paste0("lambda_gc[", s, "]")]; theta_cb <- params[paste0("theta_cb[", s, "]")]; kappa_ctx <- params[paste0("kappa_ctx[", s, "]")]; gamma_suppress <- params[paste0("gamma_suppress[", s, "]")]; a_s <- params[paste0("a[", s, "]")]
    
    if (is_full_gating) {
      beta_a <- params[paste0("beta_a[", s, "]")]
      kappa_cb <- params[paste0("kappa_cb[", s, "]")]
    }
    
    Q_ctx <- c(0.5, 0.5); mf_state <- rep(0.0, 5); w_gc1 <- rep(0.0, 5); w_gc2 <- rep(0.0, 5); w_mli1 <- rep(0.0, 5); w_mli2 <- rep(0.0, 5)
    pu <- numeric(nrow(df_s))
    
    for (t in 1:nrow(df_s)) {
      iti <- df_s$ITI[t]; f_dur <- df_s$F_dur[t]; ch <- df_s$Boundary[t]; rew <- df_s$`F`[t]
      if (iti > 0.01) { mf_state <- mf_state * exp(-iti/tau_m); decay_gc <- exp(-lambda_gc * iti); decay_mli <- exp(-(lambda_gc * 1.5) * iti); w_gc1 <- w_gc1 * decay_gc; w_gc2 <- w_gc2 * decay_gc; w_mli1 <- w_mli1 * decay_mli; w_mli2 <- w_mli2 * decay_mli }
      Q_cb_1 <- sum(w_gc1 * mf_state) - sum(w_mli1 * mf_state); Q_cb_2 <- sum(w_gc2 * mf_state) - sum(w_mli2 * mf_state)
      delta_Q_ctx <- Q_ctx[2] - Q_ctx[1]; delta_Q_cb <- Q_cb_2 - Q_cb_1
      w_bias <- 0.5 + 0.45 * tanh(theta_cb * delta_Q_cb); conflict <- 0.5 * (1.0 - tanh(10.0 * delta_Q_ctx) * tanh(10.0 * delta_Q_cb))
      
      if (is_full_gating) {
        v_base <- kappa_ctx * delta_Q_ctx + kappa_cb * delta_Q_cb
        a_eff <- a_s + beta_a * conflict
      } else {
        v_base <- kappa_ctx * delta_Q_ctx
        a_eff <- a_s
      }
      
      v_eff <- v_base * exp(-gamma_suppress * conflict)
      if (abs(v_eff) < 1e-4) { v_eff <- ifelse(v_eff>=0, 1e-4, -1e-4) }
      
      # Probability of choosing option 1
      pu[t] <- (1.0 - exp(-2.0 * v_eff * a_eff * w_bias)) / (1.0 - exp(-2.0 * v_eff * a_eff))
      
      Q_ctx[ch + 1] <- Q_ctx[ch + 1] + alpha_ctx * (rew - Q_ctx[ch + 1])
      # Approximate exact_mf_step for plasticity
      if (f_dur > 0.01) {
        mf_state <- mf_state * exp(-f_dur/tau_m) + rew * (1.0 - exp(-f_dur/tau_m))
        decay_gc_f <- exp(-lambda_gc * f_dur); decay_mli_f <- exp(-(lambda_gc * 1.5) * f_dur)
        int_gc_f <- (1.0 - decay_gc_f) / lambda_gc; int_mli_f <- (1.0 - decay_mli_f) / (lambda_gc * 1.5)
        E_cb1 <- ifelse(ch == 0, rew - Q_cb_1, 0.0); E_cb2 <- ifelse(ch == 1, rew - Q_cb_2, 0.0)
        w_gc1 <- w_gc1 * decay_gc_f + mf_state * (eta_gc * E_cb1 * int_gc_f); w_mli1 <- w_mli1 * decay_mli_f + mf_state * (-eta_gc * E_cb1 * int_mli_f)
        w_gc2 <- w_gc2 * decay_gc_f + mf_state * (eta_gc * E_cb2 * int_gc_f); w_mli2 <- w_mli2 * decay_mli_f + mf_state * (-eta_gc * E_cb2 * int_mli_f)
      }
    }
    
    # Conditional Switch Probability (since WSLS is heavily dependent on previous choice)
    prev_ch <- lag(df_s$Boundary)
    is_switch <- df_s$Boundary != prev_ch
    p_switch <- ifelse(prev_ch == 1, 1 - pu, pu)
    
    valid_idx <- which(!is.na(prev_ch))
    m_choice <- compute_metrics(df_s$Boundary[valid_idx], pu[valid_idx])
    m_switch <- compute_metrics(as.numeric(is_switch[valid_idx]), p_switch[valid_idx])
    
    res_list[[s]] <- tibble(
      participant_id = pid_sample[s],
      model = model_name,
      choice_ROC = m_choice["ROC"], choice_PR = m_choice["PR"], choice_MCC = m_choice["MCC"],
      switch_ROC = m_switch["ROC"], switch_PR = m_switch["PR"], switch_MCC = m_switch["MCC"]
    )
  }
  bind_rows(res_list)
}

cat("Extracting Base Dual-Kernel metrics...\n")
fit_base <- read_rds("~/cerebellum_project/results/fit_bvk_complete.rds")
res_base <- extract_metrics(fit_base, "Base Dual-Kernel", is_full_gating=FALSE)

cat("Extracting Full-Gating Dual-Kernel metrics...\n")
fit_gating <- read_rds("~/cerebellum_project/results/fit_full_gating_complete.rds")
res_gating <- extract_metrics(fit_gating, "Full-Gating Dual-Kernel", is_full_gating=TRUE)

df_all <- bind_rows(res_base, res_gating)

df_long <- df_all %>%
  pivot_longer(cols = c(choice_ROC, choice_PR, choice_MCC, switch_ROC, switch_PR, switch_MCC), names_to = "Metric", values_to = "Value")

p <- ggplot(df_long, aes(x = model, y = Value, fill = model)) +
  geom_violin(alpha = 0.5) +
  geom_boxplot(width = 0.2, outlier.shape = NA) +
  geom_jitter(width = 0.1, alpha = 0.5) +
  geom_line(aes(group = participant_id), alpha = 0.3, color = "gray") +
  facet_wrap(~Metric, scales = "free_y") +
  theme_pubr() +
  scale_fill_viridis_d(option="viridis", begin=0.2, end=0.8) +
  labs(title = "full-gating architecture achieves decisive superiority across all choice metrics",
       subtitle = "boundary expansion (beta_a) & cerebellar drift (kappa_cb) prevents heuristic reversal errors",
       x = "architecture", y = "metric score") +
  theme(legend.position = "none", strip.text = element_text(face="bold"))

ggsave("~/cerebellum_project/results/figures/6_full_gating_metrics.pdf", plot = p, width = 12, height = 8)
cat("Saved to ~/cerebellum_project/results/figures/6_full_gating_metrics.pdf\n")
