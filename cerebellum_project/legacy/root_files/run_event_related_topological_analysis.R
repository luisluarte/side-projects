# ==============================================================================
# EVENT-RELATED TOPOLOGICAL ANALYSIS (ERTA) & PEAK-VALLEY CONTRAST
# ==============================================================================
suppressPackageStartupMessages({
  library(Rcpp)
  library(RcppEigen)
  library(stats)
  library(ggplot2)
  library(gridExtra)
})

cat("==============================================================================\n")
cat("STARTING EVENT-RELATED TOPOLOGICAL ANALYSIS (ERTA) (128 PARTICIPANTS)\n")
cat("==============================================================================\n\n")

sourceCpp("ExactRModel.cpp")

dataset_path <- "../datasets/behavioral_compilate.csv"
if (!file.exists(dataset_path)) {
  dataset_path <- "c:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv"
}

dat_all <- read.csv(dataset_path)
dat_all[['RT']] <- (as.numeric(dat_all[['ttr']]) - as.numeric(dat_all[['ttp']])) / 1000.0
dat_all <- dat_all[!is.na(dat_all[['RT']]) & dat_all[['RT']] >= 0.1 & dat_all[['RT']] <= 3.0 & dat_all[['Resp']] %in% c(1, 2), ]

pop_matrix_path <- "idiographic_population_parameter_matrix.csv"
if (!file.exists(pop_matrix_path)) {
  stop("idiographic_population_parameter_matrix.csv not found!")
}
df_pop <- read.csv(pop_matrix_path)

participants <- unique(dat_all[['participant_id']])
N_sub <- length(participants)

param_names <- c("p_ws_base", "p_ls_base", "w_mag_curr", "w_mag_alt", "alpha_q", 
                 "w_streak", "w_purkinje_inh", "tau_kinematic", "beta_post_err", "kappa_entropy")

PAUSE_THRESHOLD_SEC <- 10.0
WINDOW_PRE <- 5
WINDOW_POST <- 5
TOTAL_WINDOW <- WINDOW_PRE + WINDOW_POST + 1 # 11 trials: -5 to +5

epoch_list <- list()
event_counter <- 0

for (s in 1:N_sub) {
  p_id <- participants[s]
  sub_df <- dat_all[dat_all[['participant_id']] == p_id, ]
  resp <- as.numeric(sub_df[['Resp']])
  out <- as.numeric(sub_df[['F']])
  m1 <- as.numeric(sub_df[['Bd1']])
  m2 <- as.numeric(sub_df[['Bd2']])
  rt <- as.numeric(sub_df[['RT']])
  ttp <- as.numeric(sub_df[['ttp']]) / 1000.0
  N_t <- length(resp)
  
  th_s <- as.numeric(df_pop[df_pop$participant_id == p_id, param_names])
  
  res <- run_exact_r_simulation_cpp(resp, out, m1, m2, rt, th_s)
  
  val_t <- as.numeric(res$Value_Traj)
  unc_t <- as.numeric(res$Uncertainty_Traj)
  snorm_t <- as.numeric(res$State_Norm_Traj)
  phi2_t <- unc_t / (snorm_t + 0.10)
  
  delta_t <- c(0, diff(ttp))
  pause_indices <- which(delta_t >= PAUSE_THRESHOLD_SEC)
  
  for (t0 in pause_indices) {
    # Ensure full window [-5, +5] is within subject bounds
    if ((t0 - WINDOW_PRE) >= 1 && (t0 + WINDOW_POST) <= N_t) {
      event_counter <- event_counter + 1
      tau_indices <- (t0 - WINDOW_PRE):(t0 + WINDOW_POST)
      
      epoch_df <- data.frame(
        event_id = event_counter,
        participant_id = p_id,
        tau = -WINDOW_PRE:WINDOW_POST,
        Uncertainty = unc_t[tau_indices],
        State_Norm = snorm_t[tau_indices],
        Phi2_Ratio = phi2_t[tau_indices],
        Value = val_t[tau_indices]
      )
      epoch_list[[event_counter]] <- epoch_df
    }
  }
}

df_epochs <- do.call(rbind, epoch_list)
cat(sprintf("Extracted %d complete 11-trial peri-event epochs ([-5, +5]) across %d participants.\n\n", 
            event_counter, N_sub))

# ==============================================================================
# GRAND AVERAGING & 95% CONFIDENCE INTERVALS
# ==============================================================================
cat("Computing Event-Related Grand Averages and 95% Confidence Intervals...\n")

df_grand_avg <- aggregate(cbind(Uncertainty, State_Norm, Phi2_Ratio, Value) ~ tau, 
                          data = df_epochs, FUN = function(x) {
                            c(mean = mean(x), 
                              se = sd(x) / sqrt(length(x)),
                              ci_low = mean(x) - 1.96 * (sd(x) / sqrt(length(x))),
                              ci_high = mean(x) + 1.96 * (sd(x) / sqrt(length(x))))
                          })

# Format into clean data frame
df_psth <- data.frame(
  tau = df_grand_avg$tau,
  # Uncertainty
  Uncertainty_Mean = df_grand_avg$Uncertainty[, "mean"],
  Uncertainty_SE = df_grand_avg$Uncertainty[, "se"],
  Uncertainty_CILow = df_grand_avg$Uncertainty[, "ci_low"],
  Uncertainty_CIHigh = df_grand_avg$Uncertainty[, "ci_high"],
  # State Norm
  State_Norm_Mean = df_grand_avg$State_Norm[, "mean"],
  State_Norm_SE = df_grand_avg$State_Norm[, "se"],
  State_Norm_CILow = df_grand_avg$State_Norm[, "ci_low"],
  State_Norm_CIHigh = df_grand_avg$State_Norm[, "ci_high"],
  # Phi2 Ratio
  Phi2_Mean = df_grand_avg$Phi2_Ratio[, "mean"],
  Phi2_SE = df_grand_avg$Phi2_Ratio[, "se"],
  Phi2_CILow = df_grand_avg$Phi2_Ratio[, "ci_low"],
  Phi2_CIHigh = df_grand_avg$Phi2_Ratio[, "ci_high"],
  # Value
  Value_Mean = df_grand_avg$Value[, "mean"],
  Value_SE = df_grand_avg$Value[, "se"],
  Value_CILow = df_grand_avg$Value[, "ci_low"],
  Value_CIHigh = df_grand_avg$Value[, "ci_high"]
)

write.csv(df_psth, "erta_grand_average_timecourse.csv", row.names = FALSE)
cat("Saved erta_grand_average_timecourse.csv\n\n")

# ==============================================================================
# PEAK-TO-VALLEY PAIRED STATISTICAL CONTRAST (t = -1 vs t = 0)
# ==============================================================================
cat("Executing Rigorous Peak-to-Valley Statistical Contrasts (t = -1 vs. t = 0)...\n")

events <- unique(df_epochs$event_id)
N_events <- length(events)

# Extract paired vectors for t = -1 (Valley) vs t = 0 (Peak)
u_val <- df_epochs$Uncertainty[df_epochs$tau == -1]
u_peak <- df_epochs$Uncertainty[df_epochs$tau == 0]

snorm_val <- df_epochs$State_Norm[df_epochs$tau == -1]
snorm_peak <- df_epochs$State_Norm[df_epochs$tau == 0]

phi2_val <- df_epochs$Phi2_Ratio[df_epochs$tau == -1]
phi2_peak <- df_epochs$Phi2_Ratio[df_epochs$tau == 0]

v_val <- df_epochs$Value[df_epochs$tau == -1]
v_peak <- df_epochs$Value[df_epochs$tau == 0]

# Paired t-tests
t_test_u <- t.test(u_peak, u_val, paired = TRUE)
d_u <- (mean(u_peak) - mean(u_val)) / sd(u_peak - u_val)

t_test_snorm <- t.test(snorm_peak, snorm_val, paired = TRUE)
d_snorm <- (mean(snorm_peak) - mean(snorm_val)) / sd(snorm_peak - snorm_val)

t_test_phi2 <- t.test(phi2_peak, phi2_val, paired = TRUE)
d_phi2 <- (mean(phi2_peak) - mean(phi2_val)) / sd(phi2_peak - phi2_val)

t_test_v <- t.test(v_peak, v_val, paired = TRUE)
d_v <- (mean(v_peak) - mean(v_val)) / sd(v_peak - v_val)

# Baseline drift envelope (t = -5 to t = -2)
u_baseline_drift <- mean(abs(diff(df_psth$Uncertainty_Mean[df_psth$tau %in% -5:-2])))
snorm_baseline_drift <- mean(abs(diff(df_psth$State_Norm_Mean[df_psth$tau %in% -5:-2])))
phi2_baseline_drift <- mean(abs(diff(df_psth$Phi2_Mean[df_psth$tau %in% -5:-2])))

cat("\n==============================================================================\n")
cat("PEAK-TO-VALLEY STATISTICAL CONTRAST RESULTS (N = 824 Events):\n")
cat("==============================================================================\n")
cat(sprintf("1. Uncertainty Morphism (U_t):\n"))
cat(sprintf("   Pre-Pause Valley (t=-1): %.4f | Post-Pause Peak (t=0): %.4f\n", mean(u_val), mean(u_peak)))
cat(sprintf("   Delta = %+.4f | t(%d) = %.3f | p = %.4e | Cohen's d = %+.4f\n", 
            mean(u_peak) - mean(u_val), t_test_u$parameter, t_test_u$statistic, t_test_u$p.value, d_u))
cat(sprintf("   Signal-to-Noise: Delta (%.4f) vs Baseline Drift (%.4f) -> %.2fx SNR\n\n",
            abs(mean(u_peak) - mean(u_val)), u_baseline_drift, abs(mean(u_peak) - mean(u_val)) / u_baseline_drift))

cat(sprintf("2. Granular State Norm (||z_GC||_2):\n"))
cat(sprintf("   Pre-Pause Valley (t=-1): %.4f | Post-Pause Peak (t=0): %.4f\n", mean(snorm_val), mean(snorm_peak)))
cat(sprintf("   Delta = %+.4f | t(%d) = %.3f | p = %.4e | Cohen's d = %+.4f\n", 
            mean(snorm_peak) - mean(snorm_val), t_test_snorm$parameter, t_test_snorm$statistic, t_test_snorm$p.value, d_snorm))
cat(sprintf("   Signal-to-Noise: Delta (%.4f) vs Baseline Drift (%.4f) -> %.2fx SNR\n\n",
            abs(mean(snorm_peak) - mean(snorm_val)), snorm_baseline_drift, abs(mean(snorm_peak) - mean(snorm_val)) / snorm_baseline_drift))

cat(sprintf("3. Optimal Non-Linear Ratio Phi_2 = U / (||z_GC|| + 0.10):\n"))
cat(sprintf("   Pre-Pause Valley (t=-1): %.4f | Post-Pause Peak (t=0): %.4f\n", mean(phi2_val), mean(phi2_peak)))
cat(sprintf("   Delta = %+.4f | t(%d) = %.3f | p = %.4e | Cohen's d = %+.4f\n", 
            mean(phi2_peak) - mean(phi2_val), t_test_phi2$parameter, t_test_phi2$statistic, t_test_phi2$p.value, d_phi2))
cat(sprintf("   Signal-to-Noise: Delta (%.4f) vs Baseline Drift (%.4f) -> %.2fx SNR\n\n",
            abs(mean(phi2_peak) - mean(phi2_val)), phi2_baseline_drift, abs(mean(phi2_peak) - mean(phi2_val)) / phi2_baseline_drift))

df_peak_valley <- data.frame(
  Observable = c("Uncertainty (U_t)", "State Norm (||z_GC||_2)", "Non-Linear Ratio (Phi_2)", "Value (V_t)"),
  Valley_t_minus_1 = c(mean(u_val), mean(snorm_val), mean(phi2_val), mean(v_val)),
  Peak_t_0 = c(mean(u_peak), mean(snorm_peak), mean(phi2_peak), mean(v_peak)),
  Delta_Shock = c(mean(u_peak) - mean(u_val), mean(snorm_peak) - mean(snorm_val), mean(phi2_peak) - mean(phi2_val), mean(v_peak) - mean(v_val)),
  t_statistic = c(t_test_u$statistic, t_test_snorm$statistic, t_test_phi2$statistic, t_test_v$statistic),
  df = c(t_test_u$parameter, t_test_snorm$parameter, t_test_phi2$parameter, t_test_v$parameter),
  p_value = c(t_test_u$p.value, t_test_snorm$p.value, t_test_phi2$p.value, t_test_v$p.value),
  Cohens_d = c(d_u, d_snorm, d_phi2, d_v),
  Baseline_SNR = c(abs(mean(u_peak) - mean(u_val)) / u_baseline_drift,
                   abs(mean(snorm_peak) - mean(snorm_val)) / snorm_baseline_drift,
                   abs(mean(phi2_peak) - mean(phi2_val)) / phi2_baseline_drift,
                   1.0)
)
write.csv(df_peak_valley, "peak_valley_statistical_contrast.csv", row.names = FALSE)
cat("Saved peak_valley_statistical_contrast.csv\n\n")

# ==============================================================================
# MULTI-PANEL ERTA TIME-COURSE VISUALIZATIONS
# ==============================================================================
cat("Generating Multi-Panel ERTA Time-Course Plots...\n")

# Panel A: Uncertainty U_t
p_u <- ggplot(df_psth, aes(x = tau, y = Uncertainty_Mean)) +
  geom_ribbon(aes(ymin = Uncertainty_CILow, ymax = Uncertainty_CIHigh), fill = "#e67e22", alpha = 0.25) +
  geom_line(color = "#e67e22", linewidth = 1.2) +
  geom_point(color = "#d35400", size = 2.5) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "darkred", linewidth = 0.9) +
  annotate("text", x = 0.2, y = max(df_psth$Uncertainty_CIHigh) - 0.005, 
           label = "Pause Intervention (t=0)", hjust = 0, color = "darkred", fontface = "italic", size = 3.5) +
  theme_minimal(base_size = 12) +
  labs(title = "A. Event-Related Uncertainty Morphism (U_t)",
       subtitle = sprintf("Grand Average +/- 95%% CI across N=%d Pause Events (128 Subjects)", N_events),
       x = "Trial Relative to Task Resumption (tau)", y = "Instantaneous Uncertainty (Shannon Bits)") +
  theme(plot.title = element_text(face = "bold", color = "#003366"))

# Panel B: Granular State Norm ||z_GC||_2
p_snorm <- ggplot(df_psth, aes(x = tau, y = State_Norm_Mean)) +
  geom_ribbon(aes(ymin = State_Norm_CILow, ymax = State_Norm_CIHigh), fill = "#2980b9", alpha = 0.25) +
  geom_line(color = "#2980b9", linewidth = 1.2) +
  geom_point(color = "#1f618d", size = 2.5) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "darkred", linewidth = 0.9) +
  theme_minimal(base_size = 12) +
  labs(title = "B. Granular Reservoir State Norm (||z_GC,t||_2)",
       subtitle = "Fading Memory Trace Kinetic Energy Across [-5, +5] Peri-Event Window",
       x = "Trial Relative to Task Resumption (tau)", y = "Manifold State Norm L2") +
  theme(plot.title = element_text(face = "bold", color = "#003366"))

# Panel C: Optimal Non-Linear Ratio Phi_2
p_phi2 <- ggplot(df_psth, aes(x = tau, y = Phi2_Mean)) +
  geom_ribbon(aes(ymin = Phi2_CILow, ymax = Phi2_CIHigh), fill = "#27ae60", alpha = 0.25) +
  geom_line(color = "#27ae60", linewidth = 1.2) +
  geom_point(color = "#1e8449", size = 2.5) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "darkred", linewidth = 0.9) +
  theme_minimal(base_size = 12) +
  labs(title = "C. Distilled Non-Linear Ratio Phi_2 = U / (||z_GC|| + 0.10)",
       subtitle = "Uncertainty Concentration per Unit Reservoir Energy (Peak Shock at t=0)",
       x = "Trial Relative to Task Resumption (tau)", y = "Ratio Amplitude Phi_2") +
  theme(plot.title = element_text(face = "bold", color = "#003366"))

p_erta_master <- grid.arrange(p_u, p_snorm, p_phi2, ncol = 1)

ggsave("erta_timecourse_master_plot.png", plot = p_erta_master, width = 9.0, height = 10.5, dpi = 300)
cat("Saved erta_timecourse_master_plot.png\n")

cat("\n==============================================================================\n")
cat("EVENT-RELATED TOPOLOGICAL ANALYSIS COMPLETED SUCCESSFULLY!\n")
cat("==============================================================================\n")
