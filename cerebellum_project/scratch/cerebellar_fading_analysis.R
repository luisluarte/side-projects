library(dplyr)
library(ggplot2)
library(lme4)
library(zoo)

cat("Loading dataset...\n")
dataset_path <- "data/raw/behavioral_compilate.csv"
if (!file.exists(dataset_path)) {
  dataset_path <- "c:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv"
}

dat_all <- read.csv(dataset_path)
dat_all[['RT']] <- (as.numeric(dat_all[['ttr']]) - as.numeric(dat_all[['ttp']])) / 1000.0
dat_all <- dat_all[!is.na(dat_all[['RT']]) & dat_all[['RT']] >= 0.1 & dat_all[['RT']] <= 3.0 & dat_all[['Resp']] %in% c(1, 2), ]

cat("Computing WSLS adherence...\n")
df <- dat_all %>%
  group_by(participant_id) %>%
  arrange(ttp) %>%
  mutate(
    trial_idx = row_number(),
    prev_ch = lag(Resp),
    prev_out = lag(F),
    wsls_pred = ifelse(prev_out == 1, prev_ch, ifelse(prev_ch == 1, 2, 1)),
    wsls_adherence = as.integer(Resp == wsls_pred)
  ) %>%
  ungroup()

df <- df %>% filter(!is.na(wsls_adherence))

# -------------------------------------------------------------------
# Option 1: Rolling Window WSLS Adherence
# -------------------------------------------------------------------
cat("Computing Option 1: Rolling Window WSLS Adherence...\n")

window_size <- 20

# Compute rolling average per participant
df_rolling <- df %>%
  group_by(participant_id) %>%
  arrange(trial_idx) %>%
  mutate(
    rolling_adherence = rollapply(wsls_adherence, width = window_size, FUN = mean, align = "right", fill = NA)
  ) %>%
  ungroup()

# Aggregate across participants by trial
agg_rolling <- df_rolling %>%
  filter(!is.na(rolling_adherence)) %>%
  group_by(trial_idx) %>%
  summarize(
    mean_adherence = mean(rolling_adherence, na.rm = TRUE),
    se_adherence = sd(rolling_adherence, na.rm = TRUE) / sqrt(n())
  )

p <- ggplot(agg_rolling, aes(x = trial_idx, y = mean_adherence)) +
  geom_line(color = "#2c3e50", linewidth = 1.2) +
  geom_ribbon(aes(ymin = mean_adherence - se_adherence, ymax = mean_adherence + se_adherence), alpha = 0.2, fill = "#2c3e50") +
  geom_smooth(method = "lm", color = "#e74c3c", linetype = "dashed", se = FALSE) +
  theme_minimal() +
  labs(
    title = "Cerebellar Fading Hypothesis: WSLS Adherence over Time",
    subtitle = "Rolling 20-trial average of Win-Stay/Lose-Shift adherence across 128 participants",
    x = "Trial Number",
    y = "WSLS Adherence Probability"
  )

plot_path <- "C:/Users/DCCS5/.gemini/antigravity/brain/1d8f9958-fd49-4502-b57b-97a7887eb7ad/wsls_adherence_over_time.png"
ggsave(plot_path, p, width = 8, height = 5)
cat(sprintf("Saved plot to %s\n", plot_path))


# -------------------------------------------------------------------
# Option 2: Time-Varying Logistic Regression
# -------------------------------------------------------------------
cat("\nComputing Option 2: GLMM for WSLS Adherence vs Trial...\n")

# Scale trial index to prevent convergence warnings
df$scaled_trial <- scale(df$trial_idx)

model <- glmer(wsls_adherence ~ scaled_trial + (1 | participant_id), 
               family = binomial(link = "logit"), 
               data = df,
               control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=100000)))

cat("\n--- GLMM Summary ---\n")
print(summary(model))

# Extract the p-value for the trial effect
coefs <- summary(model)$coefficients
cat(sprintf("\nEffect of Time on WSLS Adherence: Z = %.3f, p-value = %.3e\n", 
            coefs["scaled_trial", "z value"], 
            coefs["scaled_trial", "Pr(>|z|)"]))
