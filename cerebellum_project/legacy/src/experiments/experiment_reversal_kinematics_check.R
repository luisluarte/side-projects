library(Rcpp)
library(lme4)
library(lmerTest)
library(ggplot2)

cat("Loading Dataset and Sampling N=30...\n")
dataset_path <- "C:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv"
dat_all <- read.csv(dataset_path)

set.seed(42)
participants <- sample(unique(dat_all[['participant_id']]), 30)

sourceCpp("src/models/topology_extractor.cpp")
sourceCpp("src/models/evaluate_metrics.cpp")

phi_test <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5)

df_list <- list()

for (p in participants) {
  p_data <- dat_all[dat_all$participant_id == p, ]
  p_data <- p_data[order(p_data$ttp), ]
  p_data$RT <- (as.numeric(p_data$ttr) - as.numeric(p_data$ttp)) / 1000.0
  
  valid_idx <- which(!is.na(p_data$RT) & p_data$RT >= 0.1 & p_data$RT <= 3.0 & p_data$Resp %in% c(1, 2))
  p_data <- p_data[valid_idx, ]
  if (nrow(p_data) < 20) next
  
  topo <- extract_topology(phi_test, p_data$Resp, p_data$F, p_data$RT)
  
  # Get predictive probabilities
  metrics <- eval_metrics_eccm(phi_test, p_data$Resp, p_data$F, p_data$RT, FALSE)
  prob_ch1 <- metrics$prob_ch1
  
  # Compute Probability of Actual Choice
  prob_actual <- ifelse(p_data$Resp == 1, prob_ch1, 1.0 - prob_ch1)
  
  dt <- c(NA, diff(p_data$ttp))
  
  for (t in 2:(nrow(p_data) - 5)) {
    is_pause <- !is.na(dt[t]) && dt[t] > 15000
    is_reversal <- p_data$prob[t - 1] != p_data$prob[t]
    
    group <- NA
    if (is_pause && !is_reversal) group <- "Pause_Stationary"
    else if (is_pause && is_reversal) group <- "Pause_Volatile"
    else if (!is_pause && is_reversal) group <- "Continuous_Reversal"
    else if (!is_pause && !is_reversal) group <- "Continuous_Stationary"
    
    if (!is.na(group)) {
      for (i in 1:5) {
        t_target <- t + (i - 1)
        disagreement <- abs(topo$Q_CTX[t_target] - topo$Q_CB[t_target])
        pred_prob <- prob_actual[t_target]
        
        df_list[[length(df_list) + 1]] <- data.frame(
          subject = p,
          event_id = paste0(p, "_", t),
          time_post = i,
          group = group,
          is_pause = is_pause,
          is_reversal = is_reversal,
          disagreement = disagreement,
          pred_power = pred_prob
        )
      }
    }
  }
}

df <- do.call(rbind, df_list)
df$group <- as.factor(df$group)
df$group <- relevel(df$group, ref = "Continuous_Stationary") # baseline

cat("\n--- Event Counts (Trials * 5) ---\n")
print(table(df$group))

cat("\n--- Mean Disagreement & Predictive Power by Group ---\n")
agg <- aggregate(cbind(disagreement, pred_power) ~ group, data = df, mean)
print(agg)

cat("\n--- Hierarchical LMM: Predictive Power ~ Pause * Reversal ---\n")
model_pred <- lmer(pred_power ~ is_pause * is_reversal + time_post + (1 | subject), data = df)
print(summary(model_pred))

# Plotting Predictive Power
p <- ggplot(agg, aes(x = group, y = pred_power, fill = group)) +
  geom_bar(stat = "identity", width = 0.6, color = "black") +
  theme_minimal() +
  scale_fill_manual(values = c("Pause_Volatile" = "#e74c3c", 
                               "Continuous_Reversal" = "#f39c12", 
                               "Pause_Stationary" = "#3498db",
                               "Continuous_Stationary" = "#2ecc71")) +
  labs(title = "Model Predictive Power Following Events",
       x = "Event Type",
       y = "Mean Probability of Actual Choice") +
  theme(legend.position = "none", text = element_text(size = 12)) +
  coord_cartesian(ylim = c(0.4, 0.7)) +
  scale_x_discrete(labels = c("Continuous\nStationary", "Continuous\nReversal", "Stationary\nPause", "Volatile\nPause"))

ggsave("results/figures/predictive_power_pauses.png", plot = p, width = 8, height = 6)

write.csv(agg, "results/tables/predictive_power_summary.csv", row.names=FALSE)
