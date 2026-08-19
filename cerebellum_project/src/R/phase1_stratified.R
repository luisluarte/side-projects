library(Rcpp)
library(lme4)
library(lmerTest)

cat("Loading Dataset and Sampling N=30...\n")
dataset_path <- "C:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv"
dat_all <- read.csv(dataset_path)

set.seed(42)
participants <- sample(unique(dat_all[['participant_id']]), 30)

sourceCpp("src/cpp/topology_extractor.cpp")
phi_test <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5)

pause_summary <- data.frame(Participant=character(), Total_Pauses=integer(), Group_A=integer(), Group_B=integer(), stringsAsFactors=FALSE)

df_list <- list()

for (p in participants) {
  p_data <- dat_all[dat_all$participant_id == p, ]
  p_data <- p_data[order(p_data$ttp), ]
  p_data$RT <- (as.numeric(p_data$ttr) - as.numeric(p_data$ttp)) / 1000.0
  
  valid_idx <- which(!is.na(p_data$RT) & p_data$RT >= 0.1 & p_data$RT <= 3.0 & p_data$Resp %in% c(1, 2))
  p_data <- p_data[valid_idx, ]
  if (nrow(p_data) < 10) next
  
  topo <- extract_topology(phi_test, p_data$Resp, p_data$F, p_data$RT)
  dt <- c(NA, diff(p_data$ttp))
  
  pause_indices <- which(dt > 15000)
  
  grp_a_count <- 0
  grp_b_count <- 0
  
  for (idx in pause_indices) {
    if (idx + 5 <= nrow(p_data) && idx - 1 >= 1) {
      pre_prob <- p_data$prob[idx - 1]
      post_prob <- p_data$prob[idx]
      
      group <- ifelse(pre_prob == post_prob, "A_Stationary", "B_Volatile")
      if (group == "A_Stationary") grp_a_count <- grp_a_count + 1
      else grp_b_count <- grp_b_count + 1
      
      for (i in 1:5) {
        t_target <- idx + (i - 1)
        disagreement <- abs(topo$Q_CTX[t_target] - topo$Q_CB[t_target])
        
        df_list[[length(df_list) + 1]] <- data.frame(
          subject = p,
          pause_id = paste0(p, "_", idx),
          time_post = i,
          group = group,
          disagreement = disagreement
        )
      }
    }
  }
  
  pause_summary <- rbind(pause_summary, data.frame(
    Participant = p,
    Total_Pauses = grp_a_count + grp_b_count,
    Group_A_Stationary = grp_a_count,
    Group_B_Volatile = grp_b_count
  ))
}

df <- do.call(rbind, df_list)

cat("\n--- Pause Summary Table ---\n")
print(head(pause_summary))
cat(sprintf("\nTotal Group A Pauses: %d\n", sum(pause_summary$Group_A_Stationary)))
cat(sprintf("Total Group B Pauses: %d\n", sum(pause_summary$Group_B_Volatile)))

cat("\n--- Hierarchical LMM: Disagreement ~ Group ---\n")
# Ensure group is a factor
df$group <- as.factor(df$group)
# We test if Group B is significantly higher than Group A
model_dis <- lmer(disagreement ~ group + time_post + (1 | subject), data = df)
print(summary(model_dis))

# Output tables for LaTeX
write.csv(pause_summary, "docs/pause_summary.csv", row.names=FALSE)
write.csv(aggregate(disagreement ~ group + time_post, data = df, mean), "docs/disagreement_timecourse.csv", row.names=FALSE)
