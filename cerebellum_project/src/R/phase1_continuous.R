library(Rcpp)
library(lme4)
library(lmerTest) # For p-values in lmer

cat("Loading Dataset...\n")
dataset_path <- "C:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv"
dat_all <- read.csv(dataset_path)

set.seed(42)
participants <- sample(unique(dat_all[['participant_id']]), 30)

sourceCpp("src/cpp/topology_extractor.cpp")
phi_test <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5)

df_list <- list()

for (p in participants) {
  p_data <- dat_all[dat_all$participant_id == p, ]
  p_data <- p_data[order(p_data$ttp), ]
  
  p_data$RT <- (as.numeric(p_data$ttr) - as.numeric(p_data$ttp)) / 1000.0
  valid_idx <- which(!is.na(p_data$RT) & p_data$RT >= 0.1 & p_data$RT <= 3.0 & p_data$Resp %in% c(1, 2))
  p_data <- p_data[valid_idx, ]
  
  topo <- extract_topology(phi_test, p_data$Resp, p_data$F, p_data$RT)
  
  dt <- c(NA, diff(p_data$ttp))
  
  # Calculate metrics for all valid trials (skip t=1 since dt is NA)
  for (t in 2:nrow(p_data)) {
    disagreement <- abs(topo$Q_CTX[t] - topo$Q_CB[t])
    gc_act <- abs(topo$GC[t, ])
    gc_prob <- gc_act / sum(gc_act)
    entropy <- -sum(gc_prob[gc_prob > 0] * log(gc_prob[gc_prob > 0]))
    
    df_list[[length(df_list) + 1]] <- data.frame(
      subject = p,
      dt = dt[t] / 1000.0, # Convert to seconds
      disagreement = disagreement,
      entropy = entropy
    )
  }
}

df <- do.call(rbind, df_list)
# Remove absurd dt outliers (e.g. overnight pauses > 1000s)
df <- df[df$dt > 0 & df$dt < 1000, ]

cat("\n--- Hierarchical Continuous Regression: Disagreement vs dt ---\n")
model_dis <- lmer(disagreement ~ dt + (1 | subject), data = df)
print(summary(model_dis))

cat("\n--- Hierarchical Continuous Regression: GC Entropy vs dt ---\n")
model_ent <- lmer(entropy ~ dt + (1 | subject), data = df)
print(summary(model_ent))
