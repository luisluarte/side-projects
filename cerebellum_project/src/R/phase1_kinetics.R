library(Rcpp)

cat("Loading Phase 1 Dataset...\n")
dataset_path <- "C:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv"
dat_all <- read.csv(dataset_path)

set.seed(42)
participants <- sample(unique(dat_all[['participant_id']]), 3)

sourceCpp("src/cpp/topology_extractor.cpp")

phi_test <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5)

pause_disagreements <- numeric()
base_disagreements <- numeric()
pause_entropies <- numeric()
base_entropies <- numeric()

for (p in participants) {
  p_data <- dat_all[dat_all$participant_id == p, ]
  p_data <- p_data[order(p_data$ttp), ]
  
  # Ensure RT is available
  p_data$RT <- (as.numeric(p_data$ttr) - as.numeric(p_data$ttp)) / 1000.0
  valid_idx <- which(!is.na(p_data$RT) & p_data$RT >= 0.1 & p_data$RT <= 3.0 & p_data$Resp %in% c(1, 2))
  p_data <- p_data[valid_idx, ]
  
  topo <- extract_topology(phi_test, p_data$Resp, p_data$F, p_data$RT)
  
  dt <- diff(p_data$ttp)
  pause_indices <- which(dt > 15000)
  
  for (idx in pause_indices) {
    if (idx + 5 <= nrow(p_data)) {
      # Extract post-pause windows (t+1 to t+5)
      for (i in 1:5) {
        t_target <- idx + i
        disagreement <- abs(topo$Q_CTX[t_target] - topo$Q_CB[t_target])
        pause_disagreements <- c(pause_disagreements, disagreement)
        
        # GC Entropy
        gc_act <- abs(topo$GC[t_target, ])
        gc_prob <- gc_act / sum(gc_act)
        entropy <- -sum(gc_prob[gc_prob > 0] * log(gc_prob[gc_prob > 0]))
        pause_entropies <- c(pause_entropies, entropy)
      }
    }
  }
  
  # Baseline (Sample random continuous blocks where dt < 8000)
  continuous_indices <- which(dt < 8000)
  sampled_base <- sample(continuous_indices, length(pause_indices))
  
  for (idx in sampled_base) {
    if (idx + 5 <= nrow(p_data)) {
      for (i in 1:5) {
        t_target <- idx + i
        disagreement <- abs(topo$Q_CTX[t_target] - topo$Q_CB[t_target])
        base_disagreements <- c(base_disagreements, disagreement)
        
        gc_act <- abs(topo$GC[t_target, ])
        gc_prob <- gc_act / sum(gc_act)
        entropy <- -sum(gc_prob[gc_prob > 0] * log(gc_prob[gc_prob > 0]))
        base_entropies <- c(base_entropies, entropy)
      }
    }
  }
}

cat("\n--- PHASE 1: DISAGREEMENT TRACKING (t-test) ---\n")
t_test_dis <- t.test(pause_disagreements, base_disagreements)
print(t_test_dis)

cat("\n--- PHASE 1: GRANULAR SPATIAL ENTROPY (t-test) ---\n")
t_test_ent <- t.test(pause_entropies, base_entropies)
print(t_test_ent)

cat("\nPhase 1 Execution Complete.\n")
