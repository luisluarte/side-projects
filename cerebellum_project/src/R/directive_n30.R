library(Rcpp)

cat("Loading Dataset and Sampling N=30...\n")
dataset_path <- "C:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv"
dat_all <- read.csv(dataset_path)

set.seed(42)
participants <- sample(unique(dat_all[['participant_id']]), 30)

sourceCpp("src/cpp/topology_extractor.cpp")
sourceCpp("src/cpp/phase2_ablation.cpp")

phi_test <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5)

# --- Phase 1: Inter-Trial Kinetics (N=30) ---
cat("\n=== PHASE 1: INTER-TRIAL KINETICS (N=30) ===\n")
pause_disagreements <- numeric()
base_disagreements <- numeric()
pause_entropies <- numeric()
base_entropies <- numeric()

for (p in participants) {
  p_data <- dat_all[dat_all$participant_id == p, ]
  p_data <- p_data[order(p_data$ttp), ]
  p_data$RT <- (as.numeric(p_data$ttr) - as.numeric(p_data$ttp)) / 1000.0
  valid_idx <- which(!is.na(p_data$RT) & p_data$RT >= 0.1 & p_data$RT <= 3.0 & p_data$Resp %in% c(1, 2))
  p_data <- p_data[valid_idx, ]
  
  topo <- extract_topology(phi_test, p_data$Resp, p_data$F, p_data$RT)
  dt <- c(NA, diff(p_data$ttp))
  
  pause_indices <- which(dt > 15000)
  for (idx in pause_indices) {
    if (idx + 5 <= nrow(p_data)) {
      for (i in 1:5) {
        t_target <- idx + i
        pause_disagreements <- c(pause_disagreements, abs(topo$Q_CTX[t_target] - topo$Q_CB[t_target]))
        gc_act <- abs(topo$GC[t_target, ])
        gc_prob <- gc_act / sum(gc_act)
        pause_entropies <- c(pause_entropies, -sum(gc_prob[gc_prob > 0] * log(gc_prob[gc_prob > 0])))
      }
    }
  }
  
  continuous_indices <- which(dt < 8000)
  if (length(continuous_indices) > 0 && length(pause_indices) > 0) {
    sampled_base <- sample(continuous_indices, min(length(pause_indices), length(continuous_indices)), replace=TRUE)
    for (idx in sampled_base) {
      if (idx + 5 <= nrow(p_data)) {
        for (i in 1:5) {
          t_target <- idx + i
          base_disagreements <- c(base_disagreements, abs(topo$Q_CTX[t_target] - topo$Q_CB[t_target]))
          gc_act <- abs(topo$GC[t_target, ])
          gc_prob <- gc_act / sum(gc_act)
          base_entropies <- c(base_entropies, -sum(gc_prob[gc_prob > 0] * log(gc_prob[gc_prob > 0])))
        }
      }
    }
  }
}

t_test_dis <- t.test(pause_disagreements, base_disagreements)
print(t_test_dis)

t_test_ent <- t.test(pause_entropies, base_entropies)
print(t_test_ent)


# --- Phase 2.1: Grid Search (N=30) ---
cat("\n=== PHASE 2.1: GRID SEARCH (N=30) ===\n")
k_grid <- c(10, 20, 40, 60, 80)
results <- numeric(length(k_grid))

for (p in participants) {
  p_data <- dat_all[dat_all$participant_id == p, ]
  p_data <- p_data[order(p_data$ttp), ]
  p_data$RT <- (as.numeric(p_data$ttr) - as.numeric(p_data$ttp)) / 1000.0
  valid_idx <- which(!is.na(p_data$RT) & p_data$RT >= 0.1 & p_data$RT <= 3.0 & p_data$Resp %in% c(1, 2))
  p_data <- p_data[valid_idx, ]
  
  for (i in seq_along(k_grid)) {
    results[i] <- results[i] + eval_eccm_k(k_grid[i], phi_test, p_data$Resp, p_data$F, p_data$RT)
  }
}

for (i in seq_along(k_grid)) {
  cat(sprintf("k = %d | Total Penalized Deviance: %.2f\n", k_grid[i], results[i]))
}
k_optimal <- k_grid[which.min(results)]
cat(sprintf("Optimal Memory Depth (k_optimal): %d\n", k_optimal))


# --- Phase 2.2: Ablation Test (N=30) ---
cat("\n=== PHASE 2.2: MANIFOLD ABLATION (N=30) ===\n")
intact_devs <- numeric(length(participants))
lesion_devs <- numeric(length(participants))

for (i in seq_along(participants)) {
  p <- participants[i]
  p_data <- dat_all[dat_all$participant_id == p, ]
  p_data <- p_data[order(p_data$ttp), ]
  p_data$RT <- (as.numeric(p_data$ttr) - as.numeric(p_data$ttp)) / 1000.0
  valid_idx <- which(!is.na(p_data$RT) & p_data$RT >= 0.1 & p_data$RT <= 3.0 & p_data$Resp %in% c(1, 2))
  p_data <- p_data[valid_idx, ]
  
  intact_devs[i] <- eval_eccm_k(k_optimal, phi_test, p_data$Resp, p_data$F, p_data$RT)
  lesion_devs[i] <- eval_lesioned(k_optimal, phi_test, p_data$Resp, p_data$F, p_data$RT)
}

test_res <- t.test(intact_devs, lesion_devs, paired=TRUE, alternative="less")
print(test_res)

cat(sprintf("Mean Intact Deviance: %.2f\n", mean(intact_devs)))
cat(sprintf("Mean Lesioned Deviance: %.2f\n", mean(lesion_devs)))

cat("\nFull N=30 Directive Execution Complete.\n")
