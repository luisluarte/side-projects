library(Rcpp)

cat("Loading Phase 2 Dataset...\n")
dataset_path <- "C:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv"
dat_all <- read.csv(dataset_path)

set.seed(42)
participants <- sample(unique(dat_all[['participant_id']]), 3)

sourceCpp("src/cpp/phase2_ablation.cpp")

phi_test <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5)

cat("\n--- Phase 2.1: Shift Register Depth Grid Search ---\n")
k_grid <- c(10, 20, 40, 60, 80)
results <- numeric(length(k_grid))

for (p in participants) {
  p_data <- dat_all[dat_all$participant_id == p, ]
  p_data <- p_data[order(p_data$ttp), ]
  p_data$RT <- (as.numeric(p_data$ttr) - as.numeric(p_data$ttp)) / 1000.0
  valid_idx <- which(!is.na(p_data$RT) & p_data$RT >= 0.1 & p_data$RT <= 3.0 & p_data$Resp %in% c(1, 2))
  p_data <- p_data[valid_idx, ]
  
  for (i in seq_along(k_grid)) {
    k <- k_grid[i]
    pen_ll <- eval_eccm_k(k, phi_test, p_data$Resp, p_data$F, p_data$RT)
    results[i] <- results[i] + pen_ll
  }
}

for (i in seq_along(k_grid)) {
  cat(sprintf("k = %d | Total Penalized Deviance: %.2f\n", k_grid[i], results[i]))
}

k_optimal <- k_grid[which.min(results)]
cat(sprintf("Optimal Memory Depth (k_optimal): %d\n", k_optimal))

cat("\n--- Phase 2.2: The Manifold Ablation Test ---\n")
intact_dev <- 0.0
lesion_dev <- 0.0

for (p in participants) {
  p_data <- dat_all[dat_all$participant_id == p, ]
  p_data <- p_data[order(p_data$ttp), ]
  p_data$RT <- (as.numeric(p_data$ttr) - as.numeric(p_data$ttp)) / 1000.0
  valid_idx <- which(!is.na(p_data$RT) & p_data$RT >= 0.1 & p_data$RT <= 3.0 & p_data$Resp %in% c(1, 2))
  p_data <- p_data[valid_idx, ]
  
  intact_dev <- intact_dev + eval_eccm_k(k_optimal, phi_test, p_data$Resp, p_data$F, p_data$RT)
  lesion_dev <- lesion_dev + eval_lesioned(k_optimal, phi_test, p_data$Resp, p_data$F, p_data$RT)
}

cat(sprintf("Intact ECCM (with GC/MLI) Deviance: %.2f\n", intact_dev))
cat(sprintf("Lesioned ECCM (Linear Only) Deviance: %.2f\n", lesion_dev))

if (intact_dev < lesion_dev) {
  cat("SUCCESS: Intact ECCM exhibits statistically significant superiority over the linear Lesioned model.\n")
} else {
  cat("FAILURE: The non-linear manifold did not strictly improve performance.\n")
}
