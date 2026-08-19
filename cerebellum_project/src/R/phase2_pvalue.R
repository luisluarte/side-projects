library(Rcpp)

cat("Loading Dataset...\n")
dataset_path <- "C:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv"
dat_all <- read.csv(dataset_path)

set.seed(42)
participants <- sample(unique(dat_all[['participant_id']]), 3)

sourceCpp("src/cpp/phase2_ablation.cpp")
phi_test <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5)
k_optimal <- 80

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

cat("\n--- Phase 2: Structural Ablation (Paired t-test) ---\n")
test_res <- t.test(intact_devs, lesion_devs, paired=TRUE, alternative="less")
print(test_res)

cat(sprintf("Mean Intact Deviance: %.2f\n", mean(intact_devs)))
cat(sprintf("Mean Lesioned Deviance: %.2f\n", mean(lesion_devs)))
