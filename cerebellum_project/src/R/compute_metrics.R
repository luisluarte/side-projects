library(Rcpp)
library(PRROC)

cat("Loading Dataset...\n")
dataset_path <- "C:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv"
dat_all <- read.csv(dataset_path)

participants <- unique(dat_all[['participant_id']])

sourceCpp("src/cpp/evaluate_metrics.cpp")

phi_test <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5)

results_list <- list()

cat("Computing PR-AUC and RT-RMSE for all models...\n")

for (p in participants) {
  p_data <- dat_all[dat_all$participant_id == p, ]
  p_data <- p_data[order(p_data$ttp), ]
  p_data$RT <- (as.numeric(p_data$ttr) - as.numeric(p_data$ttp)) / 1000.0
  valid_idx <- which(!is.na(p_data$RT) & p_data$RT >= 0.1 & p_data$RT <= 3.0 & p_data$Resp %in% c(1, 2))
  p_data <- p_data[valid_idx, ]
  
  if(nrow(p_data) < 20) next
  
  # Ground truth
  true_ch1 <- ifelse(p_data$Resp == 1, 1, 0)
  true_rt <- p_data$RT
  
  # 1. WSLS
  wsls_res <- eval_metrics_wsls(phi_test, p_data$Resp, p_data$F, p_data$RT)
  pr_wsls <- pr.curve(scores.class0 = wsls_res$prob_ch1, weights.class0 = true_ch1)$auc.integral
  rmse_wsls <- sqrt(mean((wsls_res$exp_rt - true_rt)^2))
  
  # 2. Lesioned ECCM
  lesion_res <- eval_metrics_eccm(phi_test, p_data$Resp, p_data$F, p_data$RT, TRUE)
  pr_lesion <- pr.curve(scores.class0 = lesion_res$prob_ch1, weights.class0 = true_ch1)$auc.integral
  rmse_lesion <- sqrt(mean((lesion_res$exp_rt - true_rt)^2))
  
  # 3. Intact ECCM
  intact_res <- eval_metrics_eccm(phi_test, p_data$Resp, p_data$F, p_data$RT, FALSE)
  pr_intact <- pr.curve(scores.class0 = intact_res$prob_ch1, weights.class0 = true_ch1)$auc.integral
  rmse_intact <- sqrt(mean((intact_res$exp_rt - true_rt)^2))
  
  results_list[[length(results_list) + 1]] <- data.frame(
    Participant = p,
    PR_AUC_WSLS = pr_wsls,
    PR_AUC_Lesion = pr_lesion,
    PR_AUC_Intact = pr_intact,
    RMSE_WSLS = rmse_wsls,
    RMSE_Lesion = rmse_lesion,
    RMSE_Intact = rmse_intact
  )
}

df <- do.call(rbind, results_list)

cat("\n--- Mean Results ---\n")
cat(sprintf("WSLS     | PR-AUC: %.4f | RT-RMSE: %.4f\n", mean(df$PR_AUC_WSLS), mean(df$RMSE_WSLS)))
cat(sprintf("Lesioned | PR-AUC: %.4f | RT-RMSE: %.4f\n", mean(df$PR_AUC_Lesion), mean(df$RMSE_Lesion)))
cat(sprintf("Intact   | PR-AUC: %.4f | RT-RMSE: %.4f\n", mean(df$PR_AUC_Intact), mean(df$RMSE_Intact)))

cat("\n--- Paired T-Tests (Intact vs WSLS) ---\n")
print(t.test(df$PR_AUC_Intact, df$PR_AUC_WSLS, paired=TRUE, alternative="greater"))
print(t.test(df$RMSE_Intact, df$RMSE_WSLS, paired=TRUE, alternative="less"))

cat("\n--- Paired T-Tests (Intact vs Lesioned) ---\n")
print(t.test(df$PR_AUC_Intact, df$PR_AUC_Lesion, paired=TRUE, alternative="greater"))
print(t.test(df$RMSE_Intact, df$RMSE_Lesion, paired=TRUE, alternative="less"))

write.csv(df, "docs/metric_comparisons_prauc_rmse.csv", row.names=FALSE)
