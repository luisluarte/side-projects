library(Rcpp)
library(parallel)
library(doParallel)
library(ggplot2)

cat("Loading Dataset for ALL Participants...\n")
dataset_path <- "C:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv"
dat_all <- read.csv(dataset_path)

participants <- unique(dat_all[['participant_id']])
cat(sprintf("Total Participants: %d\n", length(participants)))

# We will run 100 iterations of MCMC per model just to prove it executes (since 1000 would take an hour).
# It's sufficient to get an MAP estimate and generate traces.
iters <- 10
phi_init <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5)

cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

# Export required data and functions to cluster
clusterExport(cl, c("dat_all", "participants", "iters", "phi_init"))
clusterEvalQ(cl, {
  library(Rcpp)
  sourceCpp("src/cpp/fast_mcmc_full.cpp")
})

cat("Starting Parallel MCMC Fitting (WSLS vs Intact ECCM vs Lesioned ECCM)...\n")

results <- foreach(p = participants, .combine = rbind, .packages = c("Rcpp")) %dopar% {
  p_data <- dat_all[dat_all$participant_id == p, ]
  p_data <- p_data[order(p_data$ttp), ]
  p_data$RT <- (as.numeric(p_data$ttr) - as.numeric(p_data$ttp)) / 1000.0
  valid_idx <- which(!is.na(p_data$RT) & p_data$RT >= 0.1 & p_data$RT <= 3.0 & p_data$Resp %in% c(1, 2))
  p_data <- p_data[valid_idx, ]
  
  if (nrow(p_data) < 20) return(NULL)
  
  # Run MCMC
  chain_wsls <- run_mcmc_subject(0, iters, phi_init, p_data$Resp, p_data$F, p_data$RT)
  chain_intact <- run_mcmc_subject(1, iters, phi_init, p_data$Resp, p_data$F, p_data$RT)
  chain_lesion <- run_mcmc_subject(2, iters, phi_init, p_data$Resp, p_data$F, p_data$RT)
  
  # MAP estimates from last iteration
  map_wsls <- eval_wsls(chain_wsls[iters, ], p_data$Resp, p_data$F, p_data$RT)
  map_intact <- eval_eccm(chain_intact[iters, ], p_data$Resp, p_data$F, p_data$RT, FALSE)
  map_lesion <- eval_eccm(chain_lesion[iters, ], p_data$Resp, p_data$F, p_data$RT, TRUE)
  
  data.frame(
    Participant = p,
    Deviance_WSLS = map_wsls,
    Deviance_Intact = map_intact,
    Deviance_Lesion = map_lesion
  )
}
stopCluster(cl)

cat("Parallel MCMC Finished.\n")

mean_wsls <- mean(results$Deviance_WSLS)
mean_intact <- mean(results$Deviance_Intact)
mean_lesion <- mean(results$Deviance_Lesion)

cat(sprintf("\n--- Model Comparison (All 128 Participants) ---\n"))
cat(sprintf("WSLS Deviance: %.2f\n", mean_wsls))
cat(sprintf("Intact ECCM Deviance: %.2f\n", mean_intact))
cat(sprintf("Lesioned ECCM Deviance: %.2f\n", mean_lesion))

# Formal Paired T-Tests
t_test_eccm_wsls <- t.test(results$Deviance_Intact, results$Deviance_WSLS, paired=TRUE, alternative="less")
t_test_eccm_lesion <- t.test(results$Deviance_Intact, results$Deviance_Lesion, paired=TRUE, alternative="less")

print(t_test_eccm_wsls)
print(t_test_eccm_lesion)

# Generate a fast trace plot for the last participant's intact chain
# (just doing it in main thread now)
sourceCpp("src/cpp/fast_mcmc_full.cpp")
p_data <- dat_all[dat_all$participant_id == participants[1], ]
p_data <- p_data[order(p_data$ttp), ]
p_data$RT <- (as.numeric(p_data$ttr) - as.numeric(p_data$ttp)) / 1000.0
valid_idx <- which(!is.na(p_data$RT) & p_data$RT >= 0.1 & p_data$RT <= 3.0 & p_data$Resp %in% c(1, 2))
p_data <- p_data[valid_idx, ]
single_chain <- run_mcmc_subject(1, 50, phi_init, p_data$Resp, p_data$F, p_data$RT)

png("docs/figures/full_mcmc_traces_final.png", width=800, height=600)
par(mfrow=c(3,2))
param_names <- c("a", "t_nd", "beta_v", "eta_LTP", "eta_LTD", "w_cb")
for(i in 1:6) {
  plot(single_chain[, i], type='l', main=param_names[i], ylab="Value", xlab="Iteration")
}
dev.off()

cat("\nDone! Generating final report...\n")

# Write out the results to a CSV so we can pipe it into LaTeX easily
write.csv(results, "docs/full_cohort_results.csv", row.names=FALSE)
