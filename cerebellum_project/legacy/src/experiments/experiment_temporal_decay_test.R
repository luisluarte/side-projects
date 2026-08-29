library(Rcpp)
library(doParallel)
library(foreach)

cat("Loading Dataset and Sampling N=30...\n")
dataset_path <- "C:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv"
dat_all <- read.csv(dataset_path)

set.seed(42)
participants <- sample(unique(dat_all[['participant_id']]), 30)

iters <- 25
init_phi_intact <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5)
init_phi_decay  <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5, log(1.0))

cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

cat("Running parallel MCMC fits...\n")
results_list <- foreach(p = participants, .packages=c("Rcpp")) %dopar% {
  sourceCpp("src/fitting_procedures/mcmc_sampler.cpp")
  
  p_data <- dat_all[dat_all$participant_id == p, ]
  p_data <- p_data[order(p_data$ttp), ]
  p_data$RT <- (as.numeric(p_data$ttr) - as.numeric(p_data$ttp)) / 1000.0
  p_data$delta_t <- c(NA, diff(as.numeric(p_data$ttp)) / 1000.0) 
  p_data$delta_t[is.na(p_data$delta_t)] <- 2.0
  
  valid_idx <- which(!is.na(p_data$RT) & p_data$RT >= 0.1 & p_data$RT <= 3.0 & p_data$Resp %in% c(1, 2))
  p_data <- p_data[valid_idx, ]
  
  if (nrow(p_data) < 20) {
    return(NULL)
  }
  
  chain_intact <- run_mcmc_subject(1, iters, init_phi_intact, p_data$Resp, p_data$F, p_data$RT)
  dev_intact <- eval_eccm_intact(as.numeric(chain_intact[iters, ]), p_data$Resp, p_data$F, p_data$RT)
  
  chain_decay <- run_mcmc_subject(3, iters, init_phi_decay, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  dev_decay <- eval_eccm_temporal_decay(as.numeric(chain_decay[iters, ]), p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  
  data.frame(subject = p, dev_intact = dev_intact, dev_decay = dev_decay)
}

stopCluster(cl)

results <- do.call(rbind, results_list)

cat("\n--- Temporal Decay Benchmark ---\n")
print(head(results))
cat("\nPaired T-Test (Intact vs Temporal Decay):\n")
print(t.test(results$dev_intact, results$dev_decay, paired=TRUE, alternative="greater")) # H1: Intact deviance > Decay deviance

write.csv(results, "results/tables/temporal_decay_benchmark.csv", row.names=FALSE)
