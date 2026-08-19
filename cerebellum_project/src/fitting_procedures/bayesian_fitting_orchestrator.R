library(Rcpp)
library(doParallel)
library(foreach)

# Source the single MCMC sampler which natively includes the models
sourceCpp("src/fitting_procedures/mcmc_sampler.cpp")

run_full_cohort_mcmc <- function(dataset_path, out_csv_path, iters = 10) {
  cat("Loading Dataset for ALL Participants...\n")
  dat_all <- read.csv(dataset_path)
  participants <- unique(dat_all[['participant_id']])
  cat(sprintf("Total Participants: %d\n", length(participants)))
  
  phi_init <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5)
  
  cl <- makeCluster(detectCores() - 1)
  registerDoParallel(cl)
  
  cat("Starting Parallel MCMC Fitting (WSLS vs Intact ECCM vs Lesioned ECCM)...\n")
  
  results_list <- foreach(p = participants, .packages=c("Rcpp")) %dopar% {
    sourceCpp("src/fitting_procedures/mcmc_sampler.cpp")
    
    p_data <- dat_all[dat_all$participant_id == p, ]
    p_data <- p_data[order(p_data$ttp), ]
    p_data$RT <- (as.numeric(p_data$ttr) - as.numeric(p_data$ttp)) / 1000.0
    valid_idx <- which(!is.na(p_data$RT) & p_data$RT >= 0.1 & p_data$RT <= 3.0 & p_data$Resp %in% c(1, 2))
    p_data <- p_data[valid_idx, ]
    
    if(nrow(p_data) < 20) return(NULL)
    
    # 0=WSLS, 1=Intact, 2=Lesioned
    chain_wsls <- run_mcmc_subject(0, iters, phi_init, p_data$Resp, p_data$F, p_data$RT)
    chain_intact <- run_mcmc_subject(1, iters, phi_init, p_data$Resp, p_data$F, p_data$RT)
    chain_lesion <- run_mcmc_subject(2, iters, phi_init, p_data$Resp, p_data$F, p_data$RT)
    
    dev_wsls <- eval_wsls(chain_wsls[iters, ], p_data$Resp, p_data$F, p_data$RT)
    dev_intact <- eval_eccm_intact(chain_intact[iters, ], p_data$Resp, p_data$F, p_data$RT)
    dev_lesion <- eval_eccm_lesioned(chain_lesion[iters, ], p_data$Resp, p_data$F, p_data$RT)
    
    data.frame(Participant = p, Deviance_WSLS = dev_wsls, Deviance_Intact = dev_intact, Deviance_Lesion = dev_lesion)
  }
  
  stopCluster(cl)
  cat("Parallel MCMC Finished.\n")
  
  results <- do.call(rbind, results_list)
  write.csv(results, out_csv_path, row.names = FALSE)
  return(results)
}
