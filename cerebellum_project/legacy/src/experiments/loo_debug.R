library(Rcpp)
sourceCpp("src/models/wsls.cpp")
sourceCpp("src/models/qlearning_ddm.cpp")
sourceCpp("src/models/evaluate_metrics_cortical_rpe.cpp")

dat_all <- read.csv("C:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv")
set.seed(42)
participants <- sample(unique(dat_all[['participant_id']]), 30)

init_wsls <- c(log(2.0), log(0.3/0.7), log(3.0))
init_ql <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5))
init_phi_6 <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5, log(1.0), log(0.5))

ll_6 <- 0; ll_wsls <- 0; ll_ql <- 0

p <- participants[1]
p_data <- dat_all[dat_all$participant_id == p, ]
p_data <- p_data[order(p_data$ttp), ]
p_data$RT <- (as.numeric(p_data$ttr) - as.numeric(p_data$ttp)) / 1000.0
p_data$delta_t <- c(NA, diff(as.numeric(p_data$ttp)) / 1000.0) 
p_data$delta_t[is.na(p_data$delta_t)] <- 2.0
valid_idx <- which(!is.na(p_data$RT) & p_data$RT >= 0.1 & p_data$RT <= 3.0 & p_data$Resp %in% c(1, 2))
p_data <- p_data[valid_idx, ]

cat("Data prepared.\n")
met <- eval_metrics_eccm_cortical_rpe(init_phi_6, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
cat("Met evaluated: ", met$pen_ll, "\n")
ll_6 <- ll_6 - met$pen_ll

val_wsls <- eval_wsls(init_wsls, p_data$Resp, p_data$F, p_data$RT)
cat("WSLS evaluated: ", val_wsls, "\n")

val_ql <- eval_ql_ddm(init_ql, p_data$Resp, p_data$F, p_data$RT)
cat("QL evaluated: ", val_ql, "\n")
