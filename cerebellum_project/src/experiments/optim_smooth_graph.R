library(Rcpp)

sourceCpp("src/models/eccm_smooth_graph.cpp")

dat_all <- read.csv("C:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv")
set.seed(42)
participants <- sample(unique(dat_all[['participant_id']]), 30)

eval_subject_ll <- function(phi, resp, out, rt, delta_t) {
  -eval_eccm_smooth_graph(phi, resp, out, rt, delta_t)
}

# phi: a, t_nd, beta_v, eta_win, eta_loss, w_cb, lambda_shift, gamma_suppress, alpha_diff
init_phi <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.1/0.9), log(0.88/0.12), 0.5, log(1.0), log(0.5), log(0.1)) # try slow diffusion log(0.1)

total_ll_init <- 0
for (p in participants) {
  p_data <- dat_all[dat_all$participant_id == p, ]
  p_data <- p_data[order(p_data$ttp), ]
  p_data$RT <- (as.numeric(p_data$ttr) - as.numeric(p_data$ttp)) / 1000.0
  p_data$delta_t <- c(NA, diff(as.numeric(p_data$ttp)) / 1000.0) 
  p_data$delta_t[is.na(p_data$delta_t)] <- 2.0
  valid_idx <- which(!is.na(p_data$RT) & p_data$RT >= 0.1 & p_data$RT <= 3.0 & p_data$Resp %in% c(1, 2))
  p_data <- p_data[valid_idx, ]
  if (nrow(p_data) < 20) next
  total_ll_init <- total_ll_init + eval_subject_ll(init_phi, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
}

cat("Init Dev:", -total_ll_init, "\n")

# Try to optim
opt_res <- optim(
  par = init_phi,
  fn = function(phi) {
    ll <- 0
    for (p in participants) {
      p_data <- dat_all[dat_all$participant_id == p, ]
      p_data <- p_data[order(p_data$ttp), ]
      p_data$RT <- (as.numeric(p_data$ttr) - as.numeric(p_data$ttp)) / 1000.0
      p_data$delta_t <- c(NA, diff(as.numeric(p_data$ttp)) / 1000.0) 
      p_data$delta_t[is.na(p_data$delta_t)] <- 2.0
      valid_idx <- which(!is.na(p_data$RT) & p_data$RT >= 0.1 & p_data$RT <= 3.0 & p_data$Resp %in% c(1, 2))
      p_data <- p_data[valid_idx, ]
      if (nrow(p_data) < 20) next
      ll <- ll + eval_subject_ll(phi, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
    }
    return(-ll) # minimize negative log likelihood
  },
  method = "Nelder-Mead",
  control = list(maxit=50) # Just 50 iters to see if it moves
)

cat("Optimized Dev:", opt_res$value, "\n")
cat("Optimized phi:\n")
print(opt_res$par)

sink("output_smooth_optim.txt")
cat(sprintf("Init Dev: %.2f\n", -total_ll_init))
cat(sprintf("Optimized Dev: %.2f\n", opt_res$value))
cat("Optimized phi:\n")
print(opt_res$par)
sink()
