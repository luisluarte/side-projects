library(Rcpp)

cat("Loading dataset...\n")
dataset_path <- "data/raw/behavioral_compilate.csv"
if (!file.exists(dataset_path)) {
  dataset_path <- "c:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv"
}

dat_all <- read.csv(dataset_path)
dat_all[['RT']] <- (as.numeric(dat_all[['ttr']]) - as.numeric(dat_all[['ttp']])) / 1000.0
dat_all <- dat_all[!is.na(dat_all[['RT']]) & dat_all[['RT']] >= 0.1 & dat_all[['RT']] <= 3.0 & dat_all[['Resp']] %in% c(1, 2), ]

set.seed(42)
participants <- sample(unique(dat_all[['participant_id']]), 10)
p_data <- dat_all[dat_all$participant_id == participants[1], ]
p_data <- p_data[order(p_data$ttp), ]

sourceCpp("src/cpp/check_raw.cpp")

phi_wsls <- c(1.0, 0.5, 2.0, 0.5, 0.5)
phi_eccm <- c(1.0, 0.5, 2.0, 0.5, 0.5, 0.5)

res <- eval_both_raw(phi_wsls, phi_eccm, p_data$Resp, p_data$F, p_data$RT)

D_wsls <- res$D_wsls
D_eccm <- res$D_eccm

# Calculate Slope m_D manually for both
T <- length(D_wsls)
t_seq <- 1:T
mean_t <- mean(t_seq)
m_wsls <- sum((t_seq - mean_t) * (D_wsls - mean(D_wsls))) / sum((t_seq - mean_t)^2)
m_eccm <- sum((t_seq - mean_t) * (D_eccm - mean(D_eccm))) / sum((t_seq - mean_t)^2)

cat(sprintf("WSLS Raw Deviance: %.2f | Slope m_D: %.5f | Penalized: %.2f\n", sum(D_wsls), m_wsls, sum(D_wsls) + abs(m_wsls)))
cat(sprintf("ECCM Raw Deviance: %.2f | Slope m_D: %.5f | Penalized: %.2f\n", sum(D_eccm), m_eccm, sum(D_eccm) + abs(m_eccm)))
