library(dplyr)
library(readr)
library(cmaes)
library(Rcpp)
library(parallel)

cat("VM Containment Audit: \n")
system("uname -a")

cat("\nLoading datasets and C++ models...\n")
sourceCpp("src/models/pooled_map_m006.cpp")
dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types=FALSE)
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>%
    group_by(participant_id) %>%
    dplyr::mutate(RT = (ttr-ttp)/1000, Boundary = ifelse(Resp==2, 1, 2), ITI = (ttp - lag(ttr))/1000) %>%
    ungroup() %>% filter(RT>0.1, RT<3.0, !is.na(Resp), !is.na(F)) %>%
    mutate(ITI = ifelse(is.na(ITI) | ITI < 0, median(ITI, na.rm=TRUE), ITI), 
           participant_idx = as.integer(as.factor(participant_id)))

set.seed(42)

subjects <- unique(dat_clean$participant_id)
N_total <- length(subjects)
N_train <- floor(0.3 * N_total)
N_test <- N_total - N_train

train_subjs <- sample(subjects, N_train)
test_subjs <- setdiff(subjects, train_subjs)

cat(sprintf("\n[Phase I: Epistemological Partitioning]\n"))
cat(sprintf("Total Subjects: %d | Training Manifold: %d (30%%) | Inference Manifold: %d (70%%)\n", N_total, N_train, N_test))
cat("Quarantine established. The test set is mathematically isolated.\n")

cat("\nSpawning independent CMA-ES swarms for Training Manifold...\n")
results <- mclapply(train_subjs, function(s) {
    s_dat <- dat_clean %>% filter(participant_id == s)
    s_dat$participant_idx <- 1
    min_rt_s <- min(s_dat$RT)
    W_exp_s <- matrix(rnorm(32, 0, 1), nrow=1, ncol=32)
    
    obj_fn_single <- function(x) {
        val <- get_pooled_map_006(x, s_dat$participant_idx, s_dat$Boundary, 
                                  s_dat$F, s_dat$RT, s_dat$ITI, min_rt_s, W_exp_s)
        if(is.nan(val) || is.infinite(val)) return(1e9)
        return(val)
    }

    init_p <- rnorm(9, 0, 0.1)
    res <- cma_es(init_p, obj_fn_single, control=list(maxit=300, sigma=0.5))
    return(res$par)
}, mc.cores = 32)

cat("\n[Phase II: Geometric Extraction]\n")
theta_mat <- do.call(rbind, results)
theta_train_mean <- colMeans(theta_mat)
Sigma_train <- cov(theta_mat)
Sigma_inv <- solve(Sigma_train)

sigma_train <- sqrt(diag(Sigma_train))
Corr_train <- cov2cor(Sigma_train)
L_train <- t(chol(Corr_train))

cat("Empirical Mean Vector (theta_train_mean):\n")
print(theta_train_mean)

cat("\nInverse Covariance / Dense Mass Matrix (M):\n")
print(Sigma_inv[1:4, 1:4])
cat("... (truncated for display)\n")

cat("\nCholesky Factor (L_train) [Lower Triangular]:\n")
print(L_train[1:4, 1:4])
cat("... (truncated for display)\n")

cat("\nEmpirical Std Devs (sigma_train):\n")
print(sigma_train)

dir.create("results", showWarnings=FALSE)
saveRDS(list(
    theta_train_mean = theta_train_mean,
    Sigma_train = Sigma_train,
    Sigma_inv = Sigma_inv,
    L_train = L_train,
    sigma_train = sigma_train,
    test_subjs = test_subjs,
    train_subjs = train_subjs
), "results/epistemic_geometry.rds")

cat("\nExtraction complete. Artifacts saved to results/epistemic_geometry.rds\n")
