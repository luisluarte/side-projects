library(cmdstanr)
library(dplyr)
library(readr)

cat("MAGI LOOP 1: Benchmark reduce_sum implementation\n")

dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types=FALSE)
epistemic <- readRDS("results/epistemic_geometry.rds")

# Subsample N=30 subjects for benchmark
test_subjs <- head(epistemic$test_subjs, 30)

test_dat <- dat_raw %>% 
    filter(participant_id %in% test_subjs) %>%
    arrange(participant_id, ttp) %>%
    group_by(participant_id) %>%
    mutate(RT = (ttr-ttp)/1000, Boundary = ifelse(Resp==2, 1, 2), ITI = (ttp - lag(ttr))/1000) %>%
    ungroup() %>% filter(RT>0.1, RT<3.0, !is.na(Resp), !is.na(F)) %>%
    mutate(ITI = ifelse(is.na(ITI) | ITI < 0, median(ITI, na.rm=TRUE), ITI), 
           subj_idx = as.integer(as.factor(participant_id)))

subj_indices <- test_dat %>%
    mutate(row_num = row_number()) %>%
    group_by(subj_idx) %>%
    summarise(start_idx = min(row_num), end_idx = max(row_num)) %>%
    arrange(subj_idx)

min_rt_df <- test_dat %>% group_by(subj_idx) %>% summarise(min_rt = min(RT)) %>% arrange(subj_idx)

stan_data <- list(
    N_trials = nrow(test_dat),
    N_subj = max(test_dat$subj_idx),
    subj = test_dat$subj_idx,
    resp = test_dat$Boundary,
    reward = test_dat$F,
    rt = test_dat$RT,
    iti = test_dat$ITI,
    min_rt = min_rt_df$min_rt,
    W_exp = matrix(rnorm(max(test_dat$subj_idx) * 32, 0, 1), nrow=max(test_dat$subj_idx), ncol=32),
    start_idx = subj_indices$start_idx,
    end_idx = subj_indices$end_idx,
    theta_mean = epistemic$theta_train_mean,
    L_Sigma = t(chol(epistemic$Sigma_train))
)

init_fun <- function() {
    list(
        theta_raw = rnorm(9, 0, 0.01),
        sigma = runif(9, 0.01, 0.05),
        z = matrix(rnorm(9 * stan_data$N_subj, 0, 0.01), nrow=9, ncol=stan_data$N_subj)
    )
}

# Add stan_threads = TRUE for reduce_sum compilation
mod <- cmdstan_model("src/stan/m006_strict_hmc.stan", cpp_options = list(stan_threads = TRUE))

t0 <- Sys.time()
fit <- mod$sample(
    data = stan_data,
    chains = 4,
    parallel_chains = 4,
    threads_per_chain = 8,
    iter_warmup = 10,      
    iter_sampling = 0,    
    metric = "diag_e",
    adapt_engaged = TRUE,
    init = init_fun,
    max_treedepth = 10,
    refresh = 1
)
t1 <- Sys.time()
cat(sprintf("\nBENCHMARK LOOP 1 COMPLETED. Total Time: %s seconds\n", as.numeric(difftime(t1, t0, units="secs"))))
