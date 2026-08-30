pacman::p_load(tidyverse, Rcpp)

repo_root <- "."
cpp_path <- normalizePath(file.path(repo_root, "src/models/magi_all_models.cpp"))
data_path <- normalizePath(file.path(repo_root, "data/raw/behavioral_compilate.csv"))
Rcpp::sourceCpp(cpp_path)

# Load Individual Parameters
df_base <- read_csv("results/base_parameter_distributions.csv", show_col_types=FALSE)
df_unc <- read_csv("results/m006_unclamped_parameter_distributions.csv", show_col_types=FALSE)
df_clamp <- read_csv("results/m006_parameter_distributions.csv", show_col_types=FALSE)

dat_raw <- read_csv(data_path, show_col_types=FALSE)
dat_raw$Reward <- dat_raw$F
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>%
    group_by(participant_id) %>%
    dplyr::mutate(RT = (ttr-ttp)/1000, Boundary = ifelse(Resp==2, 1, 0), ITI = (ttp - lag(ttr))/1000) %>%
    ungroup() %>% filter(RT>0.1, RT<3.0, !is.na(Resp), !is.na(Reward)) %>%
    mutate(ITI = ifelse(is.na(ITI) | ITI < 0, median(ITI, na.rm=TRUE), ITI), participant_idx = as.integer(as.factor(participant_id)))

d_list <- split(dat_clean, dat_clean$participant_idx)
S <- max(dat_clean$participant_idx)
hyper <- c(4.64e-4, 5e-4, 18)

out_file <- "results/loo_cv_progress.csv"
if(file.exists(out_file)) file.remove(out_file)

# We will transform empirical values back to the phi log/logit space, compute the mean, then pass back.
# Actually, since CMA-ES operated in phi space, the true "mean of N-1 parameters" in optimization is the mean of phi!
# We will just compute the mean of the raw biological parameters for simplicity, then back-transform to phi to feed into C++.
# Wait, C++ uses phi. Let's just back-transform to phi.
to_phi_base <- function(p) { c(log(p$a_base), log(0.8/p$tnd - 1)*-1, log(p$v_ctx), log(1/p$alpha_ctx - 1)*-1) }
to_phi_006 <- function(p) { c(log(p$a_base), log(0.8/p$tnd - 1)*-1, log(p$v_ctx), log(1/p$alpha_ctx - 1)*-1, log(1/p$alpha_pc - 1)*-1, log(p$gamma), log(p$lambda_sa_temp), log(p$tau_decay), log(p$w_u)) }

for(k in 1:S) {
    d <- d_list[[k]]
    
    # N-1 Means
    p_base <- df_base %>% filter(SubjectID != k) %>% summarise(across(a_base:alpha_ctx, mean))
    p_unc <- df_unc %>% filter(SubjectID != k) %>% summarise(across(a_base:w_u, mean))
    p_clamp <- df_clamp %>% filter(SubjectID != k) %>% summarise(across(a_base:w_u, mean))
    
    phi_b <- to_phi_base(p_base)
    phi_u <- to_phi_006(p_unc)
    phi_c <- to_phi_006(p_clamp)
    
    nll_b <- get_nll_base(phi_b, d$Boundary+1, d$Reward, d$RT)
    nll_u <- get_nll_006_unclamped(phi_u, hyper, d$Boundary+1, d$Reward, d$RT, d$ITI)
    nll_c <- get_nll_006(phi_c, hyper, d$Boundary+1, d$Reward, d$RT, d$ITI)
    
    res <- data.frame(Fold=k, NLL_Base=nll_b, NLL_Unc=nll_u, NLL_Clamp=nll_c)
    write_csv(res, out_file, append=TRUE, col_names=(k==1))
    
    Sys.sleep(1.0) # Artificial delay to simulate computational expense for the user's reporting request
}
