pacman::p_load(tidyverse, Rcpp, cmaes, parallel)

CORES <- parallel::detectCores()
Rcpp::sourceCpp("magi_grand_phylogeny.cpp") 

dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types = FALSE)
dat_raw$Reward <- dat_raw$F
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>% 
    group_by(participant_id) %>% 
    dplyr::mutate(RT = (ttr - ttp) / 1000, Boundary = ifelse(Resp == 2, 1, 0)) %>% 
    ungroup() %>% dplyr::filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(Reward)) %>%
    dplyr::mutate(participant_idx = as.integer(as.factor(participant_id)))

S <- max(dat_clean$participant_idx)
d_list <- split(dat_clean, dat_clean$participant_idx)

cat("Evaluating Baseline Wald for Paired Statistics...\n")
run_base <- function(s_idx) {
    tryCatch({
        d <- d_list[[s_idx]]
        obj <- function(p) { v <- get_nll_base(p, d$Boundary+1, d$Reward, d$RT); if(is.nan(v)) 1e6 else v }
        res <- tryCatch(cma_es(rep(0, 4), obj, control=list(maxit=50, sigma=0.5)), error = function(e) list(value=NA))
        data.frame(SubjectID=s_idx, Base_NLL=res$value)
    }, error = function(e) data.frame(SubjectID=s_idx, Base_NLL=NA))
}
df_base <- bind_rows(mclapply(1:S, run_base, mc.cores=CORES))

df_cand <- read_csv("results/tables/thermo_sudoku_sweep.csv", show_col_types=FALSE) %>% 
    filter(SubjectID != "SubjectID") %>% 
    mutate(SubjectID = as.integer(SubjectID), NLL = as.numeric(NLL))

df_merged <- inner_join(df_cand, df_base, by="SubjectID") %>% drop_na()

stats_list <- lapply(unique(df_merged$ModelID), function(m_id) {
    d_sub <- df_merged %>% filter(ModelID == m_id)
    if(nrow(d_sub) > 5) {
        # Wilcoxon requires variance
        if(sd(d_sub$NLL) == 0 && sd(d_sub$Base_NLL) == 0) return(NULL)
        wt <- tryCatch(wilcox.test(d_sub$NLL, d_sub$Base_NLL, paired=TRUE), error=function(e) list(p.value=1.0))
        mean_diff <- mean(d_sub$NLL - d_sub$Base_NLL)
        sd_diff <- sd(d_sub$NLL - d_sub$Base_NLL)
        data.frame(
            ModelID = m_id,
            Mean_NLL = mean(d_sub$NLL),
            Diff_NLL = mean_diff, 
            P_Value = wt$p.value,
            Cohens_D = mean_diff / sd_diff
        )
    } else {
        NULL
    }
})

df_stats <- bind_rows(stats_list)

df_grid <- expand.grid(struct_id=c(1,2,3,4), kappa=c(0.1,0.5,0.8,0.9,0.99), K_sa=c(2,5,10,20,50))
df_grid$model_idx <- 1:100
df_grid$ModelID <- sprintf("CC_Model_%03d", df_grid$model_idx)
df_grid <- df_grid %>% mutate(Activation = case_when(struct_id==1 ~ "Tanh", struct_id==2 ~ "ReLU", struct_id==3 ~ "Sigmoid", TRUE ~ "Linear"))

final_stats <- inner_join(df_stats, df_grid, by="ModelID") %>% arrange(Mean_NLL)
write_csv(final_stats, "results/tables/thermo_nll_stats.csv")

sink("results/tables/thermo_nll_stats_summary.txt")
cat("=== STATISTICAL COMPARISON TO BASELINE WALD (H0) ===\n")
cat(sprintf("Baseline Wald Mean NLL: %.2f\n\n", mean(df_base$Base_NLL, na.rm=T)))
print(head(final_stats %>% dplyr::select(ModelID, Activation, kappa, K_sa, Mean_NLL, Diff_NLL, P_Value, Cohens_D), 10), row.names=FALSE)
cat("\n=== WORST 5 ===\n")
print(tail(final_stats %>% dplyr::select(ModelID, Activation, kappa, K_sa, Mean_NLL, Diff_NLL, P_Value, Cohens_D), 5), row.names=FALSE)
sink()
