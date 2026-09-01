pacman::p_load(tidyverse)

df_grid <- expand.grid(struct_id=c(1,2,3,4), kappa=c(0.1,0.5,0.8,0.9,0.99), K_sa=c(2,5,10,20,50))
df_grid$model_idx <- 1:100
df_grid$ModelID <- sprintf("CC_Model_%03d", df_grid$model_idx)
df_grid <- df_grid %>% mutate(
    Activation = case_when(struct_id==1 ~ "Tanh", struct_id==2 ~ "ReLU", struct_id==3 ~ "Sigmoid", TRUE ~ "Linear")
)

df <- read_csv("results/tables/thermo_sudoku_sweep.csv", show_col_types=FALSE) %>% 
      filter(SubjectID != "SubjectID") %>% 
      mutate(NLL = as.numeric(NLL))

model_stats <- df %>% group_by(ModelID) %>% summarize(
    Mean_NLL = mean(NLL, na.rm=TRUE),
    Median_NLL = median(NLL, na.rm=TRUE),
    Subject_Failures = sum(NLL > 9e5 | is.na(NLL), na.rm=TRUE) 
) %>% 
left_join(df_grid, by="ModelID") %>%
arrange(Mean_NLL)

write_csv(model_stats, "results/tables/thermo_nll_summary.csv")

sink("results/tables/thermo_nll_top10.txt")
cat("=== TOP 10 THERMODYNAMIC ARCHITECTURES BY NLL ===\n")
print(head(model_stats %>% dplyr::select(ModelID, Activation, kappa, K_sa, Mean_NLL, Subject_Failures), 10), row.names=FALSE)
cat("\n=== WORST 5 ARCHITECTURES ===\n")
print(tail(model_stats %>% dplyr::select(ModelID, Activation, kappa, K_sa, Mean_NLL, Subject_Failures), 5), row.names=FALSE)
sink()
