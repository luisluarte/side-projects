library(cmdstanr)
fit_base <- readRDS("/home/DCCS5/cerebellum_project/results/baseline_urgency.rds")
print(fit_base$summary("a_base_raw"))