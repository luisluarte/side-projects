library(loo)
loo_v2 <- readRDS("/home/DCCS5/cerebellum_project/results/loo_v2_n15.rds")
loo_m011a <- readRDS("/home/DCCS5/cerebellum_project/results/loo_m011a_n15.rds")
loo_m011b <- readRDS("/home/DCCS5/cerebellum_project/results/loo_m011b_n15.rds")

cat("=========================================\n")
cat("LOO COMPARISON: BASELINE V2 vs M011a\n")
cat("=========================================\n")
print(loo_compare(loo_v2, loo_m011a))

cat("\n=========================================\n")
cat("LOO COMPARISON: BASELINE V2 vs M011b\n")
cat("=========================================\n")
print(loo_compare(loo_v2, loo_m011b))

cat("\n=========================================\n")
cat("LOO COMPARISON: ALL MODELS\n")
cat("=========================================\n")
print(loo_compare(list(V2=loo_v2, M011a=loo_m011a, M011b=loo_m011b)))
