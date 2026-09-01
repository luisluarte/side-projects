library(loo)
cat("\n=== M012 ===\n")
l1 <- readRDS("/home/DCCS5/cerebellum_project/results/loo_m012_ctrl_n30.rds")
print(l1)

cat("\n=== V-OPT ===\n")
l2 <- readRDS("/home/DCCS5/cerebellum_project/results/loo_vopt_n30.rds")
print(l2)
