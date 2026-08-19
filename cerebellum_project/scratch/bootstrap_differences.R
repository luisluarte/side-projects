library(ggplot2)

cat("Loading cross-validation results...\n")
res <- read.csv("C:/Users/DCCS5/Documents/GitHub/side-projects/cerebellum_project/results/tables/ngrc_cfmr_biological_benchmark.csv")

diff_pure_ngrc <- res$Pure_CFMR_NLL - res$NGRC_CFMR_NLL

set.seed(42)
n_boot <- 10000
n_subj <- nrow(res)

boot_ngrc <- numeric(n_boot)

cat(sprintf("Running %d bootstrap iterations...\n", n_boot))
for(i in 1:n_boot) {
    idx <- sample(1:n_subj, n_subj, replace = TRUE)
    boot_ngrc[i] <- mean(diff_pure_ngrc[idx])
}

ci_ngrc <- quantile(boot_ngrc, probs = c(0.025, 0.975))

cat("\n--- BOOTSTRAPPED MEAN DIFFERENCES (95% CI) ---\n")
cat(sprintf("Pure CFMR - NGRC CFMR: Mean = %.3f, 95%% CI = [%.3f, %.3f]\n", mean(boot_ngrc), ci_ngrc[1], ci_ngrc[2]))

# Note: Positive values mean NGRC CFMR has lower NLL (better performance).

boot_df <- data.frame(
    Difference = boot_ngrc,
    Comparison = factor(rep("Pure CFMR vs NGRC CFMR", n_boot))
)

p <- ggplot(boot_df, aes(x = Difference, fill = Comparison)) +
    geom_density(alpha = 0.6) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "black", linewidth = 1) +
    theme_minimal() +
    labs(title = "Bootstrapped NLL Differences (N=128)",
         x = "Mean NLL Difference (Pure CFMR - NGRC CFMR)",
         y = "Density") +
    scale_fill_manual(values = c("Pure CFMR vs NGRC CFMR" = "#8e44ad")) +
    annotate("text", x = 0, y = 0, label = "NGRC CFMR Better ->", hjust = -0.1, vjust = -1) +
    annotate("text", x = 0, y = 0, label = "<- Pure CFMR Better", hjust = 1.1, vjust = -1)

ggsave("C:/Users/DCCS5/Documents/GitHub/side-projects/cerebellum_project/results/tables/bootstrapped_ngrc_differences_density.png", p, width = 8, height = 5)
cat("Saved plot to results/tables/bootstrapped_ngrc_differences_density.png\n")
