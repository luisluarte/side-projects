library(loo)
res <- readRDS("results/final_loo_comparison.rds")

# 1. Check dimensions
cat("Dimensions M006: ", dim(res$loo_m006$pointwise), "\n")
cat("Dimensions Base: ", dim(res$loo_base$pointwise), "\n")

# 2. Check for missing or infinite values
cat("M006 Inf/NA: ", sum(is.infinite(res$loo_m006$pointwise[,"elpd_loo"])), sum(is.na(res$loo_m006$pointwise[,"elpd_loo"])), "\n")
cat("Base Inf/NA: ", sum(is.infinite(res$loo_base$pointwise[,"elpd_loo"])), sum(is.na(res$loo_base$pointwise[,"elpd_loo"])), "\n")

# 3. Sum of raw log likelihoods (not LOO) to see in-sample fit
cat("M006 In-Sample ELPD (sum): ", sum(res$loo_m006$pointwise[,"elpd_loo"]), "\n")
cat("Base In-Sample ELPD (sum): ", sum(res$loo_base$pointwise[,"elpd_loo"]), "\n")

# 4. Compare distribution of pointwise ELPD differences
diff_elpd <- res$loo_m006$pointwise[,"elpd_loo"] - res$loo_base$pointwise[,"elpd_loo"]
cat("Mean pointwise diff: ", mean(diff_elpd), "\n")
cat("Median pointwise diff: ", median(diff_elpd), "\n")
cat("Max pointwise diff: ", max(diff_elpd), "\n")
cat("Min pointwise diff: ", min(diff_elpd), "\n")
cat("Trials where M006 is better: ", sum(diff_elpd > 0), " / ", length(diff_elpd), "\n")

# 5. Check if there are extreme outliers driving the entire 800 margin
sorted_diffs <- sort(diff_elpd, decreasing=TRUE)
cat("Top 10 highest positive diffs (M006 better): ", head(sorted_diffs, 10), "\n")
cat("Top 10 highest negative diffs (Base better): ", head(sort(diff_elpd), 10), "\n")
cat("Sum of top 100 diffs: ", sum(head(sorted_diffs, 100)), "\n")

# 6. Pareto K summary
cat("M006 Pareto > 0.7: ", sum(res$loo_m006$pointwise[,"influence_pareto_k"] > 0.7), "\n")
cat("Base Pareto > 0.7: ", sum(res$loo_base$pointwise[,"influence_pareto_k"] > 0.7), "\n")
