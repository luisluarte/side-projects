# ==============================================================================
# DE_audit_diagnostics.R
# Comprehensive Theoretical & Computational Audit of Differential Expression (DE)
# ==============================================================================

pacman::p_load(
  QFeatures,
  limma,
  tidyverse,
  ggplot2,
  patchwork,
  imputeLCMD
)

setwd(this.path::here())
dir.create("../figures", showWarnings = FALSE, recursive = TRUE)

cat("==================================================================\n")
cat("Starting DE Analysis Diagnostic & Benchmark Audit\n")
cat("==================================================================\n\n")

# 1. DATA LOADING AND PREPROCESSING --------------------------------------------
cat("[Step 1] Loading raw data and applying filtering & normalization...\n")

raw_dat <- read_tsv("../data/combined_protein.tsv", show_col_types = FALSE)

intensity_cols <- colnames(raw_dat) %>%
  str_subset(pattern = "(Control|Tratamiento)_[0-9]+ Intensity")

ctrl_target_cols <- intensity_cols[str_detect(intensity_cols, "Control")]
trat_target_cols <- intensity_cols[str_detect(intensity_cols, "Tratamiento")]

dat <- raw_dat %>%
  mutate(across(all_of(intensity_cols), ~ na_if(., 0)))

# Filtering: detected in >=8 total and (>=4 ctrl OR >=4 trat)
valid_proteins <- dat %>%
  rowwise() %>%
  mutate(
    valid_ctrl = sum(!is.na(c_across(all_of(ctrl_target_cols)))),
    valid_trat = sum(!is.na(c_across(all_of(trat_target_cols)))),
    valid_total = valid_ctrl + valid_trat
  ) %>%
  ungroup() %>%
  filter(valid_total >= 8 & (valid_ctrl >= 4 | valid_trat >= 4))

cat(sprintf("Retained %d out of %d proteins (%.1f%%)\n",
            nrow(valid_proteins), nrow(dat), (nrow(valid_proteins)/nrow(dat))*100))

# QFeatures data object
metadata <- data.frame(
  row.names = intensity_cols,
  sample_id = intensity_cols,
  condition = factor(
    if_else(str_detect(intensity_cols, "Control"), "Control", "Treatment"),
    levels = c("Control", "Treatment")
  )
)

valid_df <- as.data.frame(valid_proteins)
fts <- readQFeatures(assayData = valid_df, quantCols = intensity_cols, name = "raw_maxlfq")
colData(fts) <- DataFrame(metadata)

# Log2 Transform & Median Normalization
fts <- logTransform(fts, i = "raw_maxlfq", name = "log2maxlfq")
fts <- normalize(fts, i = "log2maxlfq", name = "normalized_maxlfq", method = "center.median")
norm_matrix <- assay(fts, "normalized_maxlfq")
row_metadata <- rowData(fts[["normalized_maxlfq"]])

genes <- as.character(row_metadata$Gene)
prot_ids <- as.character(row_metadata$Protein.ID)
resolved_labels <- make.unique(ifelse(is.na(genes) | genes == "", prot_ids, genes))
rownames(norm_matrix) <- resolved_labels

# 2. MISSINGNESS CHARACTERIZATION ----------------------------------------------
cat("\n[Step 2] Characterizing missingness (MNAR vs MCAR analysis)...\n")

missing_df <- tibble(
  protein = rownames(norm_matrix),
  mean_intensity = rowMeans(norm_matrix, na.rm = TRUE),
  na_count = rowSums(is.na(norm_matrix)),
  na_prop = na_count / ncol(norm_matrix),
  ctrl_na = rowSums(is.na(norm_matrix[, ctrl_target_cols])),
  trat_na = rowSums(is.na(norm_matrix[, trat_target_cols]))
)

p_missing <- ggplot(missing_df, aes(x = mean_intensity, y = na_prop)) +
  geom_point(alpha = 0.2, color = "#2b5c8f") +
  geom_smooth(method = "gam", color = "#d95f02", se = TRUE) +
  theme_minimal(base_size = 12) +
  labs(
    title = "Missingness Rate vs. Mean Protein Log2 Intensity",
    subtitle = "Assessing Limit of Detection (MNAR) vs Random Missingness (MCAR)",
    x = "Mean Log2 Normalized Intensity",
    y = "Proportion of Missing Values (NAs)"
  )

ggsave("../figures/1_missingness_profile.png", p_missing, width = 8, height = 5, dpi = 300)
cat("Saved figures/1_missingness_profile.png\n")


# 3. BENCHMARK PIPELINES EXECUTION ---------------------------------------------
cat("\n[Step 3] Running candidate DE pipelines for comparison...\n")

design_matrix <- model.matrix(~ 0 + condition, data = colData(fts))
colnames(design_matrix) <- c("conditionControl", "conditionTreatment")

contrast.matrix <- makeContrasts(
  Diff = conditionTreatment - conditionControl,
  levels = design_matrix
)

# Helper function to run limma eBayes pipeline
run_limma_pipeline <- function(expr_mat, pipeline_name) {
  fit <- lmFit(expr_mat, design_matrix)
  fit_contrast <- contrasts.fit(fit, contrast.matrix)
  eb_fit <- eBayes(fit_contrast, trend = TRUE, robust = TRUE)
  
  res <- topTable(eb_fit, coef = "Diff", adjust.method = "BH", number = Inf) %>%
    rownames_to_column("protein") %>%
    as_tibble() %>%
    mutate(pipeline = pipeline_name)
  
  list(fit = eb_fit, res = res)
}

# Pipeline 1: Current MinProb Imputation
cat(" Running Pipeline 1: MinProb Imputation + limma...\n")
fts_minprob <- impute(fts, i = "normalized_maxlfq", name = "imp_minprob", method = "MinProb")
mat_minprob <- assay(fts_minprob, "imp_minprob")
rownames(mat_minprob) <- resolved_labels
p1_out <- run_limma_pipeline(mat_minprob, "MinProb (Current)")

# Pipeline 2: NA-Aware Limma (No Imputation)
cat(" Running Pipeline 2: NA-Aware limma (No Imputation)...\n")
p2_out <- run_limma_pipeline(norm_matrix, "NA-Aware (No Imp)")

# Pipeline 3: QRILC Imputation
cat(" Running Pipeline 3: QRILC Imputation + limma...\n")
fts_qrilc <- impute(fts, i = "normalized_maxlfq", name = "imp_qrilc", method = "QRILC")
mat_qrilc <- assay(fts_qrilc, "imp_qrilc")
rownames(mat_qrilc) <- resolved_labels
p3_out <- run_limma_pipeline(mat_qrilc, "QRILC Imputation")

# Pipeline 4: KNN Imputation
cat(" Running Pipeline 4: KNN Imputation + limma...\n")
fts_knn <- impute(fts, i = "normalized_maxlfq", name = "imp_knn", method = "knn")
mat_knn <- assay(fts_knn, "imp_knn")
rownames(mat_knn) <- resolved_labels
p4_out <- run_limma_pipeline(mat_knn, "KNN Imputation")


# Combine results
all_res <- bind_rows(
  p1_out$res,
  p2_out$res,
  p3_out$res,
  p4_out$res
)


# 4. P-VALUE DISTRIBUTION & CALIBRATION ANALYSIS -------------------------------
cat("\n[Step 4] Analyzing P-value calibration and Genomic Inflation Factor (Lambda)...\n")

calc_lambda <- function(pvals) {
  pvals <- pvals[!is.na(pvals)]
  chisq <- qchisq(1 - pvals, df = 1)
  median(chisq) / qchisq(0.5, df = 1)
}

lambda_df <- all_res %>%
  group_by(pipeline) %>%
  summarise(
    total_proteins = n(),
    raw_p_lt_05 = sum(P.Value < 0.05, na.rm = TRUE),
    fdr_lt_05 = sum(adj.P.Val < 0.05, na.rm = TRUE),
    lambda_inflation = calc_lambda(P.Value),
    .groups = "drop"
  )

print(lambda_df)

# P-value Histogram comparison
p_pval_hist <- ggplot(all_res, aes(x = P.Value, fill = pipeline)) +
  geom_histogram(bins = 40, color = "white", boundary = 0) +
  facet_wrap(~ pipeline, scales = "free_y") +
  scale_fill_brewer(palette = "Set2") +
  theme_minimal(base_size = 11) +
  theme(legend.position = "none") +
  labs(
    title = "Raw P-Value Distributions Across Imputation Strategies",
    subtitle = "Checking for anti-conservative spikes near 0 or severe distortion",
    x = "Raw P-Value",
    y = "Frequency Count"
  )

# QQ Plot against Uniform
all_res_qq <- all_res %>%
  group_by(pipeline) %>%
  arrange(P.Value) %>%
  mutate(
    expected = -log10(ppoints(n())),
    observed = -log10(P.Value)
  ) %>%
  ungroup()

p_qq <- ggplot(all_res_qq, aes(x = expected, y = observed, color = pipeline)) +
  geom_point(alpha = 0.5, size = 1) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "black") +
  scale_color_brewer(palette = "Set2") +
  theme_minimal(base_size = 11) +
  labs(
    title = "QQ-Plot (-log10 P-value vs Uniform Null)",
    subtitle = "Deviations above diagonal indicate significant signal / inflation",
    x = "Expected -log10(P-value)",
    y = "Observed -log10(P-value)"
  )

p_pval_comb <- p_pval_hist / p_qq + plot_layout(heights = c(1, 1))
ggsave("../figures/2_pvalue_histograms_and_qq.png", p_pval_comb, width = 10, height = 9, dpi = 300)
cat("Saved figures/2_pvalue_histograms_and_qq.png\n")


# 5. MEAN-VARIANCE TREND DIAGNOSTICS (limma plotSA) --------------------------
cat("\n[Step 5] Extracting Mean-Variance trends...\n")

get_sa_df <- function(eb_fit, pipeline_name) {
  tibble(
    Amean = eb_fit$Amean,
    sigma = eb_fit$sigma,
    s2_post = eb_fit$s2.post,
    pipeline = pipeline_name
  )
}

sa_df <- bind_rows(
  get_sa_df(p1_out$fit, "MinProb (Current)"),
  get_sa_df(p2_out$fit, "NA-Aware (No Imp)"),
  get_sa_df(p3_out$fit, "QRILC Imputation"),
  get_sa_df(p4_out$fit, "KNN Imputation")
)

p_sa <- ggplot(sa_df, aes(x = Amean, y = sigma, color = pipeline)) +
  geom_point(alpha = 0.2, size = 0.8) +
  geom_smooth(method = "loess", color = "red", se = FALSE) +
  facet_wrap(~ pipeline, scales = "free_y") +
  scale_color_brewer(palette = "Set2") +
  theme_minimal(base_size = 11) +
  theme(legend.position = "none") +
  labs(
    title = "Limma Mean-Variance Residual Trends (SA Plot)",
    subtitle = "Residual Standard Deviation vs Average Log2 Intensity",
    x = "Average Log2 Expression (Amean)",
    y = "Residual Std Dev (Sigma)"
  )

ggsave("../figures/3_mean_variance_trends.png", p_sa, width = 10, height = 7, dpi = 300)
cat("Saved figures/3_mean_variance_trends.png\n")


# 6. CONCORDANCE AND SENSITIVITY ANALYSIS -------------------------------------
cat("\n[Step 6] Analyzing concordance between MinProb and NA-aware limma...\n")

wide_res <- all_res %>%
  select(protein, pipeline, logFC, P.Value, adj.P.Val) %>%
  pivot_wider(
    names_from = pipeline,
    values_from = c(logFC, P.Value, adj.P.Val)
  )

# Merge missingness info
wide_res <- wide_res %>%
  left_join(missing_df, by = "protein")

# LogFC Correlation scatter
p_logfc_corr <- ggplot(wide_res, aes(x = `logFC_NA-Aware (No Imp)`, y = `logFC_MinProb (Current)`)) +
  geom_point(aes(color = as.factor(na_count)), alpha = 0.6, size = 1.2) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
  scale_color_viridis_d(option = "plasma", name = "NA Count") +
  theme_minimal(base_size = 11) +
  labs(
    title = "Log2 Fold Change Concordance: MinProb vs NA-Aware Limma",
    subtitle = sprintf("Spearman r = %.4f",
                        cor(wide_res$`logFC_NA-Aware (No Imp)`, wide_res$`logFC_MinProb (Current)`, use = "complete.obs", method = "spearman")),
    x = "Log2FC (NA-Aware Limma / No Imputation)",
    y = "Log2FC (MinProb Imputation)"
  )

ggsave("../figures/4_pipeline_concordance_volcano.png", p_logfc_corr, width = 8, height = 6, dpi = 300)
cat("Saved figures/4_pipeline_concordance_volcano.png\n")


# 7. MINPROB ARTIFACT INVESTIGATION -------------------------------------------
cat("\n[Step 7] Investigating potential MinProb imputation artifacts...\n")

# Proteins significant in MinProb (FDR < 0.05) but NOT in NA-aware limma
minprob_only_hits <- wide_res %>%
  filter(`adj.P.Val_MinProb (Current)` < 0.05 & (`adj.P.Val_NA-Aware (No Imp)` >= 0.05 | is.na(`adj.P.Val_NA-Aware (No Imp)`)))

robust_hits <- wide_res %>%
  filter(`adj.P.Val_MinProb (Current)` < 0.05 & `adj.P.Val_NA-Aware (No Imp)` < 0.05)

na_only_hits <- wide_res %>%
  filter(`adj.P.Val_MinProb (Current)` >= 0.05 & `adj.P.Val_NA-Aware (No Imp)` < 0.05)

cat(sprintf("Significant Hits (FDR < 0.05):\n"))
cat(sprintf(" - MinProb Current Pipeline: %d hits\n", sum(wide_res$`adj.P.Val_MinProb (Current)` < 0.05, na.rm = TRUE)))
cat(sprintf(" - NA-Aware Limma Pipeline: %d hits\n", sum(wide_res$`adj.P.Val_NA-Aware (No Imp)` < 0.05, na.rm = TRUE)))
cat(sprintf(" - QRILC Imputation Pipeline: %d hits\n", sum(wide_res$`adj.P.Val_QRILC Imputation` < 0.05, na.rm = TRUE)))
cat(sprintf(" - Robust Hits (Overlap MinProb & NA-aware): %d hits\n", nrow(robust_hits)))
cat(sprintf(" - MinProb-ONLY Hits (Potential Imputation Artifacts): %d hits\n", nrow(minprob_only_hits)))

# Missingness breakdown of MinProb-only hits vs Robust hits
artifact_summary <- tibble(
  Category = c("Robust Hits (Both)", "MinProb-Only Hits", "NA-Aware Only Hits"),
  Count = c(nrow(robust_hits), nrow(minprob_only_hits), nrow(na_only_hits)),
  Mean_NAs = c(mean(robust_hits$na_count), mean(minprob_only_hits$na_count), mean(na_only_hits$na_count)),
  Mean_LogFC_MinProb = c(mean(abs(robust_hits$`logFC_MinProb (Current)`)), mean(abs(minprob_only_hits$`logFC_MinProb (Current)`)), mean(abs(na_only_hits$`logFC_MinProb (Current)`)))
)

print(artifact_summary)

p_artifact <- ggplot(wide_res, aes(x = as.factor(na_count), fill = `adj.P.Val_MinProb (Current)` < 0.05)) +
  geom_bar(position = "fill") +
  scale_fill_manual(values = c("FALSE" = "grey70", "TRUE" = "#e41a1c"), name = "MinProb Significant (FDR < 0.05)") +
  theme_minimal(base_size = 11) +
  labs(
    title = "Proportion of Significant Hits by Number of Missing Values (NAs)",
    subtitle = "Higher NA counts show dramatically higher significance rate under MinProb",
    x = "Total Number of Missing Values (out of 10 samples)",
    y = "Proportion of Proteins"
  )

ggsave("../figures/5_minprob_artifact_analysis.png", p_artifact, width = 8, height = 5, dpi = 300)
cat("Saved figures/5_minprob_artifact_analysis.png\n")


# 8. SAVE AUDIT METRICS SUMMARY ------------------------------------------------
write_csv(lambda_df, "../data/audit_lambda_summary.csv")
write_csv(artifact_summary, "../data/audit_artifact_summary.csv")
write_csv(wide_res, "../data/audit_wide_comparison.csv")

cat("\n==================================================================\n")
cat("DE Audit Diagnostics Completed Successfully!\n")
cat("Metrics and figures saved to data/ and figures/\n")
cat("==================================================================\n")
