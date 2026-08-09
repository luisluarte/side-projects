# libs --------------------------------------------------------------------

pacman::p_load(
  QFeatures,
  limma,
  EnhancedVolcano,
  clusterProfiler,
  org.Hs.eg.db,
  tidyverse
)

# data --------------------------------------------------------------------

setwd(this.path::here())

raw_dat <- read_tsv("../data/combined_protein.tsv")

intensity_cols <- colnames(raw_dat) %>%
  str_subset(pattern = "(Control|Tratamiento)_[0-9]+ Intensity")

ctrl_target_cols <- intensity_cols[str_detect(intensity_cols, "Control")]
trat_target_cols <- intensity_cols[str_detect(intensity_cols, "Tratamiento")]


# set 0 read as NA
dat <- raw_dat %>%
  mutate(across(
    all_of(intensity_cols), ~ na_if(., 0)
  ))

# keep only that are detected in at least 3 replicates
valid_proteins <- dat %>%
  rowwise() %>%
  mutate(
    valid_ctrl = sum(!is.na(c_across(all_of(ctrl_target_cols)))),
    valid_trat = sum(!is.na(c_across(all_of(trat_target_cols)))),
    valid_total = valid_ctrl + valid_trat
  ) %>%
  ungroup() %>%
  filter(
    valid_total >= 8 & (valid_ctrl >= 4 | valid_trat >= 4)
  )
valid_proteins

# change in detected proteins
nrow(dat) - nrow(valid_proteins)


# QFeatures ---------------------------------------------------------------

metadata <- data.frame(
  row.names = intensity_cols,
  sample_id = intensity_cols,
  condition = factor(
    if_else(str_detect(intensity_cols, "Control"), "Control", "Treatment"),
    levels = c("Control", "Treatment")
  )
)
metadata

valid_df <- as.data.frame(valid_proteins)

# data container
fts <- readQFeatures(
  assayData = valid_df,
  quantCols = intensity_cols,
  name = "raw_maxlfq"
)
fts

colData(fts) <- DataFrame(metadata)


# data transformation -----------------------------------------------------

# log2 transform
fts <- logTransform(fts, i = "raw_maxlfq", name = "log2maxlfq")

# median centering
fts <- normalize(fts,
  i = "log2maxlfq",
  name = "normalized_maxlfq",
  method = "center.median"
)

# imputation based on minimum sample value (assuming LOD)

fts <- impute(
  fts,
  i = "normalized_maxlfq",
  name = "imputed_maxlfq",
  method = "MinProb"
)


# Identity mapping --------------------------------------------------------

exprs_matrix <- assay(fts, "imputed_maxlfq")

row_metadata <- rowData(fts[["imputed_maxlfq"]])

# guard
resolved_labels <- if_else(
  is.na(row_metadata$Gene) | row_metadata$Gene == "",
  row_metadata$Protein.ID,
  row_metadata$Gene
)

rownames(exprs_matrix) <- resolved_labels
nrow(exprs_matrix)

data_out <- as_tibble(exprs_matrix, rownames = "row_id")
write_rds(x = data_out, file = "../data/normalized_centered_data.rds")


# lineal models -----------------------------------------------------------

write_rds(x = fts, file = "../data/fts.rds")
design_matrix <- model.matrix(~ 0 + condition, data = colData(fts))


fit <- lmFit(exprs_matrix, design_matrix)

contrast.matrix <- makeContrasts(
  Diff = conditionTreatment - conditionControl,
  levels = design_matrix
)

fit_contrast <- contrasts.fit(fit, contrast.matrix)

eb_fit_contrast <- eBayes(fit_contrast, trend = TRUE, robust = TRUE)

# empirial bayes shrinkage to moderate standard error
# across parallel feature vectors
eb_fit <- eBayes(fit, trend = TRUE, robust = TRUE)

stat_results <- topTable(
  eb_fit_contrast,
  coef = "Diff",
  adjust.method = "BH",
  number = Inf
)

head(stat_results)

write_csv(x = stat_results, file = "../data/DE_analysis.csv")
