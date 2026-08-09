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

comp_de <- read_csv("../data/DE_results_FINAL.csv")

# target proteins ---------------------------------------------------------

high_diff <- stat_results %>%
  as_tibble() %>%
  filter(
    adj.P.Val <= 0.05,
    logFC <= -1.0
  ) %>%
  pull(ID)
clipr::write_clip(high_diff)


# pathway enrichment ------------------------------------------------------

library(tidyverse)
library(clusterProfiler)
library(org.Hs.eg.db)

run_pathway_pipeline <- function(de_results,
                                 organism_db = org.Hs.eg.db,
                                 ont = "BP",
                                 p_cutoff = 0.05,
                                 fc_cutoff = 1.0) {
  de_df <- as.data.frame(de_results)

  if (!"ID" %in% colnames(de_df)) {
    cat("--> Notice: 'ID' column not found. Extracting gene symbols from rownames.\n")
    de_df <- de_df %>% rownames_to_column(var = "ID")
  }

  ont <- toupper(ont)
  ont <- match.arg(ont, choices = c("BP", "MF", "CC", "ALL"))

  cat("--> Mapping Gene Symbols to Entrez IDs...\n")

  id_map <- tryCatch(
    {
      bitr(
        de_df$ID,
        fromType = "SYMBOL",
        toType   = "ENTREZID",
        OrgDb    = organism_db
      )
    },
    error = function(e) {
      stop("Failed to map IDs using bitr(). Ensure IDs are valid Human Gene Symbols.")
    }
  )

  annotated_de <- de_df %>%
    inner_join(id_map, by = c("ID" = "SYMBOL")) %>%
    filter(!is.na(ENTREZID) & ENTREZID != "") %>%
    distinct(ENTREZID, .keep_all = TRUE)

  cat("--> Filtering Foreground Protein Set (p <", p_cutoff, ", |logFC| >=", fc_cutoff, ")...\n")

  sig_proteins <- annotated_de %>%
    filter(adj.P.Val < p_cutoff & abs(logFC) >= fc_cutoff) %>%
    pull(ENTREZID)

  proteome_universe <- annotated_de$ENTREZID

  ora_go <- NULL

  if (length(sig_proteins) == 0) {
    warning(
      "Zero proteins passed the strict threshold (adj.P.Val < ", p_cutoff,
      " & |logFC| >= ", fc_cutoff, "). Skipping ORA.\n",
      "Consider relaxing fc_cutoff (e.g., fc_cutoff = 0.58 for 1.5-fold) or running GSEA."
    )
  } else {
    cat("--> Executing ORA for", length(sig_proteins), "significant proteins...\n")
    ora_go <- enrichGO(
      gene          = sig_proteins,
      universe      = proteome_universe,
      OrgDb         = organism_db,
      ont           = ont,
      pAdjustMethod = "BH",
      pvalueCutoff  = 0.05,
      qvalueCutoff  = 0.20
    )
  }

  cat("--> Executing GSEA across full ranked proteome...\n")

  ranked_vector <- annotated_de %>%
    mutate(rank_metric = sign(logFC) * -log10(P.Value)) %>%
    arrange(desc(rank_metric)) %>%
    {
      setNames(.$rank_metric, .$ENTREZID)
    }

  gsea_go <- gseGO(
    geneList      = ranked_vector,
    OrgDb         = organism_db,
    ont           = ont,
    pAdjustMethod = "BH",
    pvalueCutoff  = 0.05,
    verbose       = FALSE
  )

  return(list(
    ORA_GO = ora_go,
    GSEA_GO = gsea_go,
    RankedVec = ranked_vector
  ))
}

enrichment_outputs <- run_pathway_pipeline(
  de_results = stat_results,
  p_cutoff = 0.05,
  fc_cutoff = 1.0
)

view(enrichment_outputs$GSEA_GO)

p1 <- dotplot(enrichment_outputs$ORA_GO, showCategory = 15) +
  ggpubr::theme_pubr() +
  theme(
    legend.position = "right",
    axis.text.y = element_text(size = 8)
  )
p1

p2 <- dotplot(enrichment_outputs$GSEA_GO, split = ".sign") +
  ggpubr::theme_pubr() +
  theme(
    legend.position = "right",
    axis.text.y = element_text(size = 8)
  ) +
  facet_wrap(~.sign)
p2


# figures -----------------------------------------------------------------

## volcano plot
p3 <- EnhancedVolcano(
  toptable = stat_results,
  lab = stat_results$ID,
  x = "logFC",
  y = "adj.P.Val",
  pCutoff = 0.05,
  FCcutoff = 1.0,
  titleLabSize = 1,
  labSize = 3,
  drawConnectors = TRUE,
  col = c("gray", "gray", "gray", "gray")
) +
  scale_y_continuous(
    breaks = seq(-1, 5, 1),
    limits = c(-1, 5)
  ) +
  scale_x_continuous(
    breaks = seq(-5, 5, 1),
    limits = c(-5, 5)
  ) +
  ggpubr::theme_pubr() +
  theme(legend.position = "none")
p3
