# libs --------------------------------------------------------------------

pacman::p_load(
  QFeatures,
  limma,
  EnhancedVolcano,
  clusterProfiler,
  org.Hs.eg.db,
  tidyverse,
  BiocParallel,
  furrr
)

setwd(this.path::here())

# pathway enrichment ------------------------------------------------------

run_pathway_pipeline <- function(de_results,
                                 organism_db = org.Hs.eg.db,
                                 ont = "BP",
                                 p_cutoff = 0.05,
                                 fc_cutoff = 1.0) {
  # everything uses data.frame as default do not go with tibbles
  de_df <- as.data.frame(de_results)

  # just in case
  if (!"ID" %in% colnames(de_df)) {
    de_df <- de_df %>% rownames_to_column(var = "ID")
  }

  ont <- toupper(ont)
  ont <- match.arg(ont, choices = c("BP", "MF", "CC", "ALL"))

  # map to entrez
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
      stop("error with gene mapping to entrez")
    }
  )

  annotated_de <- de_df %>%
    inner_join(id_map, by = c("ID" = "SYMBOL")) %>%
    filter(!is.na(ENTREZID) & ENTREZID != "") %>%
    distinct(ENTREZID, .keep_all = TRUE)

  # protein filter
  sig_proteins <- annotated_de %>%
    filter(adj.P.Val < p_cutoff & abs(logFC) >= fc_cutoff) %>%
    pull(ENTREZID)

  proteome_universe <- annotated_de$ENTREZID

  ora_go <- NULL

  if (length(sig_proteins) == 0) {
    warning(
      "no proteins passed the strict threshold (adj.P.Val < ", p_cutoff,
      " & |logFC| >= ", fc_cutoff, "). skipping ORA.\n"
    )
  } else {
    cat("executing ORA for", length(sig_proteins), "significant proteins \n")
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

  cat("Executing GSEA")

  ranked_vector <- annotated_de %>%
    mutate(rank_metric = sign(logFC) * -log10(P.Value)) %>% # standard metric
    arrange(desc(rank_metric)) %>%
    {
      setNames(.$rank_metric, .$ENTREZID)
    }

  gsea_go <- gseGO(
    geneList = ranked_vector,
    OrgDb = organism_db,
    ont = ont,
    pAdjustMethod = "BH",
    pvalueCutoff = 0.05,
    eps = 0,
    BPPARAM = SerialParam(),
    verbose = FALSE
  )

  return(list(
    ORA_GO = ora_go,
    GSEA_GO = gsea_go,
    RankedVec = ranked_vector,
    Annotated = annotated_de
  ))
}

stat_results <- read_csv("../data/DE_analysis.csv")

enrichment_outputs <- run_pathway_pipeline(
  de_results = stat_results,
  p_cutoff = 0.05,
  fc_cutoff = 1.0
)

# robust analysis ---------------------------------------------------------

# leading-edge stability index S_{lesi}
# given a biological pathway S and protein p \in S
# S_{lesi} is the empirical selection frequency of p across
# B boostrap or noise-injection iterations
# S_{\text{lesi}}(p) = \frac{1}{B} \sum_{b=1}^{B} \mathbb{I}\left(p \in \mathcal{LE}_b(S)\right)
# \mathcal{LE}_b(S) is the subset of proteins positioned prior to the maximum peak
# running enrichment score (ES) for pathway S in iteration b
# \mathbb{I}(\cdot) is the indicator function, 1 if p \in \mathcal{LE}_b(S) and 0 otherwise
# B is the total number of resampled bootstrap runs (~1000)

compute_lesi_noise_injection <- function(
  de_results,
  pathway_id,
  B = 100,
  eta = 1.0,
  organism_db = org.Hs.eg.db
) {
  de_df <- as.data.frame(de_results)
  if (!"ID" %in% colnames(de_df)) de_df <- de_df %>% rownames_to_column(var = "ID")

  required_cols <- c("ID", "logFC", "P.Value")
  stopifnot(all(required_cols %in% colnames(de_df)))

  # infer standard errors
  de_df <- de_df %>%
    mutate(SE = abs(logFC / t))

  df_degrees <- if ("df" %in% colnames(de_df)) de_df$df[1] else 10

  id_map <- bitr(de_df$ID,
    fromType = "SYMBOL",
    toType = "ENTREZID",
    OrgDb = organism_db
  )

  annotated_de <- de_df %>%
    inner_join(id_map, by = c("ID" = "SYMBOL")) %>%
    filter(!is.na(ENTREZID) & ENTREZID != "") %>%
    distinct(ENTREZID, .keep_all = TRUE)

  # base GSEA
  base_ranks <- annotated_de %>%
    mutate(
      rank_metric = sign(logFC) * -log10(P.Value)
    ) %>%
    arrange(desc(rank_metric)) %>%
    {
      setNames(.$rank_metric, .$ENTREZID)
    }

  base_gsea <- suppressWarnings(
    gseGO(
      geneList = base_ranks,
      OrgDb = organism_db,
      ont = "BP",
      eps = 0,
      verbose = TRUE
    )
  )

  base_df <- as.data.frame(base_gsea)
  target_row <- base_df %>%
    filter(
      ID == pathway_id | Description == pathway_id
    ) %>%
    slice(1)

  if (nrow(target_row) == 0) stop("target pathway not found in GSEA outputs")

  target_go_id <- target_row$ID
  pathway_genes <- unlist(strsplit(target_row$core_enrichment, "/"))

  le_accumulator <- vector("list", B)

  for (b in seq_len(B)) {
    prg <- length(B) / b
    print(paste("bootstrap progress:", prg))
    noise <- rnorm(
      n = nrow(annotated_de),
      mean = 0,
      sd = sqrt(eta) * annotated_de$SE
    )

    noisy_de <- annotated_de %>%
      mutate(
        logFC_noisy = logFC + noise,
        t_noisy = logFC_noisy / SE,
        P_noisy = 2 * (1 - pt(abs(t_noisy), df = df_degrees)),
        P_noise = pmax(P_noisy, 1e-15),
        rank_metric = sign(logFC_noisy) * -log10(P_noisy)
      )

    noisy_ranks <- noisy_de %>%
      arrange(desc(rank_metric)) %>%
      { setNames(.$rank_metric, .$ENTREZID) }

    noisy_gsea <- suppressWarnings(
      gseGO(
        geneList = noisy_ranks,
        OrgDb = organism_db,
        ont = "BP",
        eps = 0,
        verbose = FALSE
      )
    )

    noisy_df <- as.data.frame(noisy_gsea)
    hit_row <- noisy_df %>% filter(ID == target_go_id)

    if (nrow(hit_row) > 0) {
      le_accumulator[[b]] <- unlist(strsplit(hit_row$core_enrichment, "/"))
    }
  }

  all_le_genes <- unlist(le_accumulator)

  lesi_summary <- tibble(ENTREZID = all_le_genes) %>%
    group_by(ENTREZID) %>%
    summarise(
      core_hits = n(),
      S_lesi = core_hits / B,
      .groups = "drop"
    ) %>%
    inner_join(id_map, by = "ENTREZID") %>%
    inner_join(annotated_de %>% select(ENTREZID, logFC, P.Value, SE), by = "ENTREZID") %>%
    mutate(
      stability_tier = case_when(
        S_lesi >= 0.80 ~ "core_driver",
        S_lesi >= 0.50 ~ "context_dependent",
        TRUE ~ "border_rider"
      )
    ) %>%
    select(SYMBOL, ENTREZID, S_lesi, stability_tier, logFC, SE, P.Value) %>%
    arrange(desc(S_lesi), P.Value)

  return(lesi_summary)
}

pathway_list <- enrichment_outputs$GSEA_GO$ID

plan(multisession, workers = 2)
lesi_results <- pathway_list %>%
  future_map(., function(pathway) {
    compute_lesi_noise_injection(
      de_results = stat_results,
      pathway_id = pathway,
      B = 100,
      eta = 1.0
    )
  }, .options = furrr_options(seed = TRUE), .progress = TRUE)
plan(sequential)
write_rds(x = lesi_results, file = "../data/lesi_results.rds")


# figures -----------------------------------------------------------------

p1 <- dotplot(enrichment_outputs$GSEA_GO, split = ".sign") +
  ggpubr::theme_pubr() +
  theme(
    legend.position = "right",
    axis.text.y = element_text(size = 8)
  ) +
  facet_wrap(~.sign)
p1

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
