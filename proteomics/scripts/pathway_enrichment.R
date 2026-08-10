# libs --------------------------------------------------------------------

pacman::p_load(
  QFeatures,
  limma,
  EnhancedVolcano,
  clusterProfiler,
  org.Hs.eg.db,
  tidyverse,
  BiocParallel,
  furrr,
  fgsea,
  GO.db,
  AnnotationDbi
)

setwd(this.path::here())

# helper functions ---------------------------------------------------------

map_go_to_term <- function(go_ids) {
  if (!is.character(go_ids)) {
    stop("Input must be a character vector of GO IDs.")
  }

  clean_ids <- unique(na.omit(go_ids))

  if (length(clean_ids) == 0) {
    warning("No valid GO IDs provided.")
    return(tibble(Pathway_ID = character(), Pathway_Name = character(), Ontology = character()))
  }

  term_mapping <- suppressMessages(
    AnnotationDbi::select(
      x       = GO.db,
      keys    = clean_ids,
      columns = c("TERM", "ONTOLOGY"),
      keytype = "GOID"
    )
  ) %>%
    as_tibble() %>%
    rename(
      Pathway_ID   = GOID,
      Pathway_Name = TERM,
      Ontology     = ONTOLOGY
    ) %>%
    filter(!is.na(Pathway_Name))

  return(term_mapping)
}

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

compute_lesi_bulk_optimized <- function(
  de_results,
  gsea_result_obj,
  pathway_ids = NULL,
  B = 100,
  eta = 1.0,
  organism_db = org.Hs.eg.db
) {
  de_df <- as.data.frame(de_results)
  if (!"ID" %in% colnames(de_df)) de_df <- de_df %>% rownames_to_column(var = "ID")

  required_cols <- c("ID", "logFC", "P.Value")
  stopifnot(all(required_cols %in% colnames(de_df)))

  # Infer standard errors and degrees of freedom
  de_df <- de_df %>% mutate(SE = abs(logFC / t))
  df_degrees <- if ("df" %in% colnames(de_df)) de_df$df[1] else 10

  # Map IDs
  id_map <- bitr(
    de_df$ID,
    fromType = "SYMBOL",
    toType   = "ENTREZID",
    OrgDb    = organism_db
  )

  annotated_de <- de_df %>%
    inner_join(id_map, by = c("ID" = "SYMBOL")) %>%
    filter(!is.na(ENTREZID) & ENTREZID != "") %>%
    distinct(ENTREZID, .keep_all = TRUE)

  # Extract pre-built gene sets directly from the baseline gseaResult object
  all_gene_sets <- gsea_result_obj@geneSets

  if (is.null(pathway_ids)) {
    pathway_ids <- gsea_result_obj$ID
  }

  target_gene_sets <- all_gene_sets[intersect(pathway_ids, names(all_gene_sets))]

  cat(sprintf("Executing %d bootstrap runs across %d target pathways...\n", B, length(target_gene_sets)))

  # Parallelize across bootstrap noise iterations B
  boot_runs <- future_map(seq_len(B), function(b) {
    noise <- rnorm(
      n    = nrow(annotated_de),
      mean = 0,
      sd   = sqrt(eta) * annotated_de$SE
    )

    noisy_de <- annotated_de %>%
      mutate(
        logFC_noisy = logFC + noise,
        t_noisy     = logFC_noisy / SE,
        P_noisy     = 2 * (1 - pt(abs(t_noisy), df = df_degrees)),
        P_noisy     = pmax(P_noisy, 1e-15),
        rank_metric = sign(logFC_noisy) * -log10(P_noisy)
      )

    noisy_ranks <- noisy_de %>%
      arrange(desc(rank_metric)) %>%
      {
        setNames(.$rank_metric, .$ENTREZID)
      }

    # Fast C++ GSEA on ALL pathways simultaneously
    fgsea_res <- fgsea::fgsea(
      pathways = target_gene_sets,
      stats    = noisy_ranks,
      minSize  = 1,
      maxSize  = Inf,
      eps      = 0
    )

    fgsea_res %>%
      as_tibble() %>%
      select(pathway, leadingEdge)
  }, .options = furrr_options(seed = TRUE), .progress = TRUE)

  # Unnest and aggregate leading-edge frequencies
  boot_df <- bind_rows(boot_runs) %>%
    filter(lengths(leadingEdge) > 0) %>%
    unnest(leadingEdge) %>%
    rename(ENTREZID = leadingEdge)

  # Compute S_lesi metrics for each pathway
  lesi_summaries <- boot_df %>%
    group_by(pathway, ENTREZID) %>%
    summarise(
      core_hits = n(),
      S_lesi    = core_hits / B,
      .groups   = "drop"
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
    select(pathway, SYMBOL, ENTREZID, S_lesi, stability_tier, logFC, SE, P.Value) %>%
    arrange(pathway, desc(S_lesi), P.Value) %>%
    group_split(pathway)

  names(lesi_summaries) <- map_chr(lesi_summaries, ~ .x$pathway[1])
  return(lesi_summaries)
}


plan(multisession, workers = availableCores() - 1)
lesi_results <- compute_lesi_bulk_optimized(
  de_results = stat_results,
  gsea_result_obj = enrichment_outputs$GSEA_GO,
  pathway_ids = enrichment_outputs$GSEA_GO$ID,
  B = 1000,
  eta = 1.0
)
plan(sequential)
write_rds(x = lesi_results, file = "../data/lesi_results.rds")


lesi_comp <- lesi_results %>%
  map_dfr(., function(D) {
    as_tibble(D)
  })
target_pathways <- unique(lesi_comp$pathway)
pathway_dictionary <- map_go_to_term(target_pathways)

lesi_df <- lesi_comp %>%
  left_join(pathway_dictionary, by = c("pathway" = "Pathway_ID")) %>%
  relocate(pathway, Pathway_Name, SYMBOL, S_lesi, stability_tier)
write_rds(x = lesi_df, file = "../data/lesi_df.rds")


