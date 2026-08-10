# libs --------------------------------------------------------------------

pacman::p_load(
  tidyverse,
  ape,
  GOSemSim,
  org.Hs.eg.db,
  GO.db,
  cluster,
  igraph,
  plotly,
  scales
)

setwd(this.path::here())


# data --------------------------------------------------------------------

dat_raw <- read_rds("../data/lesi_df.rds")

# gene ontology embedding -------------------------------------------------

extract_eigen_pathways <- function(lesi_df, n_eigen = 15, ont = "BP") {
  # guards
  req_cols <- c("pathway", "S_lesi", "logFC")
  if (!all(req_cols %in% colnames(lesi_df))) {
    stop("missing cols...")
  }

  lesi_clean <- lesi_df %>%
    filter(!is.na(pathway), !is.na(logFC), !is.na(S_lesi))

  # micro to macro aggregation
  pathway_metrics <- lesi_clean %>%
    group_by(pathway) %>%
    summarise(
      mean_S_lesi = mean(S_lesi),
      protein_count = n(),
      mean_abs_fc = mean(abs(logFC)),
      .groups = "drop"
    )

  target_pathways <- pathway_metrics$pathway
  M <- length(target_pathways)

  if (M <= n_eigen) {
    warning("input dimension is lower that n_eigen")
    n_eigen <- max(1, M - 1)
  }

  # information content and specificity
  sem_data <- suppressMessages(godata(
    annoDb = "org.Hs.eg.db",
    ont = ont,
    computeIC = TRUE
  ))

  pathway_metrics <- pathway_metrics %>%
    mutate(
      IC = sapply(pathway, function(id) {
        if (id %in% names(sem_data@IC)) {
          val <- sem_data@IC[[id]]
          if (is.finite(val)) {
            return(val)
          }
        }
        return(1.0)
      })
    )

  # semantic similarity tensor and distance mapping
  sim_tensor <- mgoSim(
    target_pathways,
    target_pathways,
    semData = sem_data,
    measure = "Lin",
    combine = NULL
  )

  sim_tensor[is.na(sim_tensor)] <- 0
  if (!isSymmetric(sim_tensor, tol = 1e-8)) {
    sim_tensor <- (sim_tensor + t(sim_tensor)) / 2
  }
  diag(sim_tensor) <- 1.0

  dist_matrix <- as.dist(1 - sim_tensor)

  # fitness and pam

  pathway_metrics <- pathway_metrics %>%
    mutate(
      composite_weight = mean_abs_fc * IC * mean_S_lesi
    ) %>%
    arrange(match(pathway, target_pathways))

  pam_res <- pam(dist_matrix, k = n_eigen, diss = TRUE)

  eigen_ids <- as.character(pam_res$medoids)
  eigen_ids <- unique(na.omit(eigen_ids))

  if (length(eigen_ids) == 0) {
    stop("pam did not return valid eigen-pathway")
  }

  # add semantics
  term_dict <- suppressMessages(
    AnnotationDbi::select(GO.db, keys = eigen_ids, columns = "TERM", keytype = "GOID")
  ) %>%
    distinct(GOID, .keep_all = TRUE)

  eigen_pathways <- pathway_metrics %>%
    filter(pathway %in% eigen_ids) %>%
    left_join(term_dict, by = c("pathway" = "GOID")) %>%
    rename(pathway_name = TERM) %>%
    relocate(pathway, pathway_name, composite_weight, mean_S_lesi, IC) %>%
    arrange(desc(composite_weight))

  return(list(
    eigen_pathways = eigen_pathways,
    similarity_tensor = sim_tensor,
    cluster_assignments = pam_res$clustering,
    distance_matrix = dist_matrix
  ))
}

project_semantic_manifold <- function(similarity_tensor, pathway_metadata, k = 2) {
  # guards
  if (!is.matrix(similarity_tensor)) {
    stop("provide matrix")
  }

  if (any(is.na(similarity_tensor))) {
    warning("NA in similarity_tensor")
    similarity_tensor[is.na(similarity_tensor)] <- 0
  }

  if (!isSymmetric(similarity_tensor, tol = 1e-8)) {
    similarity_tensor <- (similarity_tensor + t(similarity_tensor)) / 2
  }

  target_ids <- pathway_metadata$pathway
  if (!all(target_ids %in% rownames(similarity_tensor))) {
    stop("discontinuity detected")
  }

  sim_aligned <- similarity_tensor[target_ids, target_ids]

  # distance morphism
  dist_matrix <- as.dist(1 - sim_aligned)

  # PCoA embedding
  pcoa_res <- ape::pcoa(dist_matrix)

  # extract k coordinates
  coords <- pcoa_res$vectors[, 1:k, drop = FALSE]

  coord_names <- paste0("manifold_dim", 1:k)
  colnames(coords) <- coord_names

  # object recomposition
  coords_df <- as_tibble(coords, rownames = "pathway")

  manifold_metadata <- pathway_metadata %>%
    left_join(coords_df, by = "pathway")

  # variance explained by the manifold
  var_explained <- pcoa_res$values$Relative_eig[1:k] * 100

  for (v in seq_along(var_explained)) {
    print(paste0("Dim", v, ": ", var_explained[v]))
  }

  return(manifold_metadata)
}


## topology compression ----------------------------------------------------
eigen_results <- extract_eigen_pathways(
  lesi_df = read_rds("../data/lesi_df.rds"),
  n_eigen = 10,
  ont = "BP"
)


## euclidean embedding -----------------------------------------------------
manifold_objects <- project_semantic_manifold(
  similarity_tensor = eigen_results$similarity_tensor,
  pathway_metadata = eigen_results$eigen_pathways,
  k = 3
)


# DAG ---------------------------------------------------------------------

build_causal_vector_field <- function(manifold_df, similarity_tensor) {
  req_cols <- c("pathway", "composite_weight", "manifold_dim1", "manifold_dim2",
                "manifold_dim3")
  if (!all(req_cols %in% colnames(manifold_df))) {
    stop("missing cols")
  }

  if (!is.matrix(similarity_tensor) || !isSymmetric(similarity_tensor)) {
    stop("not symmetric")
  }

  nodes <- manifold_df %>%
    mutate(potential_u = composite_weight)

  target_ids <- nodes$pathway
  if (!all(target_ids %in% rownames(similarity_tensor))) {
    stop("discontinuity")
  }

  sim_aligned <- similarity_tensor[target_ids, target_ids]
  diag(sim_aligned) <- 0

  g_base <- graph_from_adjacency_matrix(
    sim_aligned,
    mode = "undirected",
    weighted = TRUE
  )
  V(g_base)$label <- nodes$pathway_name[match(V(g_base)$name, nodes$pathway)]
  V(g_base)$u <- nodes$potential_u[match(V(g_base)$name, nodes$pathway)]
  V(g_base)$dim1 <- nodes$manifold_dim1[match(V(g_base)$name, nodes$pathway)]
  V(g_base)$dim2 <- nodes$manifold_dim2[match(V(g_base)$name, nodes$pathway)]
  V(g_base)$dim3 <- nodes$manifold_dim3[match(V(g_base)$name, nodes$pathway)]

  edges <- as_data_frame(g_base, what = "edges") %>%
    left_join(nodes %>% select(pathway, u_from = potential_u), by = c("from" = "pathway")) %>%
    left_join(nodes %>% select(pathway, u_to = potential_u), by = c("to" = "pathway")) %>%
    mutate(
      # Biological signal strictly flows down the potential energy gradient
      causal_direction = ifelse(u_from > u_to, "forward", "reverse"),
      # Edge weight is the energetic differential scaled by ontological similarity
      gradient_weight  = abs(u_from - u_to) * weight
    )

  directed_edges <- edges %>%
    mutate(
      final_from = ifelse(causal_direction == "forward", from, to),
      final_to   = ifelse(causal_direction == "forward", to, from)
    ) %>%
    select(from = final_from, to = final_to, weight = gradient_weight) %>%
    filter(weight > quantile(weight, 0.50, na.rm = TRUE))

  causal_dag <- graph_from_data_frame(
    d          = directed_edges,
    directed   = TRUE,
    vertices   = as_data_frame(g_base, what = "vertices")
  )

  return(causal_dag)
}

plot_causal_vector_field_3d <- function(causal_graph) {

  pacman::p_load(tidyverse, igraph, plotly, scales)

  # =========================================================================
  # GATE 1: Defensive State Verification
  # =========================================================================
  if (!is_igraph(causal_graph) || !is_directed(causal_graph)) {
    stop("Type Error: Input must be a directed igraph object.")
  }

  req_attrs <- c("dim1", "dim2", "dim3", "u", "label")
  existing_attrs <- vertex_attr_names(causal_graph)
  if (!all(req_attrs %in% existing_attrs)) {
    stop("Structural Error: Graph is missing required 3D manifold attributes.")
  }

  # =========================================================================
  # MORPHISM 1: Extract 3D Vertex Coordinates
  # =========================================================================
  nodes_df <- data.frame(
    pathway = V(causal_graph)$name,
    label   = V(causal_graph)$label,
    U       = V(causal_graph)$u,
    x       = V(causal_graph)$dim1,
    y       = V(causal_graph)$dim2,
    z       = V(causal_graph)$dim3
  )

  # =========================================================================
  # MORPHISM 2: Extract and Scale Edges for Dynamic Widths and Colors
  # =========================================================================
  edges_df <- as_data_frame(causal_graph, what = "edges") %>%
    left_join(nodes_df %>% select(pathway, x_from = x, y_from = y, z_from = z), by = c("from" = "pathway")) %>%
    left_join(nodes_df %>% select(pathway, x_to = x, y_to = y, z_to = z), by = c("to" = "pathway")) %>%
    mutate(
      scaled_width = scales::rescale(weight, to = c(1.5, 7.0), from = range(weight, na.rm = TRUE))
    )

  # Generate continuous plasma color mapping for weights
  weight_palette <- scales::col_numeric(palette = viridis::plasma(100), domain = edges_df$weight)

  # =========================================================================
  # MORPHISM 3: Render Interactive 3D Vector Field via Plotly
  # =========================================================================
  p <- plot_ly()

  # Add each edge as a distinct trace with synchronized thickness and color
  for (i in seq_len(nrow(edges_df))) {
    e <- edges_df[i, ]
    edge_color <- weight_palette(e$weight)

    p <- p %>% add_trace(
      x = c(e$x_from, e$x_to),
      y = c(e$y_from, e$y_to),
      z = c(e$z_from, e$z_to),
      type = "scatter3d",
      mode = "lines",
      line = list(color = edge_color, width = e$scaled_width),
      hoverinfo = "text",
      text = paste0("<b>Causal Flow</b><br>Weight: ", round(e$weight, 3)),
      showlegend = FALSE,
      name = "Causal Flow"
    )
  }

  # Add Eigen-Pathways as 3D gravity wells
  p <- p %>% add_trace(
    data = nodes_df,
    x = ~x, y = ~y, z = ~z,
    type = "scatter3d",
    mode = "text+markers",
    text = ~paste0("<b>", label, "</b><br>Potential U: ", round(U, 3)),
    hoverinfo = "text",
    marker = list(
      size = ~scales::rescale(U, to = c(6, 20)),
      color = ~U,
      colorscale = "Plasma",
      showscale = TRUE,
      colorbar = list(title = "Potential (U)"),
      line = list(color = "black", width = 1)
    ),
    textfont = list(size = 11, color = "black"),
    textposition = "top right",
    name = "Eigen-Pathways"
  ) %>%

    layout(
      title = "3D Causal Vector Field Manifold (Harmonized)",
      scene = list(
        xaxis = list(title = "Manifold Dim 1"),
        yaxis = list(title = "Manifold Dim 2"),
        zaxis = list(title = "Manifold Dim 3"),
        camera = list(eye = list(x = 1.25, y = 1.25, z = 1.25))
      ),
      legend = list(orientation = "h", x = 0.3, y = -0.1)
    )

  return(p)
}

causal_graph <- build_causal_vector_field(
  manifold_df = manifold_objects,
  similarity_tensor = eigen_results$similarity_tensor
)

causal_vector_plot <- plot_causal_vector_field_3d(causal_graph)
causal_vector_plot


# plots -------------------------------------------------------------------

p <- manifold_objects %>%
  ggplot(aes(
    manifold_dim1, manifold_dim2
  )) +
  geom_point(
    aes(
      size = composite_weight,
      fill = mean_S_lesi
    ),
    shape = 21
  ) +
  ggrepel::geom_text_repel(
    aes(
      label = pathway_name
    ),
    size = 3.5,
    max.overlaps = Inf,
    box.padding = 0.6,
    point.padding = 0.6,
    segment.colour = "grey50",
    force = 3
  ) +
  scale_fill_viridis_c(option = "plasma") +
  ggpubr::theme_pubr() +
  theme(legend.position = "right")
p
