# ==============================================================================
# Proteomics Visualization Orchestrator (Antigravity)
# Interactive Visualization Space: Minimalist FTP-Style Pathway Explorer
# ==============================================================================

# 0. Dependencies -------------------------------------------------------------
pacman::p_load(
  shiny,
  shinyTree,
  plotly,
  tidyverse,
  GO.db,
  AnnotationDbi
)

# Set working directory relative to file location if available
if (requireNamespace("this.path", quietly = TRUE)) {
  try(setwd(this.path::here()), silent = TRUE)
}

# 1. Gate 1 Integrity Check & Data Loader Functor ------------------------------
#' Verify and load O_lesi results object with dynamic annotation fallback
#' @param file_path Candidate RDS file name
#' @return Validated tibble containing SYMBOL, logFC, SE, S_lesi, pathway, Pathway_Name
load_lesi_data <- function(file_path = "lesi_results.rds") {
  candidate_paths <- c(
    file_path,
    file.path("data", file_path),
    file.path("..", "data", file_path),
    "lesi_df.rds",
    file.path("data", "lesi_df.rds"),
    file.path("..", "data", "lesi_df.rds")
  )
  
  actual_path <- candidate_paths[file.exists(candidate_paths)][1]
  if (is.na(actual_path) || !file.exists(actual_path)) {
    stop("Gate 1 Error: Could not locate lesi dataset object (O_lesi).")
  }
  
  res <- read_rds(actual_path)
  
  # Coerce payload list/vctrs structure to a tidy tibble
  df <- if (is.list(res) && !is.data.frame(res)) {
    bind_rows(unname(res))
  } else {
    as_tibble(res)
  }
  
  # Gate 1: Column Schema Verification
  req_cols <- c("SYMBOL", "logFC", "SE", "S_lesi", "pathway")
  missing_cols <- setdiff(req_cols, colnames(df))
  if (length(missing_cols) > 0) {
    stop(paste("Gate 1 Error: Missing required columns in O_lesi:", paste(missing_cols, collapse = ", ")))
  }
  
  # Gate 1 Dynamic Fallback: Generate Pathway_Name using GO.db if missing or unpopulated
  if (!"Pathway_Name" %in% colnames(df) || any(is.na(df$Pathway_Name))) {
    go_ids <- unique(df$pathway[!is.na(df$pathway)])
    go_terms <- suppressMessages(
      AnnotationDbi::select(GO.db, keys = go_ids, columns = "TERM", keytype = "GOID")
    ) %>%
      distinct(GOID, .keep_all = TRUE)
    
    df <- df %>%
      left_join(go_terms, by = c("pathway" = "GOID")) %>%
      mutate(Pathway_Name = ifelse(is.na(TERM), pathway, TERM)) %>%
      dplyr::select(-any_of("TERM"))
  }
  
  return(df)
}

# 2. Morphism 1: Topological DAG-to-Tree Projection ---------------------------
#' Map Gene Ontology DAG pathways to a strict 1:1 hierarchy for shinyTree
#' @param lesi_df Validated tibble of S_lesi results
#' @return A nested list structure suitable for shinyTree serialization
build_ftp_tree <- function(lesi_df) {
  # 1. Extract unique pathways
  pathways <- lesi_df %>%
    distinct(pathway, Pathway_Name) %>%
    filter(!is.na(pathway), !is.na(Pathway_Name))

  # 2. Extract topological edge lookup environments
  go_parents_env <- c(as.list(GOBPPARENTS), as.list(GOCCPARENTS), as.list(GOMFPARENTS))

  # 3. Projection Morphism: Force DAG to strict tree by selecting primary slice(1) parent
  parents_map <- pathways %>%
    mutate(PARENTS = sapply(pathway, function(id) {
      p <- go_parents_env[[id]]
      p <- p[!is.na(p) & p != "all"]
      if (length(p) > 0) p[1] else NA_character_
    }))

  # 4. Map parent GO IDs to human-readable terms via GO.db
  valid_parents <- unique(na.omit(parents_map$PARENTS))
  
  parent_names <- if (length(valid_parents) > 0) {
    suppressMessages(
      AnnotationDbi::select(GO.db, keys = valid_parents, columns = "TERM", keytype = "GOID")
    ) %>%
      distinct(GOID, .keep_all = TRUE)
  } else {
    tibble(GOID = character(0), TERM = character(0))
  }

  # 5. Construct hierarchical directory table
  ftp_structure <- pathways %>%
    left_join(parents_map %>% dplyr::select(pathway, PARENTS), by = "pathway") %>%
    left_join(parent_names, by = c("PARENTS" = "GOID")) %>%
    mutate(Parent_Folder = ifelse(is.na(TERM) | TERM == "", "Root_Processes", TERM))

  # 6. Convert to nested list for shinyTree serialization
  # Leaf nodes are explicitly typed with stselected = FALSE & go_id metadata
  tree_list <- list()
  for (folder in unique(ftp_structure$Parent_Folder)) {
    children <- ftp_structure %>% filter(Parent_Folder == folder)

    child_list <- list()
    for (i in seq_len(nrow(children))) {
      child_list[[children$Pathway_Name[i]]] <- structure("", stselected = FALSE, go_id = children$pathway[i])
    }
    tree_list[[folder]] <- child_list
  }

  return(tree_list)
}

# 3. UI Definition: Minimalist FTP Terminal Aesthetic --------------------------
ui <- fluidPage(
  tags$head(
    tags$style(HTML("
      body { font-family: 'Courier New', Courier, monospace; background-color: #FAFAFA; color: #333; }
      h3, h4 { font-weight: bold; border-bottom: 1px solid #CCC; padding-bottom: 5px; }
      .well { background-color: #FFF; border: 1px solid #EEE; box-shadow: none; border-radius: 0; }
      .jstree-default .jstree-anchor { font-family: 'Courier New', Courier, monospace; }
      .plotly { margin-top: 20px; }
    "))
  ),
  titlePanel("Index of /pub/proteomics/pathways/"),
  sidebarLayout(
    sidebarPanel(
      width = 4,
      h4("Directories"),
      shinyTree("ftp_tree", theme = "default", types = "{ 'default': { 'icon': 'fa fa-folder' } }")
    ),
    mainPanel(
      width = 8,
      h4(textOutput("current_path")),
      plotlyOutput("pathway_plot", height = "700px")
    )
  )
)

# 4. Server Logic: Reactive Functors & Gate Verification ----------------------
server <- function(input, output, session) {
  # Gate 1 & Reactive State Loader
  lesi_data <- reactive({
    load_lesi_data("lesi_results.rds")
  })

  # Morphism 1: Tree Projection Computation
  tree_data <- reactive({
    build_ftp_tree(lesi_data())
  })

  output$ftp_tree <- renderTree({
    tree_data()
  })

  # Morphism 2 & 3: JS-to-R Flattening Functor & Type-Safety Verification Gate
  selected_pathway <- reactive({
    tree <- input$ftp_tree
    req(tree)

    # Morphism 2: JS-to-R Flattening Functor
    selected <- get_selected(tree, format = "classid")

    # Gate 2: Leaf-Node Nullification
    if (length(selected) == 0) return(NULL)

    # Extract target string identifier robustly from list element or names attribute
    raw_id <- unlist(selected)[1]
    target_name <- if (!is.null(raw_id) && !is.na(raw_id) && nchar(as.character(raw_id)) > 0) {
      as.character(raw_id)
    } else if (!is.null(names(selected)) && nchar(names(selected)[1]) > 0) {
      names(selected)[1]
    } else {
      NULL
    }

    if (is.null(target_name)) return(NULL)

    # Morphism 3: Domain-Constrained Verification
    valid_pathways <- unique(lesi_data()$Pathway_Name)

    if (target_name %in% valid_pathways) {
      return(target_name)
    } else {
      # Gracefully halt downstream reactive chain if a folder or invalid node is clicked
      return(NULL)
    }
  })

  # Gate 2 Display Protocol: Quietly output status without red Shiny errors
  output$current_path <- renderText({
    target <- selected_pathway()
    if (is.null(target)) {
      return("Awaiting selection...")
    }
    paste0("Viewing: ./", target)
  })

  # Morphism 4 & Gate 3: Plotly Projection & Rendering Thresholds
  output$pathway_plot <- renderPlotly({
    target_name <- selected_pathway()
    req(target_name) # Gate 2: ensure target_name is a non-null valid Pathway_Name

    plot_data <- lesi_data() %>%
      filter(Pathway_Name == target_name) %>%
      arrange(desc(logFC))

    # Gate 3: Canvas Rendering Threshold (Abort if misfire or empty subset)
    req(nrow(plot_data) > 0)

    # Morphism 4: Plotly Projection Canvas Construction
    p <- plot_data %>%
      ggplot(aes(
        x = reorder(SYMBOL, logFC),
        y = logFC,
        fill = S_lesi,
        text = sprintf("Protein: %s\nlog2FC: %.2f ± %.2f\nS_lesi: %.2f", SYMBOL, logFC, SE, S_lesi)
      )) +
      geom_hline(yintercept = 0, color = "black", linewidth = 0.5) +
      geom_hline(yintercept = 1.0, linetype = "dashed", color = "gray50") +
      geom_hline(yintercept = -1.0, linetype = "dashed", color = "gray50") +
      geom_col(color = "black", width = 0.7) +
      geom_errorbar(aes(ymin = logFC - SE, ymax = logFC + SE), width = 0.2, color = "black") +
      scale_fill_viridis_c(option = "plasma", name = "S_lesi", limits = c(0, 1)) +
      theme_classic() +
      theme(
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1, size = 10, family = "mono"),
        axis.text.y = element_text(size = 10, family = "mono"),
        axis.title = element_text(face = "bold", family = "mono"),
        plot.title = element_blank()
      ) +
      labs(x = "Pathway Member Proteins", y = "Log2 Fold Change")

    # Gate 3: Canvas Margin Protection (b = 100 prevents label clipping)
    ggplotly(p, tooltip = "text") %>%
      layout(
        hovermode = "closest",
        margin = list(b = 100)
      ) %>%
      config(displayModeBar = TRUE, scrollZoom = TRUE)
  })
}

# 5. Application Launch --------------------------------------------------------
shinyApp(ui = ui, server = server)
