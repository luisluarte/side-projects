# libs --------------------------------------------------------------------
pacman::p_load(
  tidyverse,
  Boruta,
  ranger,
  janitor,
  caret,
  pROC,
  rsample,
  logistf
)

setwd(this.path::here())

# data --------------------------------------------------------------------
dat_raw <- read_rds("../data/normalized_centered_data.rds")

dat <- dat_raw %>%
  janitor::clean_names() %>%
  group_by(row_id) %>%
  summarise(across(
    where(is.numeric), median, na.rm = TRUE
  )) %>%
  ungroup() %>%
  column_to_rownames("row_id") %>%
  t(.) %>%
  as_tibble(., rownames = "group") %>%
  mutate(
    group = if_else(str_detect(string = group, pattern = "control"),
                     "control",
                     "treatment") %>% as.factor
  )
dat


# narrow feature space Boruta ---------------------------------------------

set.seed(420)
boruta_output <- Boruta(
  group ~ .,
  data = dat,
  doTrace = 2,
  maxRuns = 200,
  getImp = Boruta::getImpFruZ
)

att_stats <- Boruta::attStats(boruta_output)

imp_data <- att_stats %>%
  rownames_to_column(var = "protein") %>%
  as_tibble() %>%
  arrange(desc(medianImp)) %>%
  mutate(protein = fct_reorder(protein, medianImp)) %>%
  filter(decision == "Confirmed")
imp_data

write_csv(x = imp_data, file = "protein_importance_v1.csv")

imp_data %>%
  slice_max(medianImp, n = 50) %>%
  ggplot(aes(
    medianImp, protein
  )) +
  geom_col()

confirmed_proteins <- names(
  boruta_output$finalDecision[boruta_output$finalDecision == "Confirmed"]
)
confirmed_proteins

model_data <- dat[, c(confirmed_proteins, "group")] %>%
  clean_names()


# predictive engine -------------------------------------------------------

# generate folds
set.seed(420)
repeated_folds <- createMultiFolds(y = dat$group, k = 5, times = 10)

list_importances <- list()
list_predictions <- list()

# feature selection
for (fold_name in names(repeated_folds)) {

  cat("\n--- Processing Subspace:", fold_name, "---\n")

  # sub-folds
  train_idx <- repeated_folds[[fold_name]]
  train_data <- dat[train_idx, ]
  test_data  <- dat[-train_idx, ]

  x_train <- train_data %>% select(-group)
  y_train <- train_data$group

  # outer loop boruta feature selection
  set.seed(420)
  boruta_fold <- suppressWarnings(
    Boruta(
      x = x_train,
      y = y_train,
      doTrace = 0,
      getImp = getImpRfGini,
      sample.fraction = 1,
      min.node.size = 1,
      pValue = 0.05,
      maxRuns = 1500,
      num.trees = 1000
    )
  )

  # get features
  boruta_resolved <- TentativeRoughFix(boruta_fold)
  sel_features <- getSelectedAttributes(boruta_resolved, withTentative = FALSE)

  # fallbacks
  if (length(sel_features) < 2) {
    cat("Warning: 0 confirmed features. Executing fallback protocol.\n")
    stats <- attStats(boruta_fold) %>% rownames_to_column("protein")

    sel_features <- stats %>%
      filter(decision %in% c("Confirmed", "Tentative")) %>%
      pull(protein)

    # If neither Confirmed nor Tentative exist, project the top 5 by median importance
    if (length(sel_features) < 2) {
      sel_features <- stats %>% top_n(5, medianImp) %>% pull(protein)
    }
  }

  # train/test for inner var imp rf loop
  train_subset <- train_data[, c(sel_features, "group")]
  test_subset  <- test_data[, c(sel_features, "group")]

  # inner LOOCV optimization
  inner_control <- trainControl(
    method = "LOOCV",
    classProbs = TRUE,
    savePredictions = "final"
  )

  max_mtry <- max(1, floor(length(sel_features) / 2))
  tune_grid <- expand.grid(
    mtry = seq(1, max_mtry, by = 1),
    splitrule = "gini",
    min.node.size = c(1, 2)
  )

  # train RF
  rf_model <- train(
    group ~ .,
    data = train_subset,
    method = "ranger",
    trControl = inner_control,
    tuneGrid = tune_grid,
    metric = "Accuracy",
    importance = "impurity"
  )

  # out of fold predictions
  fold_preds <- predict(rf_model, test_subset, type = "prob")
  list_predictions[[fold_name]] <- data.frame(
    obs = test_subset$group,
    treatment_prob = fold_preds$treatment,
    fold_id = fold_name
  )

  # get variable importance
  list_importances[[fold_name]] <- varImp(rf_model)$importance %>%
    rownames_to_column("protein") %>%
    mutate(fold_id = fold_name)
}

df_predictions <- bind_rows(list_predictions)
df_importances <- bind_rows(list_importances)

df_importances %>%
  group_by(protein) %>%
  summarise(
    imp = mean(Overall)
  )


## auc ---------------------------------------------------------------------

auc_dist <- df_predictions %>%
  mutate(
    repetition = str_extract(fold_id, "Rep\\d+")
  ) %>%
  group_by(repetition) %>%
  summarise(
    auc = as.numeric(auc(roc(obs, treatment_prob, quiet = TRUE))),
    .groups = "drop"
  )
auc_dist

global_roc <- roc(
  df_predictions$obs,
  df_predictions$treatment_prob,
  quiet = TRUE
)

global_auc <- auc(global_roc)

## calibration -----------------------------------------------------------

df_metrics <- df_predictions %>%
  mutate(
    y_true = ifelse(obs == "treatment", 1, 0),
    p_clamp = pmin(pmax(treatment_prob, 1e-15), 1 - 1e-15),
    repetition = str_extract(fold_id, "Rep\\d+")
  )

calibration_landscape <- df_metrics %>%
  group_by(repetition) %>%
  summarise(
    AUC = as.numeric(pROC::auc(pROC::roc(obs, treatment_prob, quiet = TRUE))),
      brier_score = mean((treatment_prob - y_true)^2),
      log_loss = -mean(y_true * log(p_clamp) + (1 - y_true) * log(1 - p_clamp)),
      .groups = "drop"
  )
calibration_landscape

df_calib <- df_predictions %>%
  mutate(
    bin = cut(treatment_prob, breaks = seq(0, 1, 0.1))
  ) %>%
  group_by(bin) %>%
  summarise(
    actual_rate = mean(obs == "treatment"),
    predicted_mean = mean(treatment_prob),
    count = n()
  )
df_calib

df_calib %>%
  ggplot(aes(
    predicted_mean, actual_rate
  )) +
  geom_point(aes(size = count)) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red")

# 1. Ensure domain ordering and numeric binary targets
df_predictions <- df_predictions %>%
  mutate(y_true = ifelse(obs == "treatment", 1, 0)) %>%
  arrange(treatment_prob)

# 2. Fit Firth's Penalized Logistic Regression
# This structurally resolves the infinite-coefficient problem of perfect separation
platt_model <- logistf(y_true ~ treatment_prob, data = df_predictions)

# 3. Apply the parametric morphism
df_predictions$calibrated_prob <- predict(platt_model, type = "response")

# 4. Visualize the Penalized Sigmoid Calibration
df_predictions %>%
  ggplot(aes(x = treatment_prob, y = calibrated_prob)) +
  geom_point(aes(color = as.factor(y_true)), size = 3, alpha = 0.7) +
  geom_line(color = "black") +
  scale_color_manual(values = c("0" = "#00BFC4", "1" = "#F8766D")) +
  labs(title = "Platt Calibration via Penalized Logistic Regression",
       x = "Raw Random Forest Probability",
       y = "Calibrated Probability",
       color = "Actual State") +
  theme_minimal()
