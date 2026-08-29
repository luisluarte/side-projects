library(Rcpp)
library(dplyr)
library(lme4)

cat("Loading Dataset...\n")
dataset_path <- "C:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv"
dat_all <- read.csv(dataset_path)

set.seed(42)
# We will use 20 participants for speed
participants <- sample(unique(dat_all[['participant_id']]), 20)

# Compile C++ scripts
sourceCpp("src/fitting_procedures/mcmc_sampler.cpp")
sourceCpp("src/models/extract_layer_metrics.cpp")

init_phi_baseline <- c(log(2.0), log(0.3/0.7), log(3.0), log(0.5/0.5), log(0.5/0.5), 0.5, log(1.0), log(0.5))

all_metrics <- list()

cat("Extracting optimized layer metrics per participant...\n")
for (p in participants) {
  p_data <- dat_all[dat_all$participant_id == p, ]
  p_data <- p_data[order(p_data$ttp), ]
  p_data$RT <- (as.numeric(p_data$ttr) - as.numeric(p_data$ttp)) / 1000.0
  p_data$delta_t <- c(NA, diff(as.numeric(p_data$ttp)) / 1000.0) 
  p_data$delta_t[is.na(p_data$delta_t)] <- 2.0
  
  valid_idx <- which(!is.na(p_data$RT) & p_data$RT >= 0.1 & p_data$RT <= 3.0 & p_data$Resp %in% c(1, 2))
  p_data <- p_data[valid_idx, ]
  
  if (nrow(p_data) < 20) next
  
  # Run very short MCMC to get near MAP estimates (for speed)
  chain <- run_mcmc_subject(6, 15, init_phi_baseline, p_data$Resp, p_data$F, p_data$RT, p_data$delta_t)
  map_phi <- as.numeric(chain[15, ])
  
  # Extract metrics
  metrics_df <- extract_layer_metrics_cpp(map_phi, p_data$Resp, p_data$F, p_data$delta_t)
  
  metrics_df$participant <- p
  metrics_df$trial <- 1:nrow(metrics_df)
  metrics_df$is_switch <- c(0, ifelse(p_data$Resp[-1] != p_data$Resp[-nrow(p_data)], 1, 0))
  
  all_metrics[[length(all_metrics) + 1]] <- metrics_df
}

full_df <- bind_rows(all_metrics)

cat("Computing Deltas ([pre-switch, switch])...\n")
# Compute delta(t) = metric(t) - metric(t-1)
full_df <- full_df %>%
  group_by(participant) %>%
  mutate(
    d_MF_Ent = MF_Ent - lag(MF_Ent),
    d_MF_Spa = MF_Spa - lag(MF_Spa),
    d_MF_L2 = MF_L2 - lag(MF_L2),
    d_GC_Ent = GC_Ent - lag(GC_Ent),
    d_GC_Spa = GC_Spa - lag(GC_Spa),
    d_GC_L2 = GC_L2 - lag(GC_L2),
    d_MLI_Ent = MLI_Ent - lag(MLI_Ent),
    d_MLI_Spa = MLI_Spa - lag(MLI_Spa),
    d_MLI_L2 = MLI_L2 - lag(MLI_L2)
  ) %>%
  filter(!is.na(d_MF_Ent)) %>%
  ungroup()

# Standardize variables for fair comparison of coefficients
scale_this <- function(x) as.numeric(scale(x))

full_df_scaled <- full_df %>%
  mutate(across(starts_with("d_"), scale_this))

metrics <- c("d_MF_Ent", "d_MF_Spa", "d_MF_L2",
             "d_GC_Ent", "d_GC_Spa", "d_GC_L2",
             "d_MLI_Ent", "d_MLI_Spa", "d_MLI_L2")

results <- data.frame()

cat("Fitting Logistic Regressions...\n")
for (m in metrics) {
  form <- as.formula(paste("is_switch ~", m, "+ (1 | participant)"))
  model <- glmer(form, data = full_df_scaled, family = binomial, control=glmerControl(optimizer="bobyqa"))
  
  s <- summary(model)
  coef <- s$coefficients[2, 1]
  pval <- s$coefficients[2, 4]
  aic <- AIC(model)
  
  results <- rbind(results, data.frame(
    Metric = m,
    Coefficient = coef,
    P_Value = pval,
    AIC = aic
  ))
}

results <- results[order(results$AIC), ]

report <- c(
  "# Layer Information Theory Metrics: Predictive Power on Switches",
  "",
  "We ran a brute-force analysis extracting the Shannon Entropy, Hoyer's Sparsity, and L2 Norm (Energy) for every physical layer in the Cerebellar network (Mossy Fibers, Granule Cells, Molecular Layer Interneurons) using the optimized Cortical RPE model for each participant.",
  "",
  "We then computed the Delta ($\\Delta = t_{switch} - t_{pre-switch}$) and ran mixed-effects logistic regression models to determine which information metric held the most predictive power for an incoming switch.",
  "",
  "## Results (Ranked by Predictive Power / AIC)",
  "| Layer Metric | Coefficient (Scaled) | P-Value | AIC |",
  "| :--- | :--- | :--- | :--- |"
)

for (i in 1:nrow(results)) {
  report <- c(report, sprintf("| **%s** | %.4f | %.2e | %.1f |", 
                              results$Metric[i], results$Coefficient[i], results$P_Value[i], results$AIC[i]))
}

writeLines(report, "docs/Layer_Metrics_Analysis.md")
cat("\nAnalysis complete! See docs/Layer_Metrics_Analysis.md\n")
