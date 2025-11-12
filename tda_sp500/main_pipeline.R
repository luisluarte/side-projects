# === MAIN EXECUTION PIPELINE ===

# 1. Load Libraries
pacman::p_load(
  tidyverse,
  checkmate,
  purrr,
  nnet,
  cryptoQuotes,
  lubridate,
  zoo,
  this.path # For sourcing
)

# 2. Source All HMM Morphisms
# setwd(this.path::here()) # Set working directory if needed
source("hmm_funcs.R")

# 3. Define Static Model Parameters
STATES <- c("bull", "bear", "sideways")
ASSET_NAMES <- c("BTC", "USD") # Using BTC instead of SPY

# The allocation strategy (The bot's "playbook")
REGIME_ALLOCATIONS <- list(
  "bull" = c(BTC = 1.0, USD = 0.0), # 100% Bitcoin
  "bear" = c(BTC = 0.0, USD = 1.0), # 100% Cash
  "sideways" = c(BTC = 0.5, USD = 0.5) # 50/50
)

# === 4. New Morphisms (Data Preparation) ===

#' @title Covariate Generation Morphism
#' @description Creates the Z_t covariate matrix from the raw price data.
#' @param log_returns_vec The vector of log-returns (T-1 elements).
#' @return A data.frame of covariates, aligned with the log-returns.
generate_covariates <- function(log_returns_vec, window = 20) {
  checkmate::assert_numeric(log_returns_vec)

  # Calculate 20-day rolling volatility (std dev) of log-returns
  # 'na.pad = TRUE' ensures the output vector has the same length as the input
  rolling_vol <- zoo::rollapply(
    log_returns_vec,
    width = window,
    FUN = sd,
    fill = NA,
    align = "right"
  )

  # Create the final dataframe
  covariates_df <- data.frame(volatility = rolling_vol)

  # Ensure output is a data.frame
  checkmate::assert_data_frame(covariates_df, nrows = length(log_returns_vec))
  covariates_df
}

#' @title Data Alignment Morphism
#' @description Aligns observations and covariates by removing initial NA
#'              rows caused by rolling window calculations.
#' @param obs_vec The T-1 vector of log-returns.
#' @param cov_df The T-1 data.frame of covariates.
#' @return A list containing the aligned `observations_vec` and `covariates_df`.
align_data <- function(obs_vec, cov_df) {
  # Find the first row where all covariates are complete (non-NA)
  first_valid_row <- zoo::na.trim(cov_df, is.na = "any", sides = "left")
  first_valid_index <- as.numeric(rownames(first_valid_row)[1])

  cat(paste("Data aligned. Trimming first", first_valid_index - 1, "rows for NA warmup.\n"))

  # Trim both datasets to start from that index
  aligned_obs <- obs_vec[first_valid_index:length(obs_vec)]
  aligned_cov <- cov_df[first_valid_index:nrow(cov_df), , drop = FALSE]

  list(
    observations_vec = aligned_obs,
    covariates_df = aligned_cov
  )
}


# === 5. The Main Pipeline Execution ===
cat("--- 1. LOADING DATA ---\n")
raw_data <- get_bitcoin_price_series(
  ticker = "BTCUSDT",
  source = "binance",
  start_date = "2018-01-01",
  end_date = "2025-11-01", # Use historical data for training
  interval = "1d"
)

# Convert to data.frame for easier handling
raw_data_df <- data.frame(
  Date = zoo::index(raw_data),
  Close = raw_data$close
)

cat("--- 2. PREPARING DATA & COVARIATES ---\n")
observations_vec <- calculate_log_returns(raw_data_df$close)
covariates_df <- generate_covariates(observations_vec, window = 20)

# Align the data (removes initial NAs from rolling window)
aligned_data <- align_data(observations_vec, covariates_df)

cat("--- 3. GENERATING INITIAL PARAMETERS ---\n")
initial_model_params <- generate_initial_parameters(
  aligned_data$observations_vec,
  aligned_data$covariates_df,
  STATES
)

cat("--- 4. RUNNING BAUM-WELCH TRAINING ---\n")
# This is the main training step. It may take several minutes.
trained_model <- tryCatch(
  {
    run_baum_welch_training(
      observations_vec = aligned_data$observations_vec,
      covariates_df = aligned_data$covariates_df,
      state_names = STATES,
      initial_params = initial_model_params$initial_params,
      initial_emission_params = initial_model_params$initial_emission_params,
      initial_beta_params = initial_model_params$initial_beta_params,
      max_iterations = 100,
      tolerance = 1e-6
    )
  },
  error = function(e) {
    cat("ERROR during model training:", e$message, "\n")
    NULL
  }
)

if (!is.null(trained_model)) {
  cat("\n--- 5. TRAINING COMPLETE ---\n")
  print("Final Log-Likelihood:")
  print(tail(trained_model$log_likelihood_history, 1))

  # === 6. RUN ALLOCATION MORPHISM (Example) ===
  # Get the *last* set of regime probabilities from the training data

  # Run the forward pass *one more time* using the OPTIMIZED parameters
  final_filter <- run_forward_pass(
    observations_vec = aligned_data$observations_vec,
    covariates_df = aligned_data$covariates_df,
    state_names = STATES,
    initial_params = initial_model_params$initial_params,
    emission_params = trained_model$optimized_emissions,
    beta_params = trained_model$optimized_betas
  )

  # Get the probabilities from the VERY LAST day in the dataset
  last_regime_probs <- tail(final_filter$alpha_matrix, 1)[1, ]

  cat("\n--- 7. FINAL ALLOCATION DECISION ---\n")
  print("Regime Probabilities (Last Day):")
  print(round(last_regime_probs, 4))

  # Run the final morphism
  final_target_allocation <- allocation_morphism(
    last_regime_probs,
    REGIME_ALLOCATIONS
  )

  print("Blended Target Allocation:")
  print(round(final_target_allocation, 4))
}
