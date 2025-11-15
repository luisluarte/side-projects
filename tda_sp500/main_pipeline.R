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

args <- commandArgs(trailingOnly = TRUE)

# 2. Source All HMM Morphisms
setwd(this.path::here()) # Set working directory if needed
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

# === 5. The Main Pipeline Execution ===

tdy <- Sys.Date()
cat("--- 1. LOADING DATA ---\n")

if (args[1] == "TRUE") {
  raw_data <- get_bitcoin_price_series(
    ticker = "BTCUSDT",
    source = "binance",
    start_date = "2018-01-01",
    end_date = as.character(tdy),
    interval = "1d"
  )
  saveRDS(raw_data, "btc.csv")
}
if (args[1] == "FALSE") {
  raw_data <- readRDS("btc.csv")
}

print(tail(raw_data))

# Convert to data.frame for easier handling
raw_data_df <- data.frame(
  Date = zoo::index(raw_data),
  Close = raw_data$close
)
raw_data_df <- raw_data %>%
  as.data.frame() %>%
  mutate(
    Close = close,
    Date = zoo::index(raw_data)
  )

cat("--- 2. PREPARING DATA & COVARIATES ---\n")
observations_vec <- calculate_log_returns(raw_data_df$Close)
covariates_df <- generate_covariates(observations_vec, window = 20)
print("Observations...")
print(tail(observations_vec))
print("Covariates")
print(tail(covariates_df))

# Align the data (removes initial NAs from rolling window)
aligned_data <- align_data(observations_vec, covariates_df)

cat("--- 3. GENERATING INITIAL PARAMETERS ---\n")
initial_model_params <- generate_initial_parameters(
  aligned_data$observations_vec,
  aligned_data$covariates_df,
  STATES
)
print(initial_model_params)

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

  cat("\n--- 6. RE-ATTACHING DATES & PRICES ---\n")

  # 1. Get original dates & prices, remove first (due to calculate_log_returns)
  dates_vec <- raw_data_df$Date[-1]
  prices_vec <- raw_data_df$Close[-1] # <-- ADDED THIS

  # 2. Find the same start index from align_data (to skip NA warmup)
  first_valid_index <- which(stats::complete.cases(covariates_df))[1]

  # 3. Slice the date and price vectors to match the aligned data
  aligned_dates <- dates_vec[first_valid_index:length(dates_vec)]
  aligned_prices <- prices_vec[first_valid_index:length(prices_vec)]

  # 4. Combine dates, prices, and the probability matrix
  if (length(aligned_dates) == nrow(final_filter$alpha_matrix) && length(aligned_prices) == nrow(final_filter$alpha_matrix)) {
    final_probabilities_df <- data.frame(
      Date = aligned_dates,
      Close = aligned_prices,
      final_filter$alpha_matrix
    )

    print("--- Filtered Probabilities with Dates & Prices (Last 5 Days) ---")
    print(tail(final_probabilities_df, 5))

    # This is a much more useful CSV file to save
    write_csv(
      x = final_probabilities_df,
      file = "final_probabilities_with_dates_prices.csv"
    )
  } else {
    cat("ERROR: Vector length(s) do not match alpha_matrix row count!\n")
  }
  # === END OF NEW SECTION ===


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
