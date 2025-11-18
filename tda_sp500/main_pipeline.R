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
  quantmod,
  this.path
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

cat("--- 1b. LOADING MACRO DATA ---\n")
start_date_macro <- as.Date("2018-01-01") # Match your BTC start
macro_data <- quantmod::getSymbols(
  "WALCL",
  src = "FRED",
  from = start_date_macro,
  to = tdy,
  auto.assign = FALSE
)

print(head(macro_data))

merged_xts <- merge(raw_data$close, macro_data, all = TRUE)

# Fill forward the weekly WALCL data to fill NA gaps
merged_xts$WALCL <- zoo::na.locf(merged_xts$WALCL, na.rm = FALSE)

# Now create the data.frame, trim NAs from the *start*
raw_data_df <- as.data.frame(merged_xts) %>%
  dplyr::mutate(
    Date = zoo::index(merged_xts),
    Close = close
  ) %>%
  dplyr::select(Date, Close, WALCL) %>% # Keep the new column
  stats::na.omit() # Remove NAs at the very beginning before data started

print("--- Merged Data Head (with WALCL) ---")
print(head(raw_data_df))

cat("--- 2. PREPARING DATA & COVARIATES ---\n")

# This new function prepares both observations and covariates
model_data <- generate_model_data(
  raw_data_df,
  vol_window = 20,
  obs_ma_window = 50
)

print("Observations...")
print(tail(model_data$observations_vec))
print("Covariates")
print(tail(model_data$covariates_df))

# Align the data (this function is still needed to remove NA warmup)
aligned_data <- align_data(
  model_data$observations_vec,
  model_data$covariates_df
)

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

  cat("\n--- 6a. RUNNING BACKWARD PASS for SMOOTHING ---\n")

  # Run the backward pass using the final optimized parameters
  final_beta_matrix <- run_backward_pass(
    observations_vec = aligned_data$observations_vec,
    covariates_df = aligned_data$covariates_df,
    state_names = STATES,
    emission_params = trained_model$optimized_emissions,
    beta_params = trained_model$optimized_betas,
    log_likelihood_vec = final_filter$log_likelihood_vec # <-- Use log-lik from forward pass
  )

  cat("--- 6b. CALCULATING SMOOTHED (GAMMA) PROBABILITIES ---\n")

  # Combine forward (alpha) and backward (beta) to get smoothed (gamma)
  final_gamma_matrix <- calculate_smoothed_gamma(
    alpha_matrix = final_filter$alpha_matrix,
    beta_matrix = final_beta_matrix,
    state_names = STATES
  )

  cat("\n--- 6. RE-ATTACHING DATES & PRICES ---\n")

  # 1. Get original dates & prices, remove first (due to calculate_log_returns)
  dates_vec <- raw_data_df$Date[-1]
  prices_vec <- raw_data_df$Close[-1] # <-- ADDED THIS

  # 2. Find the same start index from align_data (to skip NA warmup)
  first_valid_index <- which(stats::complete.cases(aligned_data$covariates_df))[1]

  # 3. Slice the date and price vectors to match the aligned data
  aligned_dates <- dates_vec[first_valid_index:length(dates_vec)]
  aligned_prices <- prices_vec[first_valid_index:length(prices_vec)]

  # 4. Combine dates, prices, and the probability matrix
  if (length(aligned_dates) == nrow(final_filter$alpha_matrix) && length(aligned_prices) == nrow(final_filter$alpha_matrix)) {
    final_probabilities_df <- data.frame(
      Date = aligned_dates,
      Close = aligned_prices,
      final_gamma_matrix
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

  # === 7. FINAL ALLOCATION DECISION ===

  # --- Define the window for allocation smoothing ---
  alloc_smooth_window <- 21

  cat(paste0(
    "\n--- 7. FINAL ALLOCATION DECISION (from ",
    alloc_smooth_window,
    "-day avg FILTERED probs) ---\n"
  ))

  # Get a 55-day moving average of the probabilities
  last_n_probs <- tail(final_filter$alpha_matrix, alloc_smooth_window)

  # Check if it's a matrix (fails if < 2 rows)
  if (is.matrix(last_n_probs)) {
    last_regime_probs <- colMeans(last_n_probs)
  } else {
    # Fallback for single row or vector (e.g., if dataset is very short)
    last_regime_probs <- last_n_probs
  }

  print(paste0(
    "Averaged Regime Probabilities (Last ",
    alloc_smooth_window,
    " Days):"
  ))
  print(round(last_regime_probs, 4))

  # Run the final morphism
  final_target_allocation <- allocation_morphism(
    last_regime_probs, # <--- This is now the 55-day average
    REGIME_ALLOCATIONS
  )

  print("Blended Target Allocation:")
  print(round(final_target_allocation, 4))
} else {
  cat("ERROR: Model training failed. No allocation will be made.\n")
}
