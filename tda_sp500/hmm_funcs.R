# load libs ----
pacman::p_load(
  tidyverse,
  ggplot2,
  checkmate,
  purrr,
  nnet,
  cryptoQuotes,
  lubridate,
  zoo
)


# set current script location as working directory
setwd(this.path::here())
print(getwd())

# morphisms ----

## identity function ----
identity <- function(df) {
  df
}

## initial state ----
initial_state_morphism_uniform <- function(state_names) {
  # state are expressed as characters
  checkmate::assert_character(state_names, all.missing = FALSE)
  # more than one state required
  checkmate::assert_true(length(state_names) >= 1)
  checkmate::assert_character(state_names, unique = TRUE)

  # cardinality of state set (S)
  n_states <- length(state_names)

  # compute uniform probability
  uniform_prob <- 1 / n_states

  # pi is the codomain output Dist(S)
  pi_vector <- rep(uniform_prob, n_states)
  names(pi_vector) <- state_names

  # enforce codomain properties
  # numeric and sum to 1
  checkmate::assert_numeric(pi_vector)
  checkmate::assert_true(abs(sum(pi_vector) - 1.0) < 1e-6)

  pi_vector
}

## emission morphism ----
emission_morphism <- function(observation,
                              state_names,
                              all_state_params) {
  checkmate::assert_number(observation) # O_t (observation)
  checkmate::assert_list(all_state_params, names = "named")
  checkmate::assert_character(state_names) # S_i (state)

  # check that the requested state is within the parameters
  checkmate::assert_subset(state_names, names(all_state_params))

  params_to_use <- all_state_params[state_names]

  # compute probability density
  # log to avoid numerical underflow
  # vectorized operation
  log_density_vector <- purrr::map_dbl(
    params_to_use, ~ {
      checkmate::assert_list(.x, names = "named")
      checkmate::assert_subset(c("mean", "sd"), names(.x))

      dnorm(
        observation,
        mean = .x$mean,
        sd = .x$sd,
        log = TRUE
      )
    }
  )

  log_density_vector
}

# soft-max morphism ----
softmax_morphism <- function(logits) {
  checkmate::assert_numeric(logits, any.missing = FALSE, min.len = 1)
  stable_logits <- logits - max(logits)
  exps <- exp(stable_logits)
  exps / sum(exps)
}

# dynamic_transition_morphism ----
dynamic_transition_morphism <- function(covariates_t,
                                        beta_params,
                                        state_names) {
  checkmate::assert_numeric(covariates_t, names = "named")
  checkmate::assert_list(beta_params, names = "named")
  checkmate::assert_character(state_names)
  checkmate::assert_subset(state_names, names(beta_params))

  covariates_t <- c(intercept = 1.0, covariates_t)

  transition_rows_list <- purrr::map(state_names, function(from_state) {
    betas_for_row <- beta_params[[from_state]]

    linear_predictors <- purrr::map_dbl(betas_for_row, function(beta_vector) {
      common_names <- intersect(names(beta_vector), names(covariates_t))

      sum(covariates_t[common_names] * beta_vector[common_names])
    })

    softmax_row <- softmax_morphism(linear_predictors)

    softmax_row
  })

  transition_matrix <- do.call(rbind, transition_rows_list)

  rownames(transition_matrix) <- state_names
  colnames(transition_matrix) <- state_names

  checkmate::assert_matrix(
    transition_matrix,
    mode = "numeric",
    nrows = length(state_names),
    ncols = length(state_names)
  )

  row_sums <- rowSums(transition_matrix)
  checkmate::assert_true(all(abs(row_sums - 1.0) < 1e-6))

  transition_matrix
}

# forward step algorithm ----
forward_step_morphism <- function(alpha_scaled_prev,
                                  a_t,
                                  b_log_vec,
                                  state_names) {
  checkmate::assert_numeric(alpha_scaled_prev,
    len = length(state_names),
    names = "named"
  )
  checkmate::assert_true(abs(sum(alpha_scaled_prev) - 1.0) < 1e-6)
  checkmate::assert_matrix(
    a_t,
    mode = "numeric",
    nrows = length(state_names),
    ncols = length(state_names)
  )
  checkmate::assert_numeric(
    b_log_vec,
    len = length(state_names),
    names = "named"
  )
  checkmate::assert_character(state_names)

  # predict transition
  alpha_predicted <- alpha_scaled_prev %*% a_t
  alpha_predicted_vec <- setNames(as.vector(alpha_predicted), state_names)

  # update emissions
  b_vec <- exp(b_log_vec[state_names])

  # P(S_t, O_t | O_{1:t-1})
  alpha_unscaled <- alpha_predicted_vec * b_vec

  # scale
  marginal_likelihood_t <- sum(alpha_unscaled)

  # P(S_t | O_{1:t})
  alpha_scaled_next <- alpha_unscaled / marginal_likelihood_t

  # log likelihood
  log_likelihood_t <- log(marginal_likelihood_t)

  checkmate::assert_numeric(
    alpha_scaled_next,
    len = length(state_names),
    names = "named"
  )
  checkmate::assert_true(abs(sum(alpha_scaled_next) - 1.0) < 1e-6)
  checkmate::assert_number(log_likelihood_t)

  list(
    alpha_scaled = alpha_scaled_next,
    log_likelihood = log_likelihood_t
  )
}

# backward step algorithm ----
backward_step_morphism <- function(beta_scaled_next,
                                   a_t_next,
                                   b_log_vec_next,
                                   log_likelihood_next,
                                   state_names) {
  checkmate::assert_numeric(
    beta_scaled_next,
    len = length(state_names),
    names = "named"
  )
  checkmate::assert_matrix(
    a_t_next,
    mode = "numeric",
    nrows = length(state_names),
    ncols = length(state_names)
  )
  checkmate::assert_numeric(
    b_log_vec_next,
    len = length(state_names),
    names = "named"
  )
  checkmate::assert_number(log_likelihood_next)
  checkmate::assert_character(state_names)

  b_vec_next <- exp(b_log_vec_next[state_names])

  payload_vec <- b_vec_next * beta_scaled_next

  unscaled_beta_prev <- a_t_next %*% as.matrix(payload_vec)

  beta_scaled_prev <- unscaled_beta_prev / exp(log_likelihood_next)

  beta_scaled_prev_vec <- setNames(as.vector(beta_scaled_prev), state_names)

  checkmate::assert_numeric(
    beta_scaled_prev_vec,
    len = length(state_names),
    names = "named"
  )

  beta_scaled_prev_vec
}

# allocation morphism ----
allocation_morphism <- function(regime_probabilities,
                                regime_allocations) {
  checkmate::assert_numeric(regime_probabilities, names = "named")
  checkmate::assert_list(regime_allocations, names = "named")

  checkmate::assert_true(abs(sum(regime_probabilities) - 1.0) < 1e-6)

  checkmate::assert_subset(
    names(regime_probabilities),
    names(regime_allocations)
  )

  weighted_alloc_list <- purrr::map(names(regime_probabilities), function(state) {
    prob <- regime_probabilities[[state]]
    alloc_vector <- regime_allocations[[state]]

    checkmate::assert_numeric(alloc_vector)

    prob * alloc_vector
  })

  blended_allocation_vector <- Reduce("+", weighted_alloc_list)

  checkmate::assert_numeric(blended_allocation_vector, names = "named")
  checkmate::assert_true(abs(sum(blended_allocation_vector) - 1.0) < 1e-6)

  blended_allocation_vector
}

# run forward pass
run_forward_pass <- function(observations_vec,
                             covariates_df,
                             state_names,
                             initial_params,
                             emission_params,
                             beta_params) {
  n_obs <- length(observations_vec)
  n_states <- length(state_names)
  checkmate::assert_data_frame(covariates_df, nrows = n_obs)

  alpha_matrix <- matrix(
    0.0,
    nrow = n_obs,
    ncol = n_states,
    dimnames = list(NULL, state_names)
  )

  log_likelihood_vec <- numeric(n_obs)

  b_log_vec_t1 <- emission_morphism(
    observations_vec[1],
    state_names,
    emission_params
  )

  alpha_unscaled_t1 <- initial_params * exp(b_log_vec_t1[state_names])

  marginal_likelihood_t1 <- sum(alpha_unscaled_t1)
  log_likelihood_vec[1] <- log(marginal_likelihood_t1)

  alpha_matrix[1, ] <- alpha_unscaled_t1 / marginal_likelihood_t1

  if (n_obs < 2) {
    return(list(
      alpha_matrix = alpha_matrix,
      total_log_likelihood = log_likelihood_vec[1]
    ))
  }

  for (t in 2:n_obs) {
    alpha_prev <- alpha_matrix[t - 1, ]

    covariates_t <- as.numeric(covariates_df[t, ])
    names(covariates_t) <- colnames(covariates_df)

    a_t <- dynamic_transition_morphism(
      covariates_t,
      beta_params,
      state_names
    )

    b_log_vec_t <- emission_morphism(
      observations_vec[t],
      state_names,
      emission_params
    )

    step_output <- forward_step_morphism(
      alpha_scaled_prev = alpha_prev,
      a_t = a_t,
      b_log_vec = b_log_vec_t,
      state_names = state_names
    )

    alpha_matrix[t, ] <- step_output$alpha_scaled
    log_likelihood_vec[t] <- step_output$log_likelihood
  }

  list(
    alpha_matrix = alpha_matrix,
    log_likelihood_vec = log_likelihood_vec,
    total_log_likelihood = sum(log_likelihood_vec)
  )
}

# run backward pass ----
run_backward_pass <- function(observations_vec,
                              covariates_df,
                              state_names,
                              emission_params,
                              beta_params,
                              log_likelihood_vec) {
  n_obs <- length(observations_vec)
  n_states <- length(state_names)
  checkmate::assert_data_frame(covariates_df, nrows = n_obs)
  checkmate::assert_numeric(log_likelihood_vec, len = n_obs)

  beta_matrix <- matrix(
    0.0,
    nrow = n_obs,
    ncol = n_states,
    dimnames = list(NULL, state_names)
  )

  beta_matrix[n_obs, ] <- 1.0

  if (n_obs < 2) {
    return(beta_matrix)
  }

  for (t in (n_obs - 1):1) {
    beta_next <- beta_matrix[t + 1, ]

    covariates_next <- as.numeric(covariates_df[t + 1, ])
    names(covariates_next) <- colnames(covariates_df)

    a_t_next <- dynamic_transition_morphism(
      covariates_next,
      beta_params,
      state_names
    )

    b_log_vec_next <- emission_morphism(
      observations_vec[t + 1],
      state_names,
      emission_params
    )

    log_lik_next <- log_likelihood_vec[t + 1]

    beta_prev_vec <- backward_step_morphism(
      beta_scaled_next = beta_next,
      a_t_next = a_t_next,
      b_log_vec_next = b_log_vec_next,
      log_likelihood_next = log_lik_next,
      state_names = state_names
    )

    beta_matrix[t, ] <- beta_prev_vec
  }

  beta_matrix
}

# gamma morphism ----
calculate_smoothed_gamma <- function(alpha_matrix,
                                     beta_matrix,
                                     state_names) {
  checkmate::assert_matrix(alpha_matrix,
    mode = "numeric",
    ncols = length(state_names)
  )

  alpha_dims <- dim(alpha_matrix)

  checkmate::assert_matrix(beta_matrix,
    mode = "numeric",
    nrows = alpha_dims[1],
    ncols = alpha_dims[2]
  )

  gamma_unscaled <- alpha_matrix * beta_matrix

  row_sums <- rowSums(gamma_unscaled)

  row_sums[row_sums == 0] <- 1.0

  gamma_matrix <- sweep(gamma_unscaled, MARGIN = 1, STATS = row_sums, FUN = "/")

  checkmate::assert_matrix(gamma_matrix,
    mode = "numeric",
    nrows = alpha_dims[1],
    ncols = alpha_dims[2]
  )

  new_row_sums <- rowSums(gamma_matrix)
  checkmate::assert_true(all(abs(new_row_sums - 1.0) < 1e-6))

  colnames(gamma_matrix) <- state_names

  gamma_matrix
}

# smoothed xi
calculate_smoothed_xi <- function(alpha_matrix,
                                  beta_matrix,
                                  observations_vec,
                                  covariates_df,
                                  state_names,
                                  emission_params,
                                  beta_params) {
  n_obs <- length(observations_vec)
  n_states <- length(state_names)
  checkmate::assert_matrix(alpha_matrix,
    mode = "numeric",
    nrows = n_obs,
    ncols = n_states
  )
  checkmate::assert_matrix(beta_matrix,
    mode = "numeric",
    nrows = n_obs,
    ncols = n_states
  )
  checkmate::assert_data_frame(covariates_df, nrows = n_obs)

  xi_array <- array(
    0.0,
    dim = c(n_obs - 1, n_states, n_states),
    dimnames = list(
      t = 1:(n_obs - 1),
      from = state_names,
      to = state_names
    )
  )

  for (t in 1:(n_obs - 1)) {
    alpha_t <- alpha_matrix[t, ]

    covariates_next <- as.numeric(covariates_df[t + 1, ])
    names(covariates_next) <- colnames(covariates_df)
    a_t_next <- dynamic_transition_morphism(
      covariates_next,
      beta_params,
      state_names
    )

    b_log_vec_next <- emission_morphism(
      observations_vec[t + 1],
      state_names,
      emission_params
    )
    b_vec_next <- exp(b_log_vec_next)

    beta_next <- beta_matrix[t + 1, ]

    alpha_beta_prod <- alpha_t %o% beta_next

    a_b_prod <- sweep(a_t_next, MARGIN = 2, STATS = b_vec_next, FUN = "*")

    xi_unscaled_t <- alpha_beta_prod * a_b_prod

    matrix_sum <- sum(xi_unscaled_t)

    if (matrix_sum == 0) {
      xi_array[t, , ] <- matrix(0.0, n_states, n_states)
    } else {
      xi_array[t, , ] <- xi_unscaled_t / matrix_sum
    }
  }

  checkmate::assert_array(
    xi_array,
    mode = "numeric",
    d = 3
  )

  checkmate::assert_true(
    all(dim(xi_array) == c(n_obs - 1, n_states, n_states))
  )

  xi_array
}

# maximization step ----
m_step_update_emissions <- function(gamma_matrix,
                                    observations_vec,
                                    state_names) {
  checkmate::assert_matrix(
    gamma_matrix,
    mode = "numeric",
    ncols = length(state_names),
    nrows = length(observations_vec)
  )
  checkmate::assert_numeric(observations_vec)

  new_emission_params <- purrr::map(state_names, function(state) {
    gamma_t_vec <- gamma_matrix[, state]

    total_weight <- sum(gamma_t_vec)

    if (total_weight < 1e-9) {
      return(list(mean = 0, sd = 1e-3))
    }

    new_mean <- sum(gamma_t_vec * observations_vec) / total_weight

    new_variance <- sum(gamma_t_vec * (observations_vec - new_mean)^2) /
      total_weight

    new_sd <- sqrt(new_variance)
    if (new_sd < 1e-9) {
      new_sd <- 1e-9
    }

    list(mean = new_mean, sd = new_sd)
  })

  names(new_emission_params) <- state_names

  checkmate::assert_list(
    new_emission_params,
    names = "named",
    len = length(state_names)
  )

  new_emission_params
}

# m step udpate transitions ----
m_step_update_transitions <- function(xi_array,
                                      covariates_df,
                                      state_names) {
  n_obs <- nrow(covariates_df)
  n_states <- length(state_names)
  checkmate::assert_true(
    all(dim(xi_array) == c(n_obs - 1, n_states, n_states))
  )
  checkmate::assert_data_frame(covariates_df)

  predictors_df <- covariates_df[2:n_obs, , drop = FALSE]

  new_beta_params <- purrr::map(state_names, function(from_state) {
    response_matrix <- xi_array[, from_state, ]

    if (sum(response_matrix) < 1e-9) {
      warning(paste(
        "state",
        from_state,
        "had zero probability. cannot fit model."
      ))
      default_betas <- purrr::map(state_names, ~ c(intercept = 0.0))
      names(default_betas) <- state_names
      return(default_betas)
    }

    model <- tryCatch(
      {
        nnet::multinom(
          response_matrix ~ .,
          data = predictors_df,
          trace = FALSE
        )
      },
      error = function(e) {
        warning(paste(
          "nnet::multinom failed for state:", from_state,
          "Reverting to zero coefficients. Error:", e$message
        ))
        NULL # Return NULL on failure
      }
    )

    # === Reformat Coefficients ===
    if (is.null(model)) {
      # Model failed to fit, return the default zero-vector list
      default_betas <- purrr::map(state_names, ~ c(intercept = 0.0))
      names(default_betas) <- state_names
      return(default_betas) # Return default and skip to the next state
    }


    coef_matrix <- t(coef(model))

    calculated_states <- colnames(coef_matrix)

    ref_state <- setdiff(state_names, calculated_states)[1]

    ref_vector <- rep(0.0, nrow(coef_matrix))
    names(ref_vector) <- rownames(coef_matrix)

    beta_row_list <- as.list(as.data.frame(coef_matrix))
    beta_row_list[[ref_state]] <- ref_vector

    beta_row_list <- beta_row_list[state_names]

    beta_row_list_renamed <- purrr::map(beta_row_list, function(b) {
      names(b)[names(b) == "(Intercept)"] <- "intercept"
      b
    })

    beta_row_list_renamed
  })

  names(new_beta_params) <- state_names

  checkmate::assert_list(
    new_beta_params,
    names = "named",
    len = n_states
  )

  new_beta_params
}

# baum-welch
run_baum_welch_training <- function(observations_vec,
                                    covariates_df,
                                    state_names,
                                    initial_params,
                                    initial_emission_params,
                                    initial_beta_params,
                                    max_iterations = 100,
                                    tolerance = 1e-6) {
  current_emission_params <- initial_emission_params
  current_beta_params <- initial_beta_params
  log_likelihood_history <- numeric(max_iterations)

  checkmate::assert_character(state_names)

  cat("--- Starting Baum-Welch Training ---\n")
  pb <- utils::txtProgressBar(
    min = 0,
    max = max_iterations,
    style = 3, # Style 3 is a text bar: [====   ]
    width = 50, # Width of the bar
    char = "=" # Character to use
  )

  for (iter in 1:max_iterations) {
    forward_output <- run_forward_pass(
      observations_vec = observations_vec,
      covariates_df = covariates_df,
      state_names = state_names,
      initial_params = initial_params,
      beta_params = current_beta_params,
      emission_params = current_emission_params
    )

    total_log_likelihood <- forward_output$total_log_likelihood
    log_likelihood_history[iter] <- total_log_likelihood

    beta_matrix <- run_backward_pass(
      observations_vec = observations_vec,
      covariates_df = covariates_df,
      state_names = state_names,
      emission_params = current_emission_params,
      beta_params = current_beta_params,
      log_likelihood_vec = forward_output$log_likelihood_vec
    )

    gamma_matrix <- calculate_smoothed_gamma(
      forward_output$alpha_matrix,
      beta_matrix,
      state_names
    )

    xi_array <- calculate_smoothed_xi(
      forward_output$alpha_matrix,
      beta_matrix,
      observations_vec,
      covariates_df,
      state_names,
      emission_params = current_emission_params,
      beta_params = current_beta_params
    )

    new_emission_params <- m_step_update_emissions(
      gamma_matrix,
      observations_vec,
      state_names
    )

    new_beta_params <- m_step_update_transitions(
      xi_array,
      covariates_df,
      state_names
    )

    current_emission_params <- new_emission_params
    current_beta_params <- new_beta_params

    # === NEW: Update Progress Bar ===
    utils::setTxtProgressBar(pb, iter)
    # === END NEW ===

    if (iter > 1) {
      log_lik_change <- log_likelihood_history[iter] -
        log_likelihood_history[iter - 1]

      if (log_lik_change < tolerance) {
        cat(paste(
          "converged after",
          iter,
          "iteration. likelihood change:",
          log_lik_change,
          "\n"
        ))
        break
      }

      if (log_lik_change < 0) {
        warning(paste(
          "log-likelihood decreased at iteration",
          iter,
          ". stopping early."
        ))
        break
      }
    }
  }

  list(
    optimized_emissions = current_emission_params,
    optimized_betas = current_beta_params,
    log_likelihood_history = log_likelihood_history[1:iter]
  )
}

# return log returns
calculate_log_returns <- function(price_series) {
  checkmate::assert_numeric(
    price_series,
    all.missing = FALSE,
    min.len = 2
  )

  checkmate::assert_true(all(price_series > 0))

  log_prices <- log(price_series)

  log_returns <- diff(log_prices)

  checkmate::assert_numeric(
    log_returns,
    len = (length(price_series) - 1)
  )

  log_returns
}

# get btc price series
get_bitcoin_price_series <- function(ticker,
                                     source = "binance",
                                     start_date,
                                     end_date,
                                     batch_size = 365,
                                     interval = "1d") {
  checkmate::assert_string(ticker)
  checkmate::assert_string(source)
  checkmate::assert_date(as.Date(start_date))
  checkmate::assert_date(as.Date(end_date))
  checkmate::assert_integerish(batch_size, lower = 1)

  current_start_date <- as.Date(start_date)
  current_end_date <- as.Date(end_date)
  all_data_list <- list()

  while (current_end_date > current_start_date) {
    batch_start_date <- current_end_date - lubridate::days(batch_size)
    if (batch_start_date < current_start_date) {
      batch_start_date <- current_start_date
    }

    cat(paste(
      "fetching batch from",
      batch_start_date,
      "to",
      current_end_date,
      "\n"
    ))

    tryCatch(
      {
        batch_data <- cryptoQuotes::get_quote(
          ticker = ticker,
          source = source,
          interval = "1d",
          from = as.character(batch_start_date),
          to = as.character(current_end_date),
          futures = FALSE
        )

        all_data_list <- append(all_data_list, list(batch_data))

        current_end_date <- batch_start_date - lubridate::days(1)

        Sys.sleep(1)
      },
      error = function(e) {
        warning(paste(
          "api call failed for batch starting",
          batch_start_date,
          ":",
          e$message
        ))
        current_end_date <<- current_start_date
      }
    )

    if (batch_start_date == current_start_date) {
      break
    }
  }

  full_history_df <- do.call(rbind, all_data_list)

  if (nrow(full_history_df) > 0) {
    full_history_df <- full_history_df[
      !duplicated(zoo::index(full_history_df)),
    ]
    full_history_df <- full_history_df[
      order(zoo::index(full_history_df)),
    ]
  }

  return(full_history_df)
}

# generate initial params ----
generate_initial_parameters <- function(observations_vec,
                                        covariates_df,
                                        state_names) {
  checkmate::assert_numeric(observations_vec)
  checkmate::assert_data_frame(covariates_df)
  checkmate::assert_character(state_names)

  checkmate::assert_set_equal(state_names, c("bull", "bear", "sideways"))

  initial_params <- initial_state_morphism_uniform(state_names)

  kmeans_result <- kmeans(observations_vec, centers = 3, nstart = 25)
  cluster_centers <- kmeans_result$centers

  cluster_map <- setNames(
    order(cluster_centers),
    c("bear", "sideways", "bull")
  )[state_names]

  initial_emission_params <- purrr::map(
    cluster_map, function(cluster_id) {
      cluster_data <- observations_vec[kmeans_result$cluster == cluster_id]

      list(
        mean = mean(cluster_data),
        sd = sd(cluster_data)
      )
    }
  )
  names(initial_emission_params) <- state_names

  predictor_names <- colnames(covariates_df)

  zero_coef_vec <- c(
    intercept = 0.0,
    setNames(
      rep(0.0, length(predictor_names)),
      predictor_names
    )
  )

  initial_beta_params <- purrr::map(state_names, function(from_state) {
    to_list <- purrr::map(state_names, ~zero_coef_vec)
    names(to_list) <- state_names
    to_list
  })
  names(initial_beta_params) <- state_names

  list(
    initial_params = initial_params,
    initial_emission_params = initial_emission_params,
    initial_beta_params = initial_beta_params
  )
}

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

#' @title Data Alignment Morphism (Robust Version)
#' @description Aligns observations and covariates by removing ALL rows
#'              (initial or otherwise) that contain NA/NaN/Inf values.
#' @param obs_vec The vector of log-returns.
#' @param cov_df The data.frame of covariates.
#' @return A list containing the fully cleaned `observations_vec` and `covariates_df`.
align_data <- function(obs_vec, cov_df) {
  # 1. Combine into a temporary data frame for easy alignment
  #    This ensures obs and covs are checked row-by-row
  temp_df <- data.frame(obs = obs_vec, cov_df)

  # 2. Find all row indices that are "complete"
  #    stats::complete.cases checks for NA, NaN, and Inf in all columns
  complete_indices <- stats::complete.cases(temp_df)

  if (sum(complete_indices) == 0) {
    stop("Error in align_data: No complete cases found after merging obs/covs.")
  }

  # 3. Filter the entire data frame to keep only complete rows
  aligned_temp_df <- temp_df[complete_indices, , drop = FALSE]

  # 4. Separate back into the required vector and data frame
  aligned_obs <- aligned_temp_df$obs

  # Drop the 'obs' column (which is the first column)
  aligned_cov <- aligned_temp_df[, -1, drop = FALSE]

  cat(paste(
    "Data aligned. Original rows:", length(obs_vec),
    "Final valid rows:", length(aligned_obs), "\n"
  ))

  # 5. Return the cleaned and aligned list
  list(
    observations_vec = aligned_obs,
    covariates_df = aligned_cov
  )
}

generate_model_data <- function(data_df, vol_window = 20, obs_ma_window = 5) {
  checkmate::assert_data_frame(data_df)
  checkmate::assert_subset(c("Close", "WALCL"), colnames(data_df))

  # 1. Create RAW Observations (Log-Returns)
  # We need these for the volatility covariate
  raw_obs_vec <- calculate_log_returns(data_df$Close) # From hmm_funcs.R

  # 2. Create SMOOTHED Observations (for the emission model)
  # This is the new vector that will be returned as `observations_vec`
  smoothed_obs_vec <- zoo::rollapply(
    raw_obs_vec,
    width = obs_ma_window,
    FUN = mean,
    fill = NA,
    align = "right"
  )

  # 3. Create Covariates (based on RAW data)
  # Covariate 1: Rolling Volatility (using raw returns)
  rolling_vol <- zoo::rollapply(
    raw_obs_vec, # <-- Use raw returns for vol
    width = vol_window,
    FUN = sd,
    fill = NA,
    align = "right"
  )

  # Covariate 2: Log-Return of WALCL
  # We use lag() here to align it with the obs_vec (which has T-1 elements)
  walcl_log_return <- c(NA, diff(log(data_df$WALCL)))[-1]

  covariates_df <- data.frame(
    volatility = rolling_vol,
    walcl_log_return = walcl_log_return
  )

  # 4. Return aligned data
  # Note: smoothed_obs_vec and covariates_df have the same length (T-1)
  checkmate::assert_data_frame(covariates_df, nrows = length(smoothed_obs_vec))

  list(
    observations_vec = smoothed_obs_vec, # <-- Return the SMOOTHED vector
    covariates_df = covariates_df
  )
}
