# load libs ----
pacman::p_load(
  tidyverse,
  ggplot2,
  checkmate,
  data.table,
  purrr
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
  n_states <- data.table::uniqueN(state_names)

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

softmax_morphism <- function(logits) {
  stable_logits <- logits - max(logits)
  exps <- exp(stable_logits)
  exp / sum(exps)
}

dynamic_transition_morphism <- function(covariates_t,
                                        beta_params,
                                        state_names) {
  checkmate::assert_numeric(covariates_t, names = "named")
  checkmate::assert_list(beta_params, names = "named")
  checkmate::assert_character(state_names)
  checkmate::assert_subset(state_names, names(beta_params))

  covariates_t <- c(intercept = 1.0, covariates_t)

  transition_rows_list <- purrr:map(state_names, function(from_state) {
    betas_for_row <- beta_params[[from_state]]

    linear_predictors <- purrr::map_dbl(betas_for_row, function(beta_vector) {
      common_names <- intersect(names(beta_vector), names(covariates_t))

      sum(covariates_t[common_names] * beta_vector[common_names])
    })

    softmax_row <- softmax_morphism(linear_predictors)

    softmax_row
  })
}

# test bed ----
states <- c("bull", "bear", "sideways")
example_emission_params <- list(
  "bull" = list(mean = 0.0008, sd = 0.007),
  "bear" = list(mean = -0.001, sd = 0.015),
  "sideways" = list(mean = 0.0001, sd = 0.004)
)
obs_return <- 0.01
log_likelihoods <- emission_morphism(
  obs_return,
  states,
  example_emission_params
)

cat(paste("\n", states[1], "likelihoods are:", log_likelihoods[1], "\n"))
cat(paste("\n", states[2], "likelihoods are:", log_likelihoods[2], "\n"))
cat(paste("\n", states[3], "likelihoods are:", log_likelihoods[1], "\n"))
