# load libs ----
pacman::p_load(
  tidyverse,
  ggplot2,
  checkmate,
  data.table
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
states <- c("bull", "bear", "sideways")
initial_state_moprhism_uniform <- function(state_names) {
  # state are expressed as characters
  checkmate::assert_character(state_names, all.missing = FALSE)
  # more than one state required
  checkmate::assert_true(length(state_names) >= 1)
  checkmate::assert_true(c(
    checkmate::assert_character()(state_names),
    data.table::uniqueN(state_names) == length(state_names)
  ))

  # cardinality of state set (S)
  n_states <- data.table::uniqueN(state_names)

  # compute uniform probability
  uniform_prob <- 1 / n_states

  # pi is the codomain output Dist(S)
  pi_vector <- rep(uniform_prob, state_names)
  names(pi_vector) <- state_names

  # enforce codomain properties
  # numeric and sum to 1
  checkmate::assert_numeric(pi_vector)
  checkmate::assert_true(abs(sum(pi_vector) - 1.0) < 1e-6)

  pi_vector
}
