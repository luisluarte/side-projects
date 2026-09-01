# libs --------------------------------------------------------------------
pacman::p_load(
  tidyverse
)

make_mvt_environment <- function(
  depletion_rate = 0.90,
  travel_time = 4,
  baseline_max = 1.0
) {
  # hidden environment state
  current_yield <- runif(1, min = baseline_max * 0.6, max = baseline_max)
  in_transit <- 0

  # step function
  step <- function(action) {
    # action 0 stay/exploit
    # action 1 switch/explore

    if (in_transit > 0) {
      in_transit <<- in_transit - 1

      if (in_transit == 0) {
        current_yield <<- runif(1, min = baseline_max * 0.6, max = baseline_max)
      }
      return(0.0)
    }

    # agent switches
    if (action == 1) {
      in_transit <<- travel_time
      return(0.0)
    }

    # agent stays
    if (action == 0) {
      reward_delivered <- current_yield
      current_yield <<- (current_yield * depletion_rate) + rnorm(1, 0, 0.02)
      current_yield <<- max(0, current_yield)

      return(reward_delivered)
    }
  }

  return(list(step = step))
}
