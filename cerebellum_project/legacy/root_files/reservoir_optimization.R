# ==============================================================================
# EXACT-R: In Silico Pre-Training and Phenotypic Optimization
# ==============================================================================
library(Rcpp)
library(RcppEigen)
library(Matrix)
library(plotly)

setwd(this.path::here())

# 1. Source External Modules
source("mvt_generator.R")
sourceCpp("reservoir.cpp")

# ==============================================================================
# Phase 1: Immutable Topological Generation (Executed Once)
# ==============================================================================
cat("Generating Immutable CSR Reservoir Topology...\n")
set.seed(42)

# Dimensions
N_GC <- 2000          # Granular Reservoir Size
d_channels <- 100     # Mossy Fiber Expansion Basis
N_GoC <- 200          # Golgi Cell Bottleneck Size
N_actions <- 2        # 2AFC: 0 (Stay), 1 (Switch)
state_dim <- 2        # Raw state: [Action, Reward]

# Structural Sparsity Constraints
# W_in: ~10% dense mapping from expanded Mossy Fibers
W_in <- rsparsematrix(N_GC, d_channels, density = 0.1)

# W_fb: ~10% dense feedback from Granule to Golgi
W_fb <- rsparsematrix(N_GoC, N_GC, density = 0.1)

# W_inh: Strictly k_inh = 4 non-zeros per row (Glomerular constraint)
W_inh <- Matrix(0, nrow = N_GC, ncol = N_GoC, sparse = TRUE)
for(i in 1:N_GC) {
  active_goc <- sample(1:N_GoC, 4)
  W_inh[i, active_goc] <- runif(4, 0.1, 1.0)
}

# W_collateral: Feedback from Actor-Critic Readout to Golgi
W_collateral <- rsparsematrix(N_GoC, 1 + N_actions, density = 0.5)

# Mossy Fiber Immutable Expansion Parameters
c_j <- sample(1:state_dim, d_channels, replace = TRUE)
beta_j <- runif(d_channels, 0.1, 1.0)

# ==============================================================================
# Phase 2: The Composite Objective Function (Stateful Simulation)
# ==============================================================================

# Evaluates the fitness of a specific phenotype lambda [0, 1]
evaluate_phenotype <- function(lambda, T_epochs = 1000, gamma_entropy = 0.5) {

  # 1. Algebraic Constraint Manifold (Phenotypic Axis Derivations)
  tau_half_val <- 50 * exp(2 * lambda)
  eta_val      <- 0.1 * exp(-3 * lambda)
  k_ent_val    <- 0.5 + 4 * lambda
  rho_max      <- 0.95

  # 2. Derive Granular Heterogeneity (Log-Normal Tau)
  tau_vector <- rlnorm(N_GC, meanlog = log(tau_half_val), sdlog = 0.5)
  tau_vector <- pmax(10, pmin(tau_vector, 1000))
  rho_base   <- rho_max * (tau_vector / (tau_vector + tau_half_val))

  # 3. Instantiate the C++ EXACT-R Model
  # Notice we construct the object directly from the exported module
  model <- new(ExactRModel,
               W_in, W_fb, W_inh, W_collateral,
               rho_base, tau_vector, N_actions,
               eta_val, eta_val, k_ent_val)

  # 4. Instantiate the MVT Environment
  env <- make_mvt_environment(depletion_rate = 0.90, travel_time = 4)

  # 5. Simulation Variables
  total_reward <- 0
  total_entropy <- 0

  # Initial state
  a_prev <- 0
  r_prev <- 0
  delta_t_const <- 100 # Assuming fixed 100ms between events
  ttf_t_const <- 200   # Assuming fixed 200ms feedback delay

  # State History Buffer (For MF Expansion)
  state_buffer <- matrix(0, nrow = T_epochs + 1, ncol = state_dim)

  # 6. The Closed-Loop Execution
  for(t in 1:T_epochs) {
    # Update Buffer
    state_buffer[t, ] <- c(a_prev, r_prev)

    # MF Expansion (Simplified to Tonic for speed; Phasic requires delay indexing)
    u_t <- 1 / (1 + exp(-(beta_j * state_buffer[t, c_j])))

    # Forward Pass (C++)
    fwd_out <- model$forward_pass(u_t, delta_t_const)
    pi_t <- fwd_out$Policy

    # Sample Action from Policy (Stochastic 2AFC)
    # C++ policy returns probabilities for [Stay, Switch]
    a_t <- sample(0:1, size = 1, prob = pi_t)

    # Environmental Step
    r_t <- env$step(a_t)

    # Backward Pass (C++)
    bwd_out <- model$backward_pass(a_t + 1, r_t, ttf_t_const)

    # Accumulate Fitness Metrics
    total_reward <- total_reward + r_t
    total_entropy <- total_entropy + bwd_out$S_t

    # Setup next epoch
    a_prev <- a_t
    r_prev <- r_t
  }

  # 7. Composite Loss Calculation J(\lambda)
  # We want to MAXIMIZE J, so we return negative J for minimization algorithms
  J_lambda <- total_reward + (gamma_entropy * total_entropy)
  return(-J_lambda)
}

# ==============================================================================
# Phase 3: Edge of Chaos Optimization (1D Continuous Search)
# ==============================================================================
cat("Executing in silico Evolutionary Pre-Training...\n")

# To visualize the manifold, we perform a grid search first
lambda_grid <- seq(0, 1, length.out = 15)
fitness_landscape <- numeric(length(lambda_grid))

for(i in seq_along(lambda_grid)) {
  # We negate the output to plot positive fitness
  fitness_landscape[i] <- -evaluate_phenotype(lambda_grid[i], T_epochs = 500)
  cat(sprintf("Evaluated Phenotype lambda = %.2f | Composite Fitness: %.2f\n",
              lambda_grid[i], fitness_landscape[i]))
}

# Formal Continuous Optimization (Brent's Method)
optimal_result <- optimize(f = evaluate_phenotype, interval = c(0, 1), T_epochs = 1000)
optimal_lambda <- optimal_result$minimum
max_fitness <- -optimal_result$objective

cat(sprintf("\n=== OPTIMIZATION COMPLETE ===\nOptimal Phenotypic Lambda: %.4f\nMaximum Fitness Achieved: %.2f\n",
            optimal_lambda, max_fitness))

# ==============================================================================
# Phase 4: Manifold Projection (Plotly)
# ==============================================================================
plot_data <- data.frame(Lambda = lambda_grid, Fitness = fitness_landscape)

plot_data %>%
  ggplot(aes(
    Lambda, Fitness
  )) +
  geom_line()
