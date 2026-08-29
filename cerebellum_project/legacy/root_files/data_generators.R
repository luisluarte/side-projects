# ==============================================================================
# EXACT-R: Synthetic Pre-Training Dataset Generation Protocols (\mathcal{D}_{gen})
# ==============================================================================
# Provides 4 distinct input signal families for Mossy Fiber inputs u_t \in \mathbb{R}^{N_{MF}}
# Uses low-dimensional motor/sensory manifold projection (3-5 basis dimensions -> N_channels)

generate_poisson_data <- function(T_steps = 1000, N_channels = 100, delta_t = 0.01, 
                                  min_rate = 1.0, max_rate = 100.0, smooth_tau = 0.05, n_basis = 4) {
  # Generate n_basis population rate trajectories
  t_vec <- (0:(T_steps - 1)) * delta_t
  basis_rates <- matrix(0, nrow = T_steps, ncol = n_basis)
  for (b in 1:n_basis) {
    base_f <- runif(1, 0.5, 5.0)
    basis_rates[, b] <- min_rate + (max_rate - min_rate) * 0.5 * (1 + sin(2 * pi * base_f * t_vec + runif(1, 0, 2*pi)))
  }
  
  # Project to N_channels Mossy Fibers
  set.seed(42)
  proj_mat <- matrix(runif(n_basis * N_channels, 0.1, 1.0), nrow = n_basis, ncol = N_channels)
  proj_mat <- proj_mat / rowSums(proj_mat)
  
  rate_matrix <- basis_rates %*% proj_mat
  
  # Generate Poisson spikes
  spikes <- matrix(0, nrow = T_steps, ncol = N_channels)
  for (j in 1:N_channels) {
    p_spike <- 1 - exp(-rate_matrix[, j] * delta_t)
    spikes[, j] <- rbinom(T_steps, 1, p_spike)
  }
  
  # Exponential smoothing filter
  smooth_alpha <- exp(-delta_t / smooth_tau)
  u_data <- matrix(0, nrow = T_steps, ncol = N_channels)
  u_data[1, ] <- spikes[1, ]
  for (t in 2:T_steps) {
    u_data[t, ] <- smooth_alpha * u_data[t - 1, ] + (1 - smooth_alpha) * spikes[t, ]
  }
  
  # Min-max scale to [0, 1]
  u_min <- apply(u_data, 2, min)
  u_max <- apply(u_data, 2, max)
  for (j in 1:N_channels) {
    if (u_max[j] > u_min[j]) {
      u_data[, j] <- (u_data[, j] - u_min[j]) / (u_max[j] - u_min[j])
    }
  }
  return(u_data)
}

generate_filtered_noise_data <- function(T_steps = 1000, N_channels = 100, delta_t = 0.01,
                                         alpha_psd = 1.0, f_low = 1.0, f_high = 50.0, n_basis = 4) {
  # Generate n_basis colored noise trajectories
  basis_matrix <- matrix(0, nrow = T_steps, ncol = n_basis)
  freqs <- (0:(T_steps - 1)) / (T_steps * delta_t)
  
  for (b in 1:n_basis) {
    white_noise <- rnorm(T_steps)
    fft_noise <- fft(white_noise)
    
    mag_filter <- rep(0, T_steps)
    valid_idx <- freqs > 0
    mag_filter[valid_idx] <- (freqs[valid_idx])^(-alpha_psd / 2)
    bandpass_idx <- freqs >= f_low & freqs <= f_high
    mag_filter[!bandpass_idx] <- 0
    
    filtered_fft <- fft_noise * mag_filter
    basis_matrix[, b] <- Re(fft(filtered_fft, inverse = TRUE)) / T_steps
  }
  
  # Project to N_channels Mossy Fibers
  set.seed(123)
  proj_mat <- matrix(rnorm(n_basis * N_channels), nrow = n_basis, ncol = N_channels)
  u_raw <- basis_matrix %*% proj_mat
  
  u_data <- 1 / (1 + exp(-0.5 * u_raw))
  return(u_data)
}

generate_lorenz_data <- function(T_steps = 1000, N_channels = 100, delta_t = 0.01,
                                 sigma = 10.0, rho = 28.0, beta = 8/3) {
  # RK4 integration of Lorenz-63 chaotic attractor
  lorenz_step <- function(state) {
    x <- state[1]; y <- state[2]; z <- state[3]
    dx <- sigma * (y - x)
    dy <- x * (rho - z) - y
    dz <- x * y - beta * z
    c(dx, dy, dz)
  }
  
  state_matrix <- matrix(0, nrow = T_steps, ncol = 3)
  curr <- c(1.0, 1.0, 1.0)
  state_matrix[1, ] <- curr
  
  dt <- delta_t
  for (t in 2:T_steps) {
    k1 <- lorenz_step(curr)
    k2 <- lorenz_step(curr + 0.5 * dt * k1)
    k3 <- lorenz_step(curr + 0.5 * dt * k2)
    k4 <- lorenz_step(curr + dt * k3)
    curr <- curr + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    state_matrix[t, ] <- curr
  }
  
  # Random linear projection to N_channels Mossy Fibers
  set.seed(456)
  proj_matrix <- matrix(rnorm(3 * N_channels), nrow = 3, ncol = N_channels)
  u_raw <- state_matrix %*% proj_matrix
  
  u_data <- 1 / (1 + exp(-0.1 * u_raw))
  return(u_data)
}

generate_kinematic_data <- function(T_steps = 1000, N_channels = 100, delta_t = 0.01,
                                    n_basis = 4, min_freq = 0.5, max_freq = 15.0) {
  # Generate n_basis joint angle / muscle synergy trajectories
  time_vec <- (0:(T_steps - 1)) * delta_t
  basis_matrix <- matrix(0, nrow = T_steps, ncol = n_basis)
  
  for (b in 1:n_basis) {
    f1 <- runif(1, min_freq, max_freq)
    f2 <- runif(1, min_freq, max_freq)
    p1 <- runif(1, 0, 2*pi)
    p2 <- runif(1, 0, 2*pi)
    basis_matrix[, b] <- sin(2 * pi * f1 * time_vec + p1) + 0.5 * cos(2 * pi * f2 * time_vec + p2)
  }
  
  # Project to N_channels Mossy Fibers
  set.seed(789)
  proj_matrix <- matrix(rnorm(n_basis * N_channels), nrow = n_basis, ncol = N_channels)
  u_raw <- basis_matrix %*% proj_matrix
  
  u_data <- 1 / (1 + exp(-0.5 * u_raw))
  return(u_data)
}

# Wrapper function for protocol selection
generate_pretraining_data <- function(protocol_name, T_steps = 1000, N_channels = 100, delta_t = 0.01) {
  if (protocol_name == "Poisson") {
    return(generate_poisson_data(T_steps, N_channels, delta_t))
  } else if (protocol_name == "Filtered") {
    return(generate_filtered_noise_data(T_steps, N_channels, delta_t))
  } else if (protocol_name == "Lorenz") {
    return(generate_lorenz_data(T_steps, N_channels, delta_t))
  } else if (protocol_name == "Kinematic") {
    return(generate_kinematic_data(T_steps, N_channels, delta_t))
  } else {
    stop("Unknown protocol_name: ", protocol_name)
  }
}
