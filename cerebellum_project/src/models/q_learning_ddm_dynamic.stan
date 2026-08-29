functions {
  real partial_sum(
    array[] int seq_subj_slice, int start, int end,
    data array[] int choice, data array[] real rt, data array[] real reward, data array[] real iti,
    data array[] int start_idx, data array[] int end_idx,
    vector alpha_ctx, vector kappa_ctx, vector a, vector tau_nd, vector theta_ctx, vector beta_a
  ) {
    real target_sum = 0.0;

    for (s_idx in 1:size(seq_subj_slice)) {
      int s = seq_subj_slice[s_idx];
      int start_t = start_idx[s];
      int end_t = end_idx[s];
      int n_trials = end_t - start_t + 1;

      // Cortical Value State (Isolated Markovian formulation)
      vector[2] Q_ctx = rep_vector(0.5, 2);

      array[n_trials] real rt_subj;

      // Hoist scalar parameters
      real kappa_s = kappa_ctx[s];
      real alpha_s = alpha_ctx[s];
      real a_s = a[s];
      real tau_nd_s = tau_nd[s];
      real theta_ctx_s = theta_ctx[s];
      real beta_a_s = beta_a[s];

      for (t in start_t:end_t) {
        int idx = t - start_t + 1;
        rt_subj[idx] = rt[t];

        // 1. Kinematic Readout (Static Bias, Dynamic Drift)
        real delta_Q_ctx = Q_ctx[2] - Q_ctx[1];

        // Replace expensive sqrt(square + eps) curvature with piecewise linear bound
        real v_raw = kappa_s * delta_Q_ctx;
        real v_drift = v_raw;
        if (abs(v_drift) < 1e-4) {
          v_drift = v_drift >= 0 ? 1e-4 : -1e-4;
        }

        real prev_rt = (t == start_t) ? 0.5 : rt[t-1];
        real w_bias_t = 0.5 + 0.45 * tanh(theta_ctx_s * prev_rt);
        real a_t = a_s + beta_a_s * tanh(theta_ctx_s * prev_rt);
        if (a_t < 0.01) a_t = 0.01;

        real w_bias_subj;
        real v_subj;

        if (choice[t] == 1) {
          w_bias_subj = w_bias_t;
          v_subj = v_drift;
        } else {
          w_bias_subj = 1.0 - w_bias_t;
          v_subj = -v_drift;
        }

        // Add lpdf for this trial immediately, because a_t varies per trial
        target_sum += wiener_lpdf(rt_subj[idx] | a_t, tau_nd_s, w_bias_subj, v_subj);

        // 2. Discrete Cortical Plasticity (Rescorla-Wagner)
        real RPE = reward[t] - Q_ctx[choice[t] + 1];
        Q_ctx[choice[t] + 1] += alpha_s * RPE;
      }
    }
    return target_sum;
  }
}

data {
  int<lower=1> N;
  int<lower=1> S;
  array[S] int<lower=1> start_idx;
  array[S] int<lower=1> end_idx;

  array[N] int<lower=0, upper=1> choice;
  array[N] real<lower=0> rt;
  array[N] real<lower=0, upper=1> reward;
  array[N] real<lower=0> iti;
  array[S] real<lower=0> min_rt;

  int<lower=1> grainsize;
}

transformed data {
  array[S] int seq_subj;
  for (s in 1:S) seq_subj[s] = s;
}

parameters {
  // Cortical Hyper-means
  real mu_alpha_ctx;
  real mu_kappa_ctx;
  real mu_a;
  real mu_tau_nd;
  real mu_theta_ctx;
  real mu_beta_a;

  // Cortical Hyper-scales
  real<lower=0> sigma_alpha_ctx;
  real<lower=0> sigma_kappa_ctx;
  real<lower=0> sigma_a;
  real<lower=0> sigma_tau_nd;
  real<lower=0> sigma_theta_ctx;
  real<lower=0> sigma_beta_a;

  // Subject Deviates
  vector[S] z_alpha_ctx;
  vector[S] z_kappa_ctx;
  vector[S] z_a;
  vector[S] z_tau_nd;
  vector[S] z_theta_ctx;
  vector[S] z_beta_a;
}

transformed parameters {
  // Physiological Bounding Diffeomorphisms
  vector[S] alpha_ctx = inv_logit(mu_alpha_ctx + sigma_alpha_ctx * z_alpha_ctx);
  vector[S] kappa_ctx = 10.0 * inv_logit(mu_kappa_ctx + sigma_kappa_ctx * z_kappa_ctx);
  vector[S] a = 0.5 + 4.5 * inv_logit(mu_a + sigma_a * z_a);
  vector[S] theta_ctx = 10.0 * inv_logit(mu_theta_ctx + sigma_theta_ctx * z_theta_ctx);
  vector[S] beta_a = 5.0 * inv_logit(mu_beta_a + sigma_beta_a * z_beta_a);

  // Epsilon Kinematic Buffer
  vector[S] tau_nd = 0.001 + (to_vector(min_rt) - 0.002) .* inv_logit(mu_tau_nd + sigma_tau_nd * z_tau_nd);
}

model {
  // Hyper-Priors
  mu_alpha_ctx ~ normal(0, 1.5);
  mu_kappa_ctx ~ normal(0, 1.5);
  mu_a ~ normal(0, 1);
  mu_tau_nd ~ normal(-1, 1);
  mu_theta_ctx ~ normal(0, 1.5);
  mu_beta_a ~ normal(0, 1.5);

  sigma_alpha_ctx ~ normal(0, 1);
  sigma_kappa_ctx ~ normal(0, 1);
  sigma_a ~ normal(0, 1);
  sigma_tau_nd ~ normal(0, 1);
  sigma_theta_ctx ~ normal(0, 1);
  sigma_beta_a ~ normal(0, 1);

  // Non-centered hierarchical sampling
  z_alpha_ctx ~ std_normal();
  z_kappa_ctx ~ std_normal();
  z_a ~ std_normal();
  z_tau_nd ~ std_normal();
  z_theta_ctx ~ std_normal();
  z_beta_a ~ std_normal();

  target += reduce_sum(
    partial_sum, seq_subj, grainsize,
    choice, rt, reward, iti, start_idx, end_idx,
    alpha_ctx, kappa_ctx, a, tau_nd, theta_ctx, beta_a
  );
}
