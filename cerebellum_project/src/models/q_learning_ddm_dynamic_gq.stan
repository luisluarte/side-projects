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
  real mu_alpha_ctx;
  real mu_kappa_ctx;
  real mu_a;
  real mu_tau_nd;
  real mu_theta_ctx;
  real mu_beta_a;

  real<lower=0> sigma_alpha_ctx;
  real<lower=0> sigma_kappa_ctx;
  real<lower=0> sigma_a;
  real<lower=0> sigma_tau_nd;
  real<lower=0> sigma_theta_ctx;
  real<lower=0> sigma_beta_a;

  vector[S] z_alpha_ctx;
  vector[S] z_kappa_ctx;
  vector[S] z_a;
  vector[S] z_tau_nd;
  vector[S] z_theta_ctx;
  vector[S] z_beta_a;
}

transformed parameters {
  vector[S] alpha_ctx = inv_logit(mu_alpha_ctx + sigma_alpha_ctx * z_alpha_ctx);
  vector[S] kappa_ctx = 10.0 * inv_logit(mu_kappa_ctx + sigma_kappa_ctx * z_kappa_ctx);
  vector[S] a = 0.5 + 4.5 * inv_logit(mu_a + sigma_a * z_a);
  vector[S] theta_ctx = 10.0 * inv_logit(mu_theta_ctx + sigma_theta_ctx * z_theta_ctx);
  vector[S] beta_a = 5.0 * inv_logit(mu_beta_a + sigma_beta_a * z_beta_a);
  vector[S] tau_nd = 0.001 + (to_vector(min_rt) - 0.002) .* inv_logit(mu_tau_nd + sigma_tau_nd * z_tau_nd);
}

model {
  // Empty model block for generated quantities
}

generated quantities {
  vector[N] log_lik;
  for (s in 1:S) {
    int start_t = start_idx[s];
    int end_t = end_idx[s];
    
    vector[2] Q_ctx = rep_vector(0.5, 2);

    real kappa_s = kappa_ctx[s];
    real alpha_s = alpha_ctx[s];
    real a_s = a[s];
    real tau_nd_s = tau_nd[s];
    real theta_ctx_s = theta_ctx[s];
    real beta_a_s = beta_a[s];
    
    real log_uniform_dens = log(1.0 / 5.8);

    for (t in start_t:end_t) {
      real delta_Q_ctx = Q_ctx[2] - Q_ctx[1];
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

      real wiener_lp;
      if (rt[t] - tau_nd_s < 1e-4) {
        log_lik[t] = log_uniform_dens;
      } else {
        wiener_lp = wiener_lpdf(rt[t] | a_t, tau_nd_s, w_bias_subj, v_subj);
        log_lik[t] = log_mix(0.98, wiener_lp, log_uniform_dens);
      }

      real RPE = reward[t] - Q_ctx[choice[t] + 1];
      Q_ctx[choice[t] + 1] += alpha_s * RPE;
    }
  }
}
