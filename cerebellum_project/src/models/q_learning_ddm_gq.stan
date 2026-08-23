data {
  int<lower=1> N;
  int<lower=1> S;
  array[S] int<lower=1> start_idx;
  array[S] int<lower=1> end_idx;

  array[N] int<lower=0, upper=1> choice;
  array[N] real<lower=0> rt;
  array[N] real<lower=0, upper=1> reward;
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
  real mu_w_bias;

  real<lower=0> sigma_alpha_ctx;
  real<lower=0> sigma_kappa_ctx;
  real<lower=0> sigma_a;
  real<lower=0> sigma_tau_nd;
  real<lower=0> sigma_w_bias;

  vector[S] z_alpha_ctx;
  vector[S] z_kappa_ctx;
  vector[S] z_a;
  vector[S] z_tau_nd;
  vector[S] z_w_bias;
}

transformed parameters {
  vector[S] alpha_ctx = inv_logit(mu_alpha_ctx + sigma_alpha_ctx * z_alpha_ctx);
  vector[S] kappa_ctx = 10.0 * inv_logit(mu_kappa_ctx + sigma_kappa_ctx * z_kappa_ctx);
  vector[S] a = 0.5 + 4.5 * inv_logit(mu_a + sigma_a * z_a);
  vector[S] tau_nd = 0.001 + (to_vector(min_rt) - 0.002) .* inv_logit(mu_tau_nd + sigma_tau_nd * z_tau_nd);
  vector[S] w_bias = inv_logit(mu_w_bias + sigma_w_bias * z_w_bias);
}

generated quantities {
  vector[N] log_lik; // Predictive tensor baseline

  {
    for (s in 1:S) {
      int start_t = start_idx[s];
      int end_t = end_idx[s];
      vector[2] Q_ctx = rep_vector(0.5, 2);

      for (t in start_t:end_t) {
        real delta_Q_ctx = Q_ctx[2] - Q_ctx[1];
        real drift_sign = delta_Q_ctx >= 0 ? 1.0 : -1.0;
        real v_drift = drift_sign * sqrt(square(kappa_ctx[s] * delta_Q_ctx) + 1e-4);

        real log_uniform_dens = log(1.0 / 5.8);
        real wiener_lp;

        if (choice[t] == 1) {
          log_lik[t] = wiener_lpdf(rt[t] | a[s], tau_nd[s], w_bias[s], v_drift);
        } else {
          log_lik[t] = wiener_lpdf(rt[t] | a[s], tau_nd[s], 1.0 - w_bias[s], -v_drift);
        }
        log_lik[t] = log_mix(0.98, wiener_lp, log_uniform_dens);

        real RPE = reward[t] - Q_ctx[choice[t] + 1];
        Q_ctx[choice[t] + 1] += alpha_ctx[s] * RPE;
      }
    }
  }
}