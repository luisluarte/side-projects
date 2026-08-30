data {
  int<lower=1> N_trials;
  int<lower=1> N_subj;
  array[N_trials] int<lower=1, upper=N_subj> subj;
  array[N_trials] int<lower=1, upper=2> resp;
  array[N_trials] real reward;
  array[N_trials] real rt;
  array[N_subj] real min_rt;
}
parameters {
  real mu_a_raw;
  real mu_tnd_raw;
  real mu_v_raw;
  real mu_alpha_raw;
  real<lower=0> sigma_a;
  real<lower=0> sigma_tnd;
  real<lower=0> sigma_v;
  real<lower=0> sigma_alpha;
  vector[N_subj] z_a;
  vector[N_subj] z_tnd;
  vector[N_subj] z_v;
  vector[N_subj] z_alpha;
}
transformed parameters {
  vector[N_subj] a = exp(mu_a_raw + sigma_a * z_a);
  vector[N_subj] tnd = to_vector(min_rt) .* (1.0 ./ (1.0 + exp(-(mu_tnd_raw + sigma_tnd * z_tnd))));
  vector[N_subj] v_ctx = exp(mu_v_raw + sigma_v * z_v);
  vector[N_subj] alpha_ctx = 1.0 ./ (1.0 + exp(-(mu_alpha_raw + sigma_alpha * z_alpha)));
}
model {
  mu_a_raw ~ normal(0.5, 1);
  mu_tnd_raw ~ normal(0, 1);
  mu_v_raw ~ normal(0.5, 1);
  mu_alpha_raw ~ normal(0, 1);
  sigma_a ~ normal(0, 0.5);
  sigma_tnd ~ normal(0, 0.5);
  sigma_v ~ normal(0, 0.5);
  sigma_alpha ~ normal(0, 0.5);
  z_a ~ std_normal();
  z_tnd ~ std_normal();
  z_v ~ std_normal();
  z_alpha ~ std_normal();
  
  matrix[N_subj, 2] Q = rep_matrix(0.5, N_subj, 2);
  for (t in 1:N_trials) {
    int s = subj[t];
    int ch = resp[t];
    real R = reward[t];
    real veff = v_ctx[s] * (Q[s,2] - Q[s,1]);
    if (abs(veff) < 1e-4) {
      veff = veff >= 0 ? 1e-4 : -1e-4;
    }
    target += wiener_lpdf(rt[t] | a[s], tnd[s], 0.5, veff);
    Q[s, ch] += alpha_ctx[s] * (R - Q[s,ch]);
  }
}
generated quantities {
  array[N_trials] real log_lik;
  matrix[N_subj, 2] Q = rep_matrix(0.5, N_subj, 2);
  for (t in 1:N_trials) {
    int s = subj[t];
    int ch = resp[t];
    real R = reward[t];
    real veff = v_ctx[s] * (Q[s,2] - Q[s,1]);
    if (abs(veff) < 1e-4) veff = veff >= 0 ? 1e-4 : -1e-4;
    log_lik[t] = wiener_lpdf(rt[t] | a[s], tnd[s], 0.5, veff);
    Q[s, ch] += alpha_ctx[s] * (R - Q[s,ch]);
  }
}
