data {
  int<lower=1> N_trials;
  int<lower=1> N_subj;
  array[N_trials] int<lower=1, upper=N_subj> subj;
  array[N_trials] int<lower=1, upper=2> resp;
  array[N_trials] real reward;
  array[N_trials] real rt;
  array[N_trials] real iti;
  array[N_subj] real min_rt;
}
parameters {
  real mu_a_raw;
  real<lower=0> sigma_a;
  vector[N_subj] z_a;
  real mu_tnd_raw;
  real<lower=0> sigma_tnd;
  vector[N_subj] z_tnd;
  real mu_v_raw;
  real<lower=0> sigma_v;
  vector[N_subj] z_v;
  real mu_alpha_raw;
  real<lower=0> sigma_alpha;
  vector[N_subj] z_alpha;
}
transformed parameters {
  vector[N_subj] a = exp(mu_a_raw + sigma_a * z_a);
  vector[N_subj] tnd = to_vector(min_rt) .* (1.0 ./ (1.0 + exp(-(mu_tnd_raw + sigma_tnd * z_tnd))));
  vector[N_subj] v_ctx = exp(mu_v_raw + sigma_v * z_v);
  vector[N_subj] alpha_ctx = 1.0 ./ (1.0 + exp(-(mu_alpha_raw + sigma_alpha * z_alpha)));
}
generated quantities {
  vector[N_trials] log_lik;
  {
    array[N_subj] vector[2] Q;
    for (s in 1:N_subj) {
      Q[s] = rep_vector(0.5, 2);
    }
    for (t in 1:N_trials) {
      int s = subj[t];
      int ch = resp[t];
      real r = reward[t];
      real pe = r - Q[s, ch];
      
      real veff = (ch == 1) ? v_ctx[s] * (Q[s, 1] - Q[s, 2]) : -v_ctx[s] * (Q[s, 1] - Q[s, 2]);
      
      log_lik[t] = wiener_lpdf(rt[t] | a[s], tnd[s], 0.5, veff);
      
      Q[s, ch] += alpha_ctx[s] * pe;
    }
  }
}