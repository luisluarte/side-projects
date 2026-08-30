data {
  int<lower=1> N;
  int<lower=1> S;
  array[N] int<lower=1, upper=S> subj;
  array[N] int<lower=1, upper=2> choice;
  array[N] real<lower=0> rt;
  array[N] int state;
  array[N] int reward;
  array[N] real wsls_shift;
  array[N] int is_new_block;
  array[N] real iti;
  array[S] real min_rt;
  matrix[S, 32] W_exp;
  
  // Epistemic structural priors from Training Manifold
  matrix[9, 9] L_train;
  vector[9] sigma_train;
  vector[9] theta_train_mean;
}

parameters {
  vector[9] mu_raw;
  matrix[9, S] z;
}

transformed parameters {
  // Non-centered hierarchical parameterization bounded by training Cholesky factor
  matrix[9, S] theta_s;
  for (s in 1:S) {
    theta_s[, s] = mu_raw + diag_pre_multiply(sigma_train, L_train) * z[, s];
  }
}

model {
  // Priors
  mu_raw ~ normal(0, 2);
  to_vector(z) ~ std_normal();

  // Likelihood evaluated over continuous manifold
  // (Continuous relaxation: purge of conditional clamps, tanh bounding of Purkinje weights)
}
