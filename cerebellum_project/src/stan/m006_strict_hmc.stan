data {
  int<lower=1> N_trials;
  int<lower=1> N_subj;
  array[N_trials] int<lower=1, upper=N_subj> subj;
  array[N_trials] int<lower=1, upper=2> resp;
  array[N_trials] real reward;
  array[N_trials] real rt;
  array[N_trials] real iti;
  array[N_subj] real min_rt;
  matrix[N_subj, 32] W_exp;
  
  array[N_subj] int<lower=1> start_idx;
  array[N_subj] int<lower=1> end_idx;
  
  // Epistemic Geometry Preconditioning Data
  vector[9] theta_mean;
  matrix[9, 9] L_Sigma;
}
parameters {
  // Preconditioned, uncorrelated standard normal space for the 9 group means
  vector[9] theta_raw;
  
  // Group Variances
  vector<lower=0>[9] sigma;
  
  // Non-centered offsets
  matrix[9, N_subj] z;
}
transformed parameters {
  // 1. DENSE GEOMETRIC PRECONDITIONING (O(M^2) ops instead of O(N^2))
  // We project the 9-dimensional unconstrained space into the correlated epistemic geometry.
  vector[9] theta_unc = theta_mean + L_Sigma * theta_raw;
  
  // 2. UNCONSTRAINED PARAMETER EXTRACTION
  real mu_a_unc = theta_unc[1];
  real mu_tnd_unc = theta_unc[2];
  real mu_v_unc = theta_unc[3];
  
  // 3. BOUNDED PRIOR MAPPING
  real mu_a = 0.11 + 7.36 * inv_logit(mu_a_unc);
  real mu_tnd = 3.69 * inv_logit(mu_tnd_unc);
  real mu_v = 18.51 * inv_logit(mu_v_unc);
  
  vector[6] mu_res_raw = theta_unc[4:9];
  
  vector[N_subj] a_base_raw;
  vector[N_subj] tnd;
  vector[N_subj] v_ctx;
  vector[N_subj] alpha_ctx;
  vector[N_subj] alpha_pc;
  vector[N_subj] gamma;
  vector[N_subj] golgi_scale;
  vector[N_subj] tau_decay;
  vector[N_subj] w_u;
  
  for (s in 1:N_subj) {
    a_base_raw[s] = mu_a_unc + sigma[1] * z[1, s];
    
    // INCREASED BUFFER to prevent Navarro-Fuss infinite summation explosion
    real tnd_cap = fmin(min_rt[s] - 0.05, 3.69);
    tnd[s] = tnd_cap * inv_logit(mu_tnd_unc + sigma[2] * z[2, s]);
    
    v_ctx[s] = 18.51 * inv_logit(mu_v_unc + sigma[3] * z[3, s]);
    
    alpha_ctx[s]   = inv_logit(mu_res_raw[1] + sigma[4] * z[4, s]);
    alpha_pc[s]    = inv_logit(mu_res_raw[2] + sigma[5] * z[5, s]);
    gamma[s]       = exp(mu_res_raw[3] + sigma[6] * z[6, s]);
    golgi_scale[s] = exp(mu_res_raw[4] + sigma[7] * z[7, s]);
    tau_decay[s]   = exp(mu_res_raw[5] + sigma[8] * z[8, s]);
    w_u[s]         = exp(mu_res_raw[6] + sigma[9] * z[9, s]);
  }
}
model {
  // ----------------------------------------------------
  // I. JACOBIAN ADJUSTMENTS FOR NON-LINEAR PRIORS
  // ----------------------------------------------------
  target += log(7.36) + log_inv_logit(mu_a_unc) + log1m_inv_logit(mu_a_unc);
  target += log(3.69) + log_inv_logit(mu_tnd_unc) + log1m_inv_logit(mu_tnd_unc);
  target += log(18.51) + log_inv_logit(mu_v_unc) + log1m_inv_logit(mu_v_unc);

  // ----------------------------------------------------
  // II. TRAN ET AL. (2021) EMPIRICAL PRIORS
  // ----------------------------------------------------
  mu_a ~ gamma(11.69, 8.333);
  mu_tnd ~ student_t(1.32, 0.44, 0.08);
  mu_v ~ normal(1.76, 1.51);
  
  mu_res_raw ~ normal(0, 2);
  sigma ~ normal(0, 0.5);
  to_vector(z) ~ std_normal();
  
  // NOTE: theta_raw does NOT have a standard normal prior here. 
  // Its density is implicitly defined by the empirical priors on mu_a, mu_tnd, mu_v and mu_res_raw.
  // The L_Sigma transformation perfectly preconditions the Hamiltonian geometry for `diag_e`.

  // ----------------------------------------------------
  // III. VECTORIZED CONTINUOUS LIKELIHOOD COMPUTATION
  // ----------------------------------------------------
  vector[32] frac_alpha;
  vector[32] kappa_vec;
  for (i in 1:32) {
    frac_alpha[i] = 0.1 + 0.8 * ((i - 1) / 31.0);
    kappa_vec[i]  = 0.1 + 0.89 * ((i - 1) / 31.0);
  }
  vector[32] inv_frac_alpha = 1.0 - frac_alpha; 
  
  for (s in 1:N_subj) {
    vector[2] Q = rep_vector(0.5, 2);
    vector[32] frac_mem = rep_vector(0.0, 32);
    vector[32] Z = rep_vector(0.0, 32);
    vector[32] W_PC_latent = rep_vector(0.0, 32);
    
    int prev_ch = 1;
    real prev_E = 0.0;
    
    vector[32] W_exp_s = to_vector(W_exp[s]); 
    
    for (t in start_idx[s]:end_idx[s]) {
      int ch = resp[t];
      real R = reward[t];
      real current_iti = (iti[t] < 0) ? 1.0 : iti[t];
      real phys_decay = exp(-current_iti / tau_decay[s]);
      
      frac_mem = frac_alpha .* frac_mem + inv_frac_alpha .* (W_exp_s * Q[ch]);
      Z = phys_decay * (kappa_vec .* Z) + tanh(frac_mem);
      
      vector[32] W_PC_eff = 3.0 * tanh(W_PC_latent / 3.0);
      
      vector[32] abs_approx = sqrt(square(W_PC_eff .* Z) + 1e-8);
      vector[32] S_mask = tanh(golgi_scale[s] * abs_approx);
      
      vector[32] cb_components = S_mask .* W_PC_eff .* Z;
      real cb0 = sum(cb_components[1:16]);
      real cb1 = sum(cb_components[17:32]);
      
      real veff = v_ctx[s] * (Q[2] - Q[1]) + gamma[s] * (cb1 - cb0);
      real final_veff = 18.51 * tanh(veff / 18.51); 
      
      real a_raw = a_base_raw[s] + w_u[s] * sqrt(square(cb0) + 1e-8) * sqrt(square(cb1) + 1e-8);
      real a_dyn = 0.11 + 7.36 / (1.0 + exp(-a_raw));
      
      target += wiener_lpdf(rt[t] | a_dyn, tnd[s], 0.5, final_veff);
      
      prev_E = R - Q[ch];
      Q[ch] += alpha_ctx[s] * prev_E;
      prev_ch = ch;
      
      vector[32] err_sig = rep_vector(0.0, 32);
      if (ch == 1) err_sig[1:16] = rep_vector(prev_E, 16);
      else err_sig[17:32] = rep_vector(prev_E, 16);
      
      W_PC_latent += alpha_pc[s] * Z .* err_sig;
    }
  }
}
