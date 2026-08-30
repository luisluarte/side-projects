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
}
parameters {
  // Literature-grounded Group Means (Tran et al. 2021)
  real<lower=0.11, upper=7.47> mu_a;
  real<lower=0, upper=3.69> mu_tnd;
  real<lower=0, upper=18.51> mu_v;
  
  // Uninformative Group Means for Reservoir
  vector[6] mu_res_raw;
  
  // Group Variances
  vector<lower=0>[9] sigma;
  
  // Non-centered offsets
  matrix[9, N_subj] z;
}
transformed parameters {
  // Map literature priors to unbounded base for non-centered parameterization
  real mu_a_raw = -log(7.36 / (mu_a - 0.11) - 1.0);
  real mu_v_raw = -log(18.51 / mu_v - 1.0);
  
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
    a_base_raw[s] = mu_a_raw + sigma[1] * z[1, s];
    
    // Smoothly bound subject tnd strictly below min_rt[s]
    real tnd_max = min_rt[s] - 0.01; 
    tnd[s] = tnd_max / (1.0 + exp(-(mu_tnd + sigma[2] * z[2, s])));
    
    // Subject drift bounded implicitly up to 18.51
    v_ctx[s] = 18.51 / (1.0 + exp(-(mu_v_raw + sigma[3] * z[3, s])));
    
    // Reservoir parameters
    alpha_ctx[s] = 1.0 / (1.0 + exp(-(mu_res_raw[1] + sigma[4] * z[4, s])));
    alpha_pc[s]  = 1.0 / (1.0 + exp(-(mu_res_raw[2] + sigma[5] * z[5, s])));
    gamma[s]       = exp(mu_res_raw[3] + sigma[6] * z[6, s]);
    golgi_scale[s] = exp(mu_res_raw[4] + sigma[7] * z[7, s]);
    tau_decay[s]   = exp(mu_res_raw[5] + sigma[8] * z[8, s]);
    w_u[s]         = exp(mu_res_raw[6] + sigma[9] * z[9, s]);
  }
}
model {
  // ----------------------------------------------------
  // I. TRAN ET AL. (2021) EMPIRICAL PRIORS (Melchior)
  // ----------------------------------------------------
  
  // Boundary Separation (a): Gamma distribution (shape=11.69, scale=0.12 -> rate=8.33)
  mu_a ~ gamma(11.69, 8.333);
  
  // Non-Decision Time (Ter): Truncated t-distribution (df=1.32, mu=0.44, sigma=0.08)
  mu_tnd ~ student_t(1.32, 0.44, 0.08);
  
  // Drift Rate (v): Truncated Normal (mu=1.76, sigma=1.51)
  mu_v ~ normal(1.76, 1.51);
  
  // Uninformative priors for Reservoir Latents
  mu_res_raw ~ normal(0, 2);
  sigma ~ normal(0, 0.5);
  to_vector(z) ~ std_normal();
  
  // ----------------------------------------------------
  // II. CONTINUOUS LIKELIHOOD COMPUTATION
  // ----------------------------------------------------
  vector[32] frac_alpha;
  vector[32] kappa_vec;
  for (i in 1:32) {
    frac_alpha[i] = 0.1 + 0.8 * ((i - 1) / 31.0);
    kappa_vec[i] = 0.1 + 0.89 * ((i - 1) / 31.0);
  }
  
  for (s in 1:N_subj) {
    vector[2] Q = rep_vector(0.5, 2);
    vector[32] frac_mem = rep_vector(0.0, 32);
    vector[32] Z = rep_vector(0.0, 32);
    vector[32] W_PC_latent = rep_vector(0.0, 32);
    int prev_ch = 1;
    real prev_E = 0.0;
    
    for (t in start_idx[s]:end_idx[s]) {
      int ch = resp[t];
      real R = reward[t];
      real current_iti = (iti[t] < 0) ? 1.0 : iti[t];
      real phys_decay = exp(-current_iti / tau_decay[s]);
      
      real cb0 = 0.0;
      real cb1 = 0.0;
      
      for (i in 1:32) {
        frac_mem[i] = frac_alpha[i] * frac_mem[i] + (1.0 - frac_alpha[i]) * W_exp[s, i] * Q[ch];
        Z[i] = phys_decay * kappa_vec[i] * Z[i] + tanh(frac_mem[i]);
        
        real W_PC_eff = 3.0 * tanh(W_PC_latent[i] / 3.0);
        real S_mask = tanh(golgi_scale[s] * abs(W_PC_eff * Z[i]));
        
        if (i <= 16) {
          cb0 += S_mask * W_PC_eff * Z[i];
        } else {
          cb1 += S_mask * W_PC_eff * Z[i];
        }
      }
      
      real veff = v_ctx[s] * (Q[2] - Q[1]) + gamma[s] * (cb1 - cb0);
      
      // Structural Bounds (Balthazar)
      // a_dyn restricted within empirical bounds of 0.11 to 7.47
      real a_raw = a_base_raw[s] + w_u[s] * abs(cb0) * abs(cb1);
      real a_dyn = 0.11 + 7.36 / (1.0 + exp(-a_raw));
      
      // CONTINUOUS ASYMPTOTIC SATURATION
      real final_veff = 18.51 * tanh(veff / 18.51);
      if (abs(final_veff) < 1e-4) final_veff = final_veff >= 0 ? 1e-4 : -1e-4;
      
      target += wiener_lpdf(rt[t] | a_dyn, tnd[s], 0.5, final_veff);
      
      prev_E = R - Q[ch];
      Q[ch] += alpha_ctx[s] * prev_E;
      prev_ch = ch;
      
      for (i in 1:32) {
        real err_sig = 0.0;
        if (prev_ch == 1 && i <= 16) err_sig = prev_E;
        if (prev_ch == 2 && i > 16) err_sig = prev_E;
        W_PC_latent[i] += alpha_pc[s] * Z[i] * err_sig;
      }
    }
  }
}
