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
  
  // Epistemic Priors
  matrix[9, 9] L_train;
  vector[9] sigma_train;
  vector[9] theta_train_mean;
}
parameters {
  vector[9] mu_raw;
  matrix[9, N_subj] z;
}
transformed parameters {
  matrix[9, N_subj] theta;
  for (s in 1:N_subj) {
    theta[, s] = mu_raw + diag_pre_multiply(sigma_train, L_train) * z[, s];
  }
  vector[N_subj] a_base_raw = theta[1, ]';
  vector[N_subj] tnd = to_vector(min_rt) .* (1.0 ./ (1.0 + exp(-theta[2, ]')));
  vector[N_subj] v_ctx = exp(theta[3, ]');
  vector[N_subj] alpha_ctx = 1.0 ./ (1.0 + exp(-theta[4, ]'));
  vector[N_subj] alpha_pc = 1.0 ./ (1.0 + exp(-theta[5, ]'));
  vector[N_subj] gamma = exp(theta[6, ]');
  vector[N_subj] golgi_scale = exp(theta[7, ]');
  vector[N_subj] tau_decay = exp(theta[8, ]');
  vector[N_subj] w_u = exp(theta[9, ]');
}
model {
  // Priors
  mu_raw ~ normal(0, 2);
  to_vector(z) ~ std_normal();
  
  vector[32] frac_alpha;
  vector[32] kappa_vec;
  for (i in 1:32) {
    frac_alpha[i] = 0.1 + 0.8 * ((i - 1) / 31.0);
    kappa_vec[i] = 0.1 + 0.89 * ((i - 1) / 31.0);
  }
  
  for (s in 1:N_subj) {
    // Local Subject Scopes (Melchior)
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
        
        // The Readout Bottleneck (Caspar)
        real W_PC_eff = 3.0 * tanh(W_PC_latent[i] / 3.0);
        real S_mask = tanh(golgi_scale[s] * abs(W_PC_eff * Z[i]));
        
        if (i <= 16) {
          cb0 += S_mask * W_PC_eff * Z[i];
        } else {
          cb1 += S_mask * W_PC_eff * Z[i];
        }
      }
      
      real veff = v_ctx[s] * (Q[2] - Q[1]) + gamma[s] * (cb1 - cb0);
      
      // Continuous Asymptotic Saturation
      real a_raw = a_base_raw[s] + w_u[s] * abs(cb0) * abs(cb1);
      real a_dyn = 0.1 + 4.9 / (1.0 + exp(-a_raw));
      
      target += wiener_lpdf(rt[t] | a_dyn, tnd[s], 0.5, veff);
      
      prev_E = R - Q[ch];
      Q[ch] += alpha_ctx[s] * prev_E;
      prev_ch = ch;
      
      for (i in 1:32) {
        real err_sig = 0.0;
        if (prev_ch == 1 && i <= 16) err_sig = prev_E;
        if (prev_ch == 2 && i > 16) err_sig = prev_E;
        // The Unbounded Update
        W_PC_latent[i] += alpha_pc[s] * Z[i] * err_sig;
      }
    }
  }
}
