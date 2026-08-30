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
  vector[9] mu_raw;
  vector<lower=0>[9] sigma;
  matrix[9, N_subj] z;
}
transformed parameters {
  matrix[9, N_subj] theta;
  for (s in 1:N_subj) {
    theta[, s] = mu_raw + sigma .* z[, s];
  }
  vector[N_subj] a_base = exp(theta[1, ]');
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
  mu_raw ~ normal(0, 1);
  sigma ~ normal(0, 0.5);
  to_vector(z) ~ std_normal();
  
  vector[32] frac_alpha;
  vector[32] kappa_vec;
  for (i in 1:32) {
    frac_alpha[i] = 0.1 + 0.8 * ((i - 1) / 31.0);
    kappa_vec[i] = 0.1 + 0.89 * ((i - 1) / 31.0);
  }
  
  matrix[N_subj, 2] Q = rep_matrix(0.5, N_subj, 2);
  matrix[N_subj, 32] frac_mem = rep_matrix(0.0, N_subj, 32);
  matrix[N_subj, 32] Z = rep_matrix(0.0, N_subj, 32);
  matrix[N_subj, 32] W_PC = rep_matrix(0.0, N_subj, 32);
  array[N_subj] int prev_ch = rep_array(1, N_subj);
  vector[N_subj] prev_E = rep_vector(0.0, N_subj);
  
  matrix[N_subj, 32] W_exp;
  for (s in 1:N_subj) {
    for (i in 1:32) W_exp[s, i] = sin(s * i * 0.1); 
  }
  
  for (t in 1:N_trials) {
    int s = subj[t];
    int ch = resp[t];
    real R = reward[t];
    real current_iti = (iti[t] < 0) ? 1.0 : iti[t];
    real phys_decay = exp(-current_iti / tau_decay[s]);
    
    vector[32] S_mask;
    for (i in 1:32) S_mask[i] = tanh(golgi_scale[s] * abs(W_PC[s, i] * Z[s, i]));
    
    real cb0 = 0.0;
    real cb1 = 0.0;
    for (i in 1:32) {
      frac_mem[s, i] = frac_alpha[i] * frac_mem[s, i] + (1.0 - frac_alpha[i]) * W_exp[s, i] * Q[s, ch];
      Z[s, i] = phys_decay * kappa_vec[i] * Z[s, i] + tanh(frac_mem[s, i]);
      if (i <= 16) cb0 += S_mask[i] * W_PC[s, i] * Z[s, i];
      else cb1 += S_mask[i] * W_PC[s, i] * Z[s, i];
    }
    
    real veff = v_ctx[s] * (Q[s, 2] - Q[s, 1]) + gamma[s] * (cb1 - cb0);
    if (abs(veff) < 1e-4) veff = veff >= 0 ? 1e-4 : -1e-4;
    
    real a_dyn = a_base[s] + w_u[s] * abs(cb0) * abs(cb1);
    if (a_dyn > 5.0) a_dyn = 5.0; 
    if (a_dyn < 0.1) a_dyn = 0.1;
    
    target += wiener_lpdf(rt[t] | a_dyn, tnd[s], 0.5, veff);
    
    prev_E[s] = R - Q[s, ch];
    Q[s, ch] += alpha_ctx[s] * prev_E[s];
    prev_ch[s] = ch;
    
    for (i in 1:32) {
      real err_sig = 0.0;
      if (prev_ch[s] == 1 && i <= 16) err_sig = prev_E[s];
      if (prev_ch[s] == 2 && i > 16) err_sig = prev_E[s];
      W_PC[s, i] += alpha_pc[s] * Z[s, i] * err_sig;
      if (W_PC[s, i] > 3.0) W_PC[s, i] = 3.0;
      if (W_PC[s, i] < -3.0) W_PC[s, i] = -3.0;
    }
  }
}
generated quantities {
  array[N_trials] real log_lik;
  vector[32] frac_alpha;
  vector[32] kappa_vec;
  for (i in 1:32) {
    frac_alpha[i] = 0.1 + 0.8 * ((i - 1) / 31.0);
    kappa_vec[i] = 0.1 + 0.89 * ((i - 1) / 31.0);
  }
  
  matrix[N_subj, 2] Q = rep_matrix(0.5, N_subj, 2);
  matrix[N_subj, 32] frac_mem = rep_matrix(0.0, N_subj, 32);
  matrix[N_subj, 32] Z = rep_matrix(0.0, N_subj, 32);
  matrix[N_subj, 32] W_PC = rep_matrix(0.0, N_subj, 32);
  array[N_subj] int prev_ch = rep_array(1, N_subj);
  vector[N_subj] prev_E = rep_vector(0.0, N_subj);
  
  matrix[N_subj, 32] W_exp;
  for (s in 1:N_subj) {
    for (i in 1:32) W_exp[s, i] = sin(s * i * 0.1); 
  }
  
  for (t in 1:N_trials) {
    int s = subj[t];
    int ch = resp[t];
    real R = reward[t];
    real current_iti = (iti[t] < 0) ? 1.0 : iti[t];
    real phys_decay = exp(-current_iti / tau_decay[s]);
    
    vector[32] S_mask;
    for (i in 1:32) S_mask[i] = tanh(golgi_scale[s] * abs(W_PC[s, i] * Z[s, i]));
    
    real cb0 = 0.0;
    real cb1 = 0.0;
    for (i in 1:32) {
      frac_mem[s, i] = frac_alpha[i] * frac_mem[s, i] + (1.0 - frac_alpha[i]) * W_exp[s, i] * Q[s, ch];
      Z[s, i] = phys_decay * kappa_vec[i] * Z[s, i] + tanh(frac_mem[s, i]);
      if (i <= 16) cb0 += S_mask[i] * W_PC[s, i] * Z[s, i];
      else cb1 += S_mask[i] * W_PC[s, i] * Z[s, i];
    }
    
    real veff = v_ctx[s] * (Q[s, 2] - Q[s, 1]) + gamma[s] * (cb1 - cb0);
    if (abs(veff) < 1e-4) veff = veff >= 0 ? 1e-4 : -1e-4;
    
    real a_dyn = a_base[s] + w_u[s] * abs(cb0) * abs(cb1);
    if (a_dyn > 5.0) a_dyn = 5.0; 
    if (a_dyn < 0.1) a_dyn = 0.1;
    
    log_lik[t] = wiener_lpdf(rt[t] | a_dyn, tnd[s], 0.5, veff);
    
    prev_E[s] = R - Q[s, ch];
    Q[s, ch] += alpha_ctx[s] * prev_E[s];
    prev_ch[s] = ch;
    
    for (i in 1:32) {
      real err_sig = 0.0;
      if (prev_ch[s] == 1 && i <= 16) err_sig = prev_E[s];
      if (prev_ch[s] == 2 && i > 16) err_sig = prev_E[s];
      W_PC[s, i] += alpha_pc[s] * Z[s, i] * err_sig;
      if (W_PC[s, i] > 3.0) W_PC[s, i] = 3.0;
      if (W_PC[s, i] < -3.0) W_PC[s, i] = -3.0;
    }
  }
}
