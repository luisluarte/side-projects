functions {
  real partial_sum(array[] int slice_subj,
                   int start, int end,
                   array[] int start_idx,
                   array[] int end_idx,
                   array[] int resp,
                   array[] real reward,
                   array[] real rt,
                   array[] real clean_iti,
                   array[] real ch_sign,
                   matrix W_exp,
                   vector a_base_raw,
                   vector tnd,
                   vector v_ctx,
                   vector w_bias_raw,
                   vector aw,
                   vector al,
                   vector alpha_pc,
                   vector tau_decay,
                   vector golgi_scale,
                   vector w_cb,
                   vector w_ctx,
                   vector beta_mismatch,
                   vector frac_alpha,
                   vector inv_frac_alpha,
                   vector kappa_vec) {
                     
    real pt = 0;
    
    for (idx in 1:size(slice_subj)) {
      int s = slice_subj[idx];
      vector[2] Q = rep_vector(0.5, 2);
      vector[4] frac_mem = rep_vector(0.0, 4);
      vector[4] Z = rep_vector(0.0, 4);
      vector[4] W_PC_latent = rep_vector(0.0, 4);
      vector[4] W_exp_s = to_vector(W_exp[s]); 
      vector[4] inv_W_exp = inv_frac_alpha .* W_exp_s;
      
      real v_s = v_ctx[s];
      real w_start = inv_logit(w_bias_raw[s]);
      real phys_a_base = 0.11 + 3.0 * inv_logit(a_base_raw[s]);
      real delta_max = 1.0 / phys_a_base;
      real tnd_s = tnd[s];
      real aw_s = aw[s];
      real al_s = al[s];
      real a_pc_s = alpha_pc[s];
      real inv_tau = 1.0 / tau_decay[s];
      real g_s = golgi_scale[s];
      real w_cb_s = w_cb[s];
      real w_ctx_s = w_ctx[s];
      real beta_mis_s = beta_mismatch[s];
      
      for (t in start_idx[s]:end_idx[s]) {
        int ch = resp[t];
        real R = reward[t];
        real phys_decay = exp(-clean_iti[t] * inv_tau);
        Q = 0.5 + (Q - 0.5) * phys_decay;
        real Q_diff = Q[1] - Q[2];
        
        frac_mem = frac_alpha .* frac_mem + inv_W_exp * Q[ch];
        Z = phys_decay * (kappa_vec .* Z) + tanh(frac_mem);
        
        vector[4] W_PC_eff = 3.0 * tanh(W_PC_latent * 0.3333333333333333);
        vector[4] eff_z = W_PC_eff .* Z;
        vector[4] abs_approx = sqrt((eff_z .* eff_z) + 1e-8);
        vector[4] S_mask = tanh(g_s * abs_approx);
        
        real cb0 = dot_product(S_mask[1:2], eff_z[1:2]);
        real cb1 = dot_product(S_mask[3:4], eff_z[3:4]);
        
        real Cb_diff = cb0 - cb1; 
        real M_align = tanh(w_cb_s * Cb_diff) * tanh(w_ctx_s * Q_diff);
        real caution = log1p_exp(-10.0 * M_align) * 0.1;
        
        real a_dyn = phys_a_base + delta_max * tanh(beta_mis_s * caution);
        
        if (t > start_idx[s]) {
           int prev_ch = resp[t-1];
           int is_switch = (ch != prev_ch) ? 1 : 0;
           real Q_switch = Q[3 - prev_ch]; 
           real Q_stay = Q[prev_ch];
           real veff_raw = v_s * (Q_switch - Q_stay);
           real veff = (is_switch == 1) ? veff_raw : -veff_raw;
           real w_eff = (is_switch == 1) ? w_start : (1.0 - w_start);
           pt += wiener_lpdf(rt[t] | a_dyn, tnd_s, w_eff, veff);
        }
        
        real pe = R - Q[ch];
        real alpha_eff = (pe > 0) ? aw_s : al_s;
        Q[ch] += alpha_eff * pe;
        
        real alpha_E = a_pc_s * pe;
        if (ch == 1) { W_PC_latent[1:2] += alpha_E * Z[1:2]; } 
        else { W_PC_latent[3:4] += alpha_E * Z[3:4]; }
      }
    }
    return pt;
  }
}
data {
  int<lower=1> N_trials;
  int<lower=1> N_subj;
  array[N_trials] int<lower=1, upper=N_subj> subj;
  array[N_trials] int<lower=1, upper=2> resp;
  array[N_trials] real reward;
  array[N_trials] real rt;
  array[N_trials] real iti;
  array[N_subj] real min_rt;
  matrix[N_subj, 4] W_exp;
  
  array[N_subj] int<lower=1> start_idx;
  array[N_subj] int<lower=1> end_idx;
  
  vector[12] theta_mean;
  matrix[12, 12] L_Sigma;
  
  int grainsize; 
}
transformed data {
  array[N_subj] int seq_subj;
  for (s in 1:N_subj) seq_subj[s] = s;
  
  vector[4] frac_alpha;
  vector[4] kappa_vec;
  for (i in 1:4) {
    frac_alpha[i] = 0.1 + 0.8 * ((i - 1) / 3.0);
    kappa_vec[i]  = 0.1 + 0.89 * ((i - 1) / 3.0);
  }
  vector[4] inv_frac_alpha = 1.0 - frac_alpha; 
  
  array[N_trials] real clean_iti;
  array[N_trials] real ch_sign;
  for (t in 1:N_trials) {
     clean_iti[t] = (iti[t] < 0) ? 1.0 : iti[t];
     ch_sign[t] = (resp[t] == 1) ? -1.0 : 1.0;
  }
}
parameters {
  vector[12] theta_raw;
  vector<lower=0>[12] sigma;
  matrix[12, N_subj] z;
}
transformed parameters {
  vector[12] theta_unc = theta_mean + L_Sigma * theta_raw;
  real mu_a_unc = theta_unc[1];
  real mu_tnd_unc = theta_unc[2];
  real mu_v_unc = theta_unc[3];
  vector[8] mu_res_raw = theta_unc[4:11];
  
  vector[N_subj] a_base_raw;
  vector[N_subj] tnd;
  vector[N_subj] v_ctx;
  vector[N_subj] w_bias_raw;
  vector[N_subj] aw;
  vector[N_subj] al;
  vector[N_subj] alpha_pc;
  vector[N_subj] tau_decay;
  vector[N_subj] golgi_scale;
  vector[N_subj] w_cb;
  vector[N_subj] w_ctx;
  vector[N_subj] beta_mismatch;
  
  for (s in 1:N_subj) {
    a_base_raw[s] = mu_a_unc + sigma[1] * z[1, s];
    real tnd_cap = fmin(min_rt[s] - 0.05, 3.69);
    tnd[s] = 0.01 + (tnd_cap - 0.01) * inv_logit(mu_tnd_unc + sigma[2] * z[2, s]);
    v_ctx[s] = 18.51 * inv_logit(mu_v_unc + sigma[3] * z[3, s]);
    w_bias_raw[s] = theta_unc[12] + sigma[12] * z[12, s];
    
    aw[s]            = inv_logit(mu_res_raw[1] + sigma[4] * z[4, s]);
    al[s]            = inv_logit(mu_res_raw[2] + sigma[5] * z[5, s]);
    alpha_pc[s]      = inv_logit(mu_res_raw[3] + sigma[6] * z[6, s]);
    
    tau_decay[s]     = log1p_exp(mu_res_raw[4] + sigma[7] * z[7, s]);
    golgi_scale[s]   = log1p_exp(mu_res_raw[5] + sigma[8] * z[8, s]);
    w_cb[s]          = log1p_exp(mu_res_raw[6] + sigma[9] * z[9, s]);
    w_ctx[s]         = log1p_exp(mu_res_raw[7] + sigma[10] * z[10, s]);
    beta_mismatch[s] = log1p_exp(mu_res_raw[8] + sigma[11] * z[11, s]);
  }
}
model {
  target += log(3.0) + log_inv_logit(mu_a_unc) + log1m_inv_logit(mu_a_unc);
  target += log(3.69) + log_inv_logit(mu_tnd_unc) + log1m_inv_logit(mu_tnd_unc);
  target += log(18.51) + log_inv_logit(mu_v_unc) + log1m_inv_logit(mu_v_unc);
  
  theta_unc[1] ~ normal(0, 2);
  theta_unc[2] ~ normal(0, 2);
  theta_unc[3] ~ normal(0, 2);
  mu_res_raw ~ normal(0, 2);
  sigma ~ gamma(2, 5);
  to_vector(z) ~ std_normal();

  target += reduce_sum(partial_sum, seq_subj, grainsize, 
                       start_idx, end_idx, resp, reward, rt, clean_iti, ch_sign, W_exp, 
                       a_base_raw, tnd, v_ctx, w_bias_raw, aw, al, alpha_pc, tau_decay, golgi_scale, w_cb, w_ctx, beta_mismatch, 
                       frac_alpha, inv_frac_alpha, kappa_vec);
}
generated quantities {
  vector[N_trials] log_lik;
  for (t in 1:N_trials) log_lik[t] = 0.0;
  
  {
    for (s in 1:N_subj) {
      vector[2] Q = rep_vector(0.5, 2);
      vector[4] frac_mem = rep_vector(0.0, 4);
      vector[4] Z = rep_vector(0.0, 4);
      vector[4] W_PC_latent = rep_vector(0.0, 4);
      vector[4] W_exp_s = to_vector(W_exp[s]);
      vector[4] inv_W_exp = inv_frac_alpha .* W_exp_s;
      
      real v_s = v_ctx[s];
      real w_start = inv_logit(w_bias_raw[s]);
      real phys_a_base = 0.11 + 3.0 * inv_logit(a_base_raw[s]);
      real delta_max = 1.0 / phys_a_base;
      real tnd_s = tnd[s];
      real aw_s = aw[s];
      real al_s = al[s];
      real a_pc_s = alpha_pc[s];
      real inv_tau = 1.0 / tau_decay[s];
      real g_s = golgi_scale[s];
      real w_cb_s = w_cb[s];
      real w_ctx_s = w_ctx[s];
      real beta_mis_s = beta_mismatch[s];
      
      for (t in start_idx[s]:end_idx[s]) {
        int ch = resp[t];
        real R = reward[t];
        real phys_decay = exp(-clean_iti[t] * inv_tau);
        Q = 0.5 + (Q - 0.5) * phys_decay;
        real Q_diff = Q[1] - Q[2];
        
        frac_mem = frac_alpha .* frac_mem + inv_W_exp * Q[ch];
        Z = phys_decay * (kappa_vec .* Z) + tanh(frac_mem);
        
        vector[4] W_PC_eff = 3.0 * tanh(W_PC_latent * 0.3333333333333333);
        vector[4] eff_z = W_PC_eff .* Z;
        vector[4] abs_approx = sqrt((eff_z .* eff_z) + 1e-8);
        vector[4] S_mask = tanh(g_s * abs_approx);
        
        real cb0 = dot_product(S_mask[1:2], eff_z[1:2]);
        real cb1 = dot_product(S_mask[3:4], eff_z[3:4]);
        
        real Cb_diff = cb0 - cb1;
        real M_align = tanh(w_cb_s * Cb_diff) * tanh(w_ctx_s * Q_diff);
        real caution = log1p_exp(-10.0 * M_align) * 0.1;
        
        real a_dyn = phys_a_base + delta_max * tanh(beta_mis_s * caution);
        
        if (t > start_idx[s]) {
           int prev_ch = resp[t-1];
           int is_switch = (ch != prev_ch) ? 1 : 0;
           real Q_switch = Q[3 - prev_ch]; 
           real Q_stay = Q[prev_ch];
           real veff_raw = v_s * (Q_switch - Q_stay);
           real veff = (is_switch == 1) ? veff_raw : -veff_raw;
           real w_eff = (is_switch == 1) ? w_start : (1.0 - w_start);
           log_lik[t] = wiener_lpdf(rt[t] | a_dyn, tnd_s, w_eff, veff);
        }
        
        real pe = R - Q[ch];
        real alpha_eff = (pe > 0) ? aw_s : al_s;
        Q[ch] += alpha_eff * pe;
        
        real alpha_E = a_pc_s * pe;
        if (ch == 1) { W_PC_latent[1:2] += alpha_E * Z[1:2]; }
        else { W_PC_latent[3:4] += alpha_E * Z[3:4]; }
      }
    }
  }
}

