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
                   array[] vector trial_mask,
                   matrix W_exp,
                   vector a_base_raw,
                   vector tnd,
                   vector v_ctx,
                   vector alpha_ctx,
                   vector alpha_pc,
                   vector gamma_var,
                   vector golgi_scale,
                   vector tau_decay,
                   vector w_u,
                   vector frac_alpha,
                   vector inv_frac_alpha,
                   vector kappa_vec) {
                     
    real pt = 0;
    
    for (idx in 1:size(slice_subj)) {
      int s = slice_subj[idx];
      
      vector[2] Q = rep_vector(0.5, 2);
      real Q_diff = 0.0;
      
      vector[32] frac_mem = rep_vector(0.0, 32);
      vector[32] Z = rep_vector(0.0, 32);
      vector[32] W_PC_latent = rep_vector(0.0, 32);
      
      vector[32] W_exp_s = to_vector(W_exp[s]); 
      vector[32] inv_W_exp = inv_frac_alpha .* W_exp_s;
      
      real v_ctx_s = v_ctx[s] * 0.0540248;
      real gamma_s = gamma_var[s] * 0.0540248;
      
      real a_base_raw_s = a_base_raw[s];
      real phys_a_base = 0.11 + 3.0 * inv_logit(a_base_raw_s);
      real delta_max = 1.0 / phys_a_base;
      
      real w_u_s = w_u[s];
      real tnd_s = tnd[s];
      real a_c_s = alpha_ctx[s];
      real a_pc_s = alpha_pc[s];
      
      real inv_tau = 1.0 / tau_decay[s];
      real double_g_s = 2.0 * golgi_scale[s];
      real double_w_u_s = 2.0 * w_u_s;
      
      int num_trials = end_idx[s] - start_idx[s] + 1;
      vector[num_trials] a_dyn_arr;
      vector[num_trials] veff_arr;
      
      int t_idx = 1;
      
      for (t in start_idx[s]:end_idx[s]) {
        int ch = resp[t];
        real R = reward[t];
        
        real phys_decay = exp(-clean_iti[t] * inv_tau);
        
        frac_mem = frac_alpha .* frac_mem + inv_W_exp * Q[ch];
        
        Z = phys_decay * (kappa_vec .* Z) + (2.0 * inv_logit(2.0 * frac_mem) - 1.0);
        
        vector[32] W_PC_eff = 3.0 * (2.0 * inv_logit(0.6666666666666666 * W_PC_latent) - 1.0);
        
        vector[32] eff_z = W_PC_eff .* Z;
        vector[32] abs_approx = sqrt((eff_z .* eff_z) + 1e-8);
        vector[32] S_mask = 2.0 * inv_logit(double_g_s * abs_approx) - 1.0;
        
        real cb0 = dot_product(S_mask[1:16], eff_z[1:16]);
        real cb1 = dot_product(S_mask[17:32], eff_z[17:32]);
        
        real veff_scaled = v_ctx_s * Q_diff + gamma_s * (cb0 - cb1);
        
        real veff_raw = 18.51 * (2.0 * inv_logit(2.0 * veff_scaled) - 1.0); 
        veff_arr[t_idx] = (ch == 1) ? veff_raw : -veff_raw;
        
        real U_epistemic = sqrt( (cb0 * cb0 + 1e-8) * (cb1 * cb1 + 1e-8) );
        a_dyn_arr[t_idx] = phys_a_base + delta_max * (2.0 * inv_logit(double_w_u_s * U_epistemic) - 1.0);
        
        real prev_E = R - Q[ch];
        real alpha_ctx_E = a_c_s * prev_E;
        Q[ch] += alpha_ctx_E;
        Q_diff += ch_sign[t] * alpha_ctx_E;
        
        W_PC_latent += (a_pc_s * prev_E) * (trial_mask[t] .* Z);
        
        t_idx += 1;
      }
      
      pt += wiener_lpdf(rt[start_idx[s]:end_idx[s]] | a_dyn_arr, tnd_s, 0.5, veff_arr);
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
  matrix[N_subj, 32] W_exp;
  
  array[N_subj] int<lower=1> start_idx;
  array[N_subj] int<lower=1> end_idx;
  
  vector[9] theta_mean;
  matrix[9, 9] L_Sigma;
}
transformed data {
  array[N_subj] int seq_subj;
  for (s in 1:N_subj) seq_subj[s] = s;
  
  vector[32] frac_alpha;
  vector[32] kappa_vec;
  for (i in 1:32) {
    frac_alpha[i] = 0.1 + 0.8 * ((i - 1) / 31.0);
    kappa_vec[i]  = 0.1 + 0.89 * ((i - 1) / 31.0);
  }
  vector[32] inv_frac_alpha = 1.0 - frac_alpha; 
  
  array[N_trials] real clean_iti;
  array[N_trials] real ch_sign;
  array[N_trials] vector[32] trial_mask;
  
  vector[32] MASK_1;
  vector[32] MASK_2;
  for(i in 1:16) { MASK_1[i] = 1.0; MASK_2[i] = 0.0; }
  for(i in 17:32) { MASK_1[i] = 0.0; MASK_2[i] = 1.0; }
  
  for (t in 1:N_trials) {
     clean_iti[t] = (iti[t] < 0) ? 1.0 : iti[t];
     ch_sign[t] = (resp[t] == 1) ? -1.0 : 1.0;
     trial_mask[t] = (resp[t] == 1) ? MASK_1 : MASK_2;
  }
}
parameters {
  vector[9] theta_raw;
  vector<lower=0>[9] sigma;
  matrix[9, N_subj] z;
}
transformed parameters {
  vector[9] theta_unc = theta_mean + L_Sigma * theta_raw;
  
  real mu_a_unc = theta_unc[1];
  real mu_tnd_unc = theta_unc[2];
  real mu_v_unc = theta_unc[3];
  
  real mu_a = 0.11 + 3.0 * inv_logit(mu_a_unc);
  real mu_tnd = 3.69 * inv_logit(mu_tnd_unc);
  real mu_v = 18.51 * inv_logit(mu_v_unc);
  
  vector[6] mu_res_raw = theta_unc[4:9];
  
  vector[N_subj] a_base_raw;
  vector[N_subj] tnd;
  vector[N_subj] v_ctx;
  vector[N_subj] alpha_ctx;
  vector[N_subj] alpha_pc;
  vector[N_subj] gamma_var;
  vector[N_subj] golgi_scale;
  vector[N_subj] tau_decay;
  vector[N_subj] w_u;
  
  for (s in 1:N_subj) {
    a_base_raw[s] = mu_a_unc + sigma[1] * z[1, s];
    real tnd_cap = fmin(min_rt[s] - 0.05, 3.69);
    tnd[s] = 0.01 + (tnd_cap - 0.01) * inv_logit(mu_tnd_unc + sigma[2] * z[2, s]);
    v_ctx[s] = 18.51 * inv_logit(mu_v_unc + sigma[3] * z[3, s]);
    
    alpha_ctx[s]   = inv_logit(mu_res_raw[1] + sigma[4] * z[4, s]);
    alpha_pc[s]    = inv_logit(mu_res_raw[2] + sigma[5] * z[5, s]);
    gamma_var[s]   = exp(mu_res_raw[3] + sigma[6] * z[6, s]);
    golgi_scale[s] = exp(mu_res_raw[4] + sigma[7] * z[7, s]);
    tau_decay[s]   = exp(mu_res_raw[5] + sigma[8] * z[8, s]);
    w_u[s]         = exp(mu_res_raw[6] + sigma[9] * z[9, s]);
  }
}
model {
  target += log(3.0) + log_inv_logit(mu_a_unc) + log1m_inv_logit(mu_a_unc);
  target += log(3.69) + log_inv_logit(mu_tnd_unc) + log1m_inv_logit(mu_tnd_unc);
  target += log(18.51) + log_inv_logit(mu_v_unc) + log1m_inv_logit(mu_v_unc);

  mu_a ~ gamma(11.69, 8.333);
  mu_tnd ~ student_t(1.32, 0.44, 0.08);
  mu_v ~ normal(1.76, 1.51);
  
  mu_res_raw ~ normal(0, 2);
  sigma ~ normal(0, 0.5);
  to_vector(z) ~ std_normal();

  int grainsize = 4;
  target += reduce_sum(partial_sum, seq_subj, grainsize, 
                       start_idx, end_idx, resp, reward, rt, clean_iti, ch_sign, trial_mask, W_exp, 
                       a_base_raw, tnd, v_ctx, alpha_ctx, alpha_pc, gamma_var, golgi_scale, tau_decay, w_u, 
                       frac_alpha, inv_frac_alpha, kappa_vec);
}