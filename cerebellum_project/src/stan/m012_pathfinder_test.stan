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
                   vector a_base,
                   vector tnd,
                   vector v_ctx,
                   vector alpha_ctx,
                   vector alpha_pc,
                   vector gamma_var,
                   vector golgi_scale,
                   vector tau_decay,
                   vector w_cb,
                   vector w_ctx,
                   vector beta_mis,
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
      
      real phys_a_base = 0.11 + a_base[s];
      real delta_max = 1.0 / phys_a_base;
      
      real tnd_s = tnd[s];
      real a_c_s = alpha_ctx[s];
      real a_pc_s = alpha_pc[s];
      real w_cb_s = w_cb[s];
      real w_ctx_s = w_ctx[s];
      real beta_mis_s = beta_mis[s];
      
      real inv_tau = 1.0 / tau_decay[s];
      real g_s = golgi_scale[s];
      
      int num_trials = end_idx[s] - start_idx[s] + 1;
      vector[num_trials] a_dyn_arr;
      vector[num_trials] veff_arr;
      
      int t_idx = 1;
      
      for (t in start_idx[s]:end_idx[s]) {
        int ch = resp[t];
        real R = reward[t];
        
        real phys_decay = exp(-clean_iti[t] * inv_tau);
        
        frac_mem = frac_alpha .* frac_mem + inv_W_exp * Q[ch];
        Z = phys_decay * (kappa_vec .* Z) + tanh(frac_mem);
        
        vector[32] W_PC_eff = 3.0 * tanh(W_PC_latent * 0.3333333333333333);
        vector[32] eff_z = W_PC_eff .* Z;
        vector[32] abs_approx = sqrt((eff_z .* eff_z) + 1e-8);
        vector[32] S_mask = tanh(g_s * abs_approx);
        
        real cb0 = dot_product(S_mask[1:16], eff_z[1:16]);
        real cb1 = dot_product(S_mask[17:32], eff_z[17:32]);
        
        real Cb_diff = cb0 - cb1;
        
        real veff_scaled = v_ctx_s * Q_diff + gamma_s * Cb_diff;
        real veff_raw = 18.51 * tanh(veff_scaled); 
        veff_arr[t_idx] = (ch == 1) ? veff_raw : -veff_raw;
        
        real M_align = tanh(w_cb_s * Cb_diff) * tanh(w_ctx_s * Q_diff);
        real caution = log1p_exp(-2.0 * M_align) * 0.1;
        a_dyn_arr[t_idx] = phys_a_base + delta_max * tanh(beta_mis_s * caution);
        
        real prev_E = R - Q[ch];
        real alpha_ctx_E = a_c_s * prev_E;
        Q[ch] += alpha_ctx_E;
        Q_diff += ch_sign[t] * alpha_ctx_E;
        
        real alpha_E = a_pc_s * prev_E;
        if (ch == 1) {
            W_PC_latent[1:16] += alpha_E * Z[1:16];
        } else {
            W_PC_latent[17:32] += alpha_E * Z[17:32];
        }
        
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
  
  vector[11] theta_mean;
  matrix[11, 11] L_Sigma;
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
  for (t in 1:N_trials) {
     clean_iti[t] = (iti[t] < 0) ? 1.0 : iti[t];
     ch_sign[t] = (resp[t] == 1) ? -1.0 : 1.0;
  }
}
parameters {
  vector<lower=0>[11] sigma;
  
  vector<lower=0, upper=3.0>[N_subj] a_base;
  vector<lower=0.01, upper=3.69>[N_subj] tnd;
  vector<lower=0, upper=18.51>[N_subj] v_ctx;
  vector<lower=0, upper=1.0>[N_subj] alpha_ctx;
  vector<lower=0, upper=1.0>[N_subj] alpha_pc;
  vector<lower=0, upper=50.0>[N_subj] gamma_var;
  vector<lower=0, upper=50.0>[N_subj] golgi_scale;
  vector<lower=0, upper=50.0>[N_subj] tau_decay;
  vector<lower=0, upper=50.0>[N_subj] w_cb;
  vector<lower=0, upper=50.0>[N_subj] w_ctx;
  vector<lower=0, upper=50.0>[N_subj] beta_mis;
}
model {
  sigma ~ normal(0, 0.5);
  
  a_base ~ normal(1.5, sigma[1]);
  tnd ~ normal(0.3, sigma[2]);
  v_ctx ~ normal(10.0, sigma[3]);
  alpha_ctx ~ normal(0.5, sigma[4]);
  alpha_pc ~ normal(0.5, sigma[5]);
  gamma_var ~ normal(5.0, sigma[6]);
  golgi_scale ~ normal(5.0, sigma[7]);
  tau_decay ~ normal(5.0, sigma[8]);
  w_cb ~ normal(5.0, sigma[9]);
  w_ctx ~ normal(5.0, sigma[10]);
  beta_mis ~ normal(5.0, sigma[11]);

  int grainsize = 4;
  target += reduce_sum(partial_sum, seq_subj, grainsize, 
                       start_idx, end_idx, resp, reward, rt, clean_iti, ch_sign, W_exp, 
                       a_base, tnd, v_ctx, alpha_ctx, alpha_pc, gamma_var, golgi_scale, tau_decay, 
                       w_cb, w_ctx, beta_mis,
                       frac_alpha, inv_frac_alpha, kappa_vec);
}
