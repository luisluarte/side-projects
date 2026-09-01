functions {
  real partial_sum(array[] int slice_t_idx, int start, int end, array[] int subj, array[] int resp, array[] real reward, array[] real rt, array[] real clean_iti, matrix W_exp, vector a_base, vector tnd, vector v_ctx, vector alpha_win, vector alpha_loss, vector alpha_pc, vector gamma_var, vector golgi_scale, vector tau_decay, vector beta_conf) {
    real pt = 0.0;
    int current_s = -1;
    vector[2] Q;
    vector[4] frac_mem;
    vector[4] Z;
    vector[4] W_PC_latent;
    
    real tnd_s; real a_s; real v_s; real aw_s; real al_s; real apc_s; real gam_s; real golgi_s; real inv_tau_s; real b_conf_s;
    
    vector[4] frac_alpha;
    vector[4] kappa_vec;
    for (i in 1:4) {
      frac_alpha[i] = 0.1 + 0.8 * ((i - 1) / 3.0);
      kappa_vec[i]  = 0.1 + 0.89 * ((i - 1) / 3.0);
    }
    vector[4] inv_frac_alpha = 1.0 - frac_alpha;
    vector[4] W_exp_s; vector[4] inv_W_exp;
    real scale_factor = 1.0 / sqrt(4.0);
    
    for (i in start:end) {
      int s = subj[i];
      if (s != current_s) {
        current_s = s;
        Q = rep_vector(0.5, 2);
        frac_mem = rep_vector(0.0, 4); Z = rep_vector(0.0, 4); W_PC_latent = rep_vector(0.0, 4);
        
        tnd_s = tnd[s]; a_s = a_base[s]; v_s = v_ctx[s] * 0.0540248; aw_s = alpha_win[s]; al_s = alpha_loss[s];
        apc_s = alpha_pc[s]; gam_s = gamma_var[s] * 0.0540248; golgi_s = golgi_scale[s];
        inv_tau_s = 1.0 / tau_decay[s]; b_conf_s = beta_conf[s];
        
        W_exp_s = to_vector(W_exp[s]); inv_W_exp = inv_frac_alpha .* W_exp_s;
      }
      
      int ch = resp[i]; real r = reward[i];
      real phys_decay = exp(-clean_iti[i] * inv_tau_s);
      Q = 0.5 + (Q - 0.5) * phys_decay;
      
      frac_mem = frac_alpha .* frac_mem + inv_W_exp * Q[ch];
      Z = phys_decay * (kappa_vec .* Z) + tanh(frac_mem);
      
      vector[4] eff_z = (3.0 * tanh(W_PC_latent * scale_factor)) .* Z;
      vector[4] S_mask = tanh(golgi_s * sqrt((eff_z .* eff_z) + 1e-8));
      
      real cb0 = dot_product(S_mask[1:2], eff_z[1:2]);
      real cb1 = dot_product(S_mask[3:4], eff_z[3:4]);
      
      real U_epistemic = sqrt((cb0 * cb0 + 1e-8) * (cb1 * cb1 + 1e-8));
      real a_dyn = a_s + b_conf_s * tanh(U_epistemic);
      
      real veff_raw = 18.51 * tanh(v_s * (Q[1] - Q[2]) + gam_s * (cb0 - cb1));
      real veff = (ch == 1) ? veff_raw : -veff_raw;
      
      pt += wiener_lpdf(rt[i] | a_dyn, tnd_s, 0.5, veff);
      
      real pe = r - Q[ch];
      Q[ch] += ((pe > 0) ? aw_s : al_s) * pe;
      
      real alpha_E = apc_s * pe;
      if (ch == 1) { W_PC_latent[1:2] += alpha_E * Z[1:2]; } 
      else { W_PC_latent[3:4] += alpha_E * Z[3:4]; }
    }
    return pt;
  }
}
data {
  int<lower=1> N_trials; int<lower=1> N_subj;
  array[N_trials] int subj; array[N_trials] int resp;
  array[N_trials] real reward; array[N_trials] real rt; array[N_trials] real iti;
  array[N_subj] real min_rt; matrix[N_subj, 4] W_exp;
}
transformed data {
  array[N_trials] int t_idx; array[N_trials] real clean_iti;
  for (t in 1:N_trials) { t_idx[t] = t; clean_iti[t] = (iti[t] < 0) ? 1.0 : iti[t]; }
}
parameters {
  vector[10] mu_raw; vector<lower=0>[10] sigma; matrix[10, N_subj] z;
}
transformed parameters {
  vector[N_subj] a_base = 0.1 + 3.0 * inv_logit(mu_raw[1] + sigma[1] * z[1]');
  vector[N_subj] tnd;
  vector[N_subj] v_ctx = 18.51 * inv_logit(mu_raw[3] + sigma[3] * z[3]');
  vector[N_subj] alpha_win = inv_logit(mu_raw[4] + sigma[4] * z[4]');
  vector[N_subj] alpha_loss = inv_logit(mu_raw[5] + sigma[5] * z[5]');
  vector[N_subj] alpha_pc = inv_logit(mu_raw[6] + sigma[6] * z[6]');
  vector[N_subj] gamma_var = exp(mu_raw[7] + sigma[7] * z[7]');
  vector[N_subj] golgi_scale = exp(mu_raw[8] + sigma[8] * z[8]');
  vector[N_subj] tau_decay = 10.0 * inv_logit(mu_raw[9] + sigma[9] * z[9]');
  vector[N_subj] beta_conf = 10.0 * inv_logit(mu_raw[10] + sigma[10] * z[10]');
  for (s in 1:N_subj) {
    tnd[s] = 0.01 + (fmin(min_rt[s] - 0.05, 3.69) - 0.01) * inv_logit(mu_raw[2] + sigma[2] * z[2, s]);
  }
}
model {
  mu_raw ~ normal(0, 1); sigma ~ normal(0, 0.5); to_vector(z) ~ std_normal();
  target += reduce_sum(partial_sum, t_idx, 1, subj, resp, reward, rt, clean_iti, W_exp, a_base, tnd, v_ctx, alpha_win, alpha_loss, alpha_pc, gamma_var, golgi_scale, tau_decay, beta_conf);
}
generated quantities {
  vector[N_trials] log_lik;
  {
    int current_s = -1; vector[2] Q; vector[4] frac_mem; vector[4] Z; vector[4] W_PC_latent;
    vector[4] frac_alpha; vector[4] kappa_vec;
    for (i in 1:4) { frac_alpha[i] = 0.1 + 0.8 * ((i - 1) / 3.0); kappa_vec[i]  = 0.1 + 0.89 * ((i - 1) / 3.0); }
    vector[4] inv_frac_alpha = 1.0 - frac_alpha; vector[4] W_exp_s; vector[4] inv_W_exp;
    real scale_factor = 1.0 / sqrt(4.0);
    
    for (i in 1:N_trials) {
      int s = subj[i];
      if (s != current_s) {
        current_s = s; Q = rep_vector(0.5, 2); frac_mem = rep_vector(0.0, 4); Z = rep_vector(0.0, 4); W_PC_latent = rep_vector(0.0, 4);
        W_exp_s = to_vector(W_exp[s]); inv_W_exp = inv_frac_alpha .* W_exp_s;
      }
      int ch = resp[i]; real r = reward[i];
      real phys_decay = exp(-clean_iti[i] * (1.0 / tau_decay[s]));
      Q = 0.5 + (Q - 0.5) * phys_decay;
      frac_mem = frac_alpha .* frac_mem + inv_W_exp * Q[ch];
      Z = phys_decay * (kappa_vec .* Z) + tanh(frac_mem);
      
      vector[4] eff_z = (3.0 * tanh(W_PC_latent * scale_factor)) .* Z;
      vector[4] S_mask = tanh(golgi_scale[s] * sqrt((eff_z .* eff_z) + 1e-8));
      real cb0 = dot_product(S_mask[1:2], eff_z[1:2]); real cb1 = dot_product(S_mask[3:4], eff_z[3:4]);
      
      real U_epistemic = sqrt((cb0 * cb0 + 1e-8) * (cb1 * cb1 + 1e-8));
      real a_dyn = a_base[s] + beta_conf[s] * tanh(U_epistemic);
      
      real veff_raw = 18.51 * tanh((v_ctx[s] * 0.0540248) * (Q[1] - Q[2]) + (gamma_var[s] * 0.0540248) * (cb0 - cb1));
      real veff = (ch == 1) ? veff_raw : -veff_raw;
      
      log_lik[i] = wiener_lpdf(rt[i] | a_dyn, tnd[s], 0.5, veff);
      
      real pe = r - Q[ch];
      Q[ch] += ((pe > 0) ? alpha_win[s] : alpha_loss[s]) * pe;
      real alpha_E = alpha_pc[s] * pe;
      if (ch == 1) { W_PC_latent[1:2] += alpha_E * Z[1:2]; } 
      else { W_PC_latent[3:4] += alpha_E * Z[3:4]; }
    }
  }
}