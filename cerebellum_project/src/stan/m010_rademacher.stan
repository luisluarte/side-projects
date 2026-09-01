functions {
  // We need a dummy to satisfy Stan's syntax for partial_sum if we were doing MCMC, but we are doing optimize
}
data {
  int<lower=1> N_trials;
  array[N_trials] int<lower=1, upper=2> resp;
  array[N_trials] real reward;
  array[N_trials] real rt;
  array[N_trials] real iti;
  real min_rt;
  int<lower=2> D; // reservoir dimension
  matrix[1, D] W_exp; // W_exp for this subject
}
transformed data {
  vector[D] frac_alpha;
  vector[D] kappa_vec;
  for (i in 1:D) {
    frac_alpha[i] = 0.1 + 0.8 * ((i - 1) / (D - 1.0));
    kappa_vec[i]  = 0.1 + 0.89 * ((i - 1) / (D - 1.0));
  }
  vector[D] inv_frac_alpha = 1.0 - frac_alpha; 
  
  array[N_trials] real clean_iti;
  array[N_trials] real ch_sign;
  for (t in 1:N_trials) {
     clean_iti[t] = (iti[t] < 0) ? 1.0 : iti[t];
     ch_sign[t] = (resp[t] == 1) ? -1.0 : 1.0;
  }
  
  int half_D = D / 2;
}
parameters {
  real a_base_raw;
  real tnd_raw;
  real v_ctx_raw;
  real alpha_win_raw;
  real alpha_loss_raw;
  real alpha_pc_raw;
  real gamma_var_raw;
  real golgi_scale_raw;
  real tau_decay_raw;
  real beta_conf_raw;
  real w_mix_raw;
}
transformed parameters {
  real a_base = 0.1 + 3.0 * inv_logit(a_base_raw);
  real tnd_cap = fmin(min_rt - 0.05, 3.69);
  real tnd = 0.01 + (tnd_cap - 0.01) * inv_logit(tnd_raw);
  real v_ctx = 18.51 * inv_logit(v_ctx_raw);
  real alpha_win = inv_logit(alpha_win_raw);
  real alpha_loss = inv_logit(alpha_loss_raw);
  real alpha_pc = inv_logit(alpha_pc_raw);
  real gamma_var = exp(gamma_var_raw);
  real golgi_scale = exp(golgi_scale_raw);
  real tau_decay = 10.0 * inv_logit(tau_decay_raw);
  real beta_conf = 10.0 * inv_logit(beta_conf_raw);
  real w_mix = inv_logit(w_mix_raw);
}
model {
  // weak priors to regularize L-BFGS
  a_base_raw ~ normal(0, 2);
  tnd_raw ~ normal(0, 2);
  v_ctx_raw ~ normal(0, 2);
  alpha_win_raw ~ normal(0, 2);
  alpha_loss_raw ~ normal(0, 2);
  alpha_pc_raw ~ normal(0, 2);
  gamma_var_raw ~ normal(0, 2);
  golgi_scale_raw ~ normal(0, 2);
  tau_decay_raw ~ normal(0, 2);
  beta_conf_raw ~ normal(0, 2);
  w_mix_raw ~ normal(0, 2);
  
  vector[2] Q = rep_vector(0.5, 2);
  vector[D] frac_mem = rep_vector(0.0, D);
  vector[D] Z = rep_vector(0.0, D);
  vector[D] W_PC_latent = rep_vector(0.0, D);
  
  vector[D] W_exp_s = to_vector(W_exp[1]); 
  vector[D] inv_W_exp = inv_frac_alpha .* W_exp_s;
  
  real v_ctx_s = v_ctx * 0.0540248;
  real gamma_s = gamma_var * 0.0540248;
  real inv_tau = 1.0 / tau_decay;
  real g_s = golgi_scale;
  real scale_factor = 1.0 / sqrt(D * 1.0);
  
  for (t in 1:N_trials) {
    int ch = resp[t];
    real R = reward[t];
    
    real phys_decay = exp(-clean_iti[t] * inv_tau);
    Q = 0.5 + (Q - 0.5) * phys_decay;
    
    frac_mem = frac_alpha .* frac_mem + inv_W_exp * Q[ch];
    Z = phys_decay * (kappa_vec .* Z) + tanh(frac_mem);
    
    vector[D] W_PC_eff = 3.0 * tanh(W_PC_latent * scale_factor);
    vector[D] eff_z = W_PC_eff .* Z;
    vector[D] abs_approx = sqrt((eff_z .* eff_z) + 1e-8);
    vector[D] S_mask = tanh(g_s * abs_approx);
    
    real cb0 = dot_product(S_mask[1:half_D], eff_z[1:half_D]);
    real cb1 = dot_product(S_mask[(half_D+1):D], eff_z[(half_D+1):D]);
    
    // Cerebellar Conflict
    real U_epistemic = sqrt((cb0 * cb0 + 1e-8) * (cb1 * cb1 + 1e-8));
    
    // Cortical Conflict
    real Q_diff = Q[1] - Q[2];
    real conflict_cortex = 1.0 - abs(Q_diff);
    
    // Convex Mixture
    real final_conflict = w_mix * conflict_cortex + (1.0 - w_mix) * tanh(U_epistemic);
    real a_dyn = a_base + beta_conf * final_conflict;
    
    real veff_scaled = v_ctx_s * Q_diff + gamma_s * (cb0 - cb1);
    real veff_raw = 18.51 * tanh(veff_scaled); 
    real veff = (ch == 1) ? veff_raw : -veff_raw;
    
    target += wiener_lpdf(rt[t] | a_dyn, tnd, 0.5, veff);
    
    real prev_E = R - Q[ch];
    real alpha_eff = (prev_E > 0) ? alpha_win : alpha_loss;
    Q[ch] += alpha_eff * prev_E;
    
    real alpha_E = alpha_pc * prev_E;
    if (ch == 1) {
        W_PC_latent[1:half_D] += alpha_E * Z[1:half_D];
    } else {
        W_PC_latent[(half_D+1):D] += alpha_E * Z[(half_D+1):D];
    }
  }
}
generated quantities {
    real log_lik_total = 0.0;
}