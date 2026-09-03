functions {
  real partial_sum(array[] int slice_subj,
                   int start, int end,
                   array[] int start_idx,
                   array[] int end_idx,
                   array[] int resp,
                   array[] real reward,
                   array[] real rt,
                   vector a_base_raw,
                   vector tnd,
                   vector v_ctx,
                   vector aw,
                   vector al,
                   vector w_ctx,
                   vector beta_mismatch) {
    real pt = 0;
    for (idx in 1:size(slice_subj)) {
      int s = slice_subj[idx];
      vector[2] Q = rep_vector(0.5, 2);
      real phys_a_base = 0.11 + 3.0 * inv_logit(a_base_raw[s]);
      real delta_max = 1.0 / phys_a_base;
      real tnd_s = tnd[s];
      real aw_s = aw[s];
      real al_s = al[s];
      real v_s = v_ctx[s];
      real w_ctx_s = w_ctx[s];
      real beta_mis_s = beta_mismatch[s];

      for (t in start_idx[s]:end_idx[s]) {
        int ch = resp[t];
        real R = reward[t];
        real Q_diff = Q[1] - Q[2];
        
        real M_align = tanh(w_ctx_s * Q_diff);
        real caution = log1p_exp(-5.0 * abs(M_align)) * 0.1;
        real a_dyn = phys_a_base + delta_max * tanh(beta_mis_s * caution);
        
        if (t > start_idx[s]) {
             if (ch > 0 && rt[t] > 0.0) {
                 real veff_raw = v_s * Q_diff;
                 real veff = (ch == 1) ? veff_raw : -veff_raw;
                 pt += wiener_lpdf(rt[t] | a_dyn, tnd_s, 0.5, veff);
             }
          }
        
        if (ch > 0) {
            real pe = R - Q[ch];
            real alpha_eff = (pe > 0) ? aw_s : al_s;
            Q[ch] += alpha_eff * pe;
        }
      }
    }
    return pt;
  }
}
data {
  int<lower=1> N_trials;
  int<lower=1> N_subj;
  array[N_trials] int<lower=1, upper=N_subj> subj;
  array[N_trials] int<lower=-999, upper=2> resp;
  array[N_trials] real reward;
  array[N_trials] real rt;
  array[N_subj] real min_rt;
  
  array[N_subj] int<lower=1> start_idx;
  array[N_subj] int<lower=1> end_idx;
  
  vector[8] theta_mean;
  matrix[8, 8] L_Sigma;
  int grainsize;
}
transformed data {
  array[N_subj] int seq_subj;
  for(s in 1:N_subj) seq_subj[s] = s;
}
parameters {
  vector[8] theta_raw;
  vector<lower=0>[8] sigma;
  matrix[8, N_subj] z;
}
transformed parameters {
  vector[8] theta_unc = theta_mean + L_Sigma * theta_raw;
  vector[N_subj] a_base_raw;
  vector[N_subj] tnd;
  vector[N_subj] v_ctx;
    vector[N_subj] aw;
  vector[N_subj] al;
  vector[N_subj] w_ctx;
  vector[N_subj] beta_mismatch;
  
  for (s in 1:N_subj) {
    a_base_raw[s] = theta_unc[1] + sigma[1] * z[1, s];
    real tnd_cap = fmin(min_rt[s] - 0.05, 3.69);
    tnd[s] = 0.01 + (tnd_cap - 0.01) * inv_logit(theta_unc[2] + sigma[2] * z[2, s]);
    v_ctx[s] = 18.51 * inv_logit(theta_unc[3] + sigma[3] * z[3, s]);
        aw[s] = inv_logit(theta_unc[4] + sigma[4] * z[4, s]);
    al[s] = inv_logit(theta_unc[5] + sigma[5] * z[5, s]);
    w_ctx[s] = exp(theta_unc[6] + sigma[6] * z[6, s]);
    beta_mismatch[s] = exp(theta_unc[7] + sigma[7] * z[7, s]);
  }
}
model {
  theta_unc[1:3] ~ normal(0, 2);
  theta_unc[4:8] ~ normal(0, 2);
  sigma ~ normal(0, 0.5);
  to_vector(z) ~ std_normal();

  target += reduce_sum(partial_sum, seq_subj, grainsize, 
                       start_idx, end_idx, resp, reward, rt, 
                       a_base_raw, tnd, v_ctx, aw, al, w_ctx, beta_mismatch);
}
generated quantities {
  vector[N_trials] log_lik;
  vector[N_trials] pred_sw;
  for (t in 1:N_trials) {
      log_lik[t] = 0.0;
      pred_sw[t] = -1.0;
  }
  
  {
    for (s in 1:N_subj) {
      vector[2] Q = rep_vector(0.5, 2);
      real phys_a_base = 0.11 + 3.0 * inv_logit(a_base_raw[s]);
      real delta_max = 1.0 / phys_a_base;
      real tnd_s = tnd[s];
      real aw_s = aw[s];
      real al_s = al[s];
      real v_s = v_ctx[s];
      real w_ctx_s = w_ctx[s];
      real beta_mis_s = beta_mismatch[s];

      for (t in start_idx[s]:end_idx[s]) {
        int ch = resp[t];
        real R = reward[t];
        real Q_diff = Q[1] - Q[2];
        
        real M_align = tanh(w_ctx_s * Q_diff);
        real caution = log1p_exp(-5.0 * abs(M_align)) * 0.1;
        real a_dyn = phys_a_base + delta_max * tanh(beta_mis_s * caution);
        
        if (t > start_idx[s]) {
             int prev_ch = resp[t-1];
             if (ch > 0 && rt[t] > 0.0) {
                 real veff_raw = v_s * Q_diff;
                 real veff = (ch == 1) ? veff_raw : -veff_raw;
                 log_lik[t] = wiener_lpdf(rt[t] | a_dyn, tnd_s, 0.5, veff);
                 
                 if (veff_raw == 0.0) {
                     pred_sw[t] = 0.5;
                 } else {
                     real p_left = (exp(-veff_raw * a_dyn) - 1.0) / (exp(-2.0 * veff_raw * a_dyn) - 1.0);
                     pred_sw[t] = p_left;
                 }
             }
          }
        
        if (ch > 0) {
            real pe = R - Q[ch];
            real alpha_eff = (pe > 0) ? aw_s : al_s;
            Q[ch] += alpha_eff * pe;
        }
      }
    }
  }
}

