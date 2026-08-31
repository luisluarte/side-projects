functions {
  real partial_sum(array[] int slice_t_idx,
                   int start, int end,
                   array[] int subj,
                   array[] int resp,
                   array[] real reward,
                   array[] real rt,
                   vector a_base,
                   vector tnd,
                   vector v_ctx,
                   vector alpha_win,
                   vector alpha_loss) {
    
    real pt = 0.0;
    int current_s = -1;
    vector[2] Q;
    
    real tnd_s;
    real a_s;
    real v_s;
    real aw_s;
    real al_s;
    
    for (i in start:end) {
      int s = subj[i];
      if (s != current_s) {
        current_s = s;
        Q = rep_vector(0.5, 2);
        tnd_s = tnd[s];
        a_s = a_base[s];
        v_s = v_ctx[s];
        aw_s = alpha_win[s];
        al_s = alpha_loss[s];
      }
      
      int ch = resp[i];
      real r = reward[i];
      
      real Q_diff = Q[1] - Q[2];
      real veff_raw = v_s * Q_diff;
      real veff = (ch == 1) ? veff_raw : -veff_raw;
      
      pt += wiener_lpdf(rt[i] | a_s, tnd_s, 0.5, veff);
      
      int unch = 3 - ch;
      real cf_r = 1.0 - r;
      
      real pe = r - Q[ch];
      real pe_unch = cf_r - Q[unch];
      
      real alpha_eff_ch = (pe > 0) ? aw_s : al_s;
      Q[ch] += alpha_eff_ch * pe;
      
      real alpha_eff_unch = (pe_unch > 0) ? aw_s : al_s;
      Q[unch] += alpha_eff_unch * pe_unch;
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
  array[N_subj] real min_rt;
}
transformed data {
  array[N_trials] int t_idx;
  for (t in 1:N_trials) {
    t_idx[t] = t;
  }
}
parameters {
  vector[5] mu_raw;
  vector<lower=0>[5] sigma;
  matrix[5, N_subj] z;
}
transformed parameters {
  vector[N_subj] a_base = exp(mu_raw[1] + sigma[1] * z[1]');
  vector[N_subj] tnd;
  vector[N_subj] v_ctx = exp(mu_raw[3] + sigma[3] * z[3]');
  vector[N_subj] alpha_win = inv_logit(mu_raw[4] + sigma[4] * z[4]');
  vector[N_subj] alpha_loss = inv_logit(mu_raw[5] + sigma[5] * z[5]');
  
  for (s in 1:N_subj) {
    real tnd_cap = fmin(min_rt[s] - 0.05, 3.69);
    tnd[s] = 0.01 + (tnd_cap - 0.01) * inv_logit(mu_raw[2] + sigma[2] * z[2, s]);
  }
}
model {
  mu_raw ~ normal(0, 1);
  sigma ~ normal(0, 0.5);
  to_vector(z) ~ std_normal();
  
  target += reduce_sum(partial_sum, t_idx, 1, subj, resp, reward, rt, a_base, tnd, v_ctx, alpha_win, alpha_loss);
}
generated quantities {
  vector[N_trials] log_lik;
  {
    int current_s = -1;
    vector[2] Q;
    for (i in 1:N_trials) {
      int s = subj[i];
      if (s != current_s) {
        current_s = s;
        Q = rep_vector(0.5, 2);
      }
      int ch = resp[i];
      real r = reward[i];
      
      real Q_diff = Q[1] - Q[2];
      real veff_raw = v_ctx[s] * Q_diff;
      real veff = (ch == 1) ? veff_raw : -veff_raw;
      
      log_lik[i] = wiener_lpdf(rt[i] | a_base[s], tnd[s], 0.5, veff);
      
      int unch = 3 - ch;
      real cf_r = 1.0 - r;
      
      real pe = r - Q[ch];
      real pe_unch = cf_r - Q[unch];
      
      real alpha_eff_ch = (pe > 0) ? alpha_win[s] : alpha_loss[s];
      Q[ch] += alpha_eff_ch * pe;
      
      real alpha_eff_unch = (pe_unch > 0) ? alpha_win[s] : alpha_loss[s];
      Q[unch] += alpha_eff_unch * pe_unch;
    }
  }
}