functions {
  real partial_sum(array[] int slice_t_idx,
                   int start, int end,
                   array[] int subj,
                   array[] int resp,
                   array[] real reward,
                   array[] real rt,
                   array[] real iti,
                   vector a_base,
                   vector tnd,
                   vector v_ctx,
                   vector alpha_win,
                   vector alpha_loss,
                   vector k_decay,
                   vector beta_conflict) {
    
    int chunk_size = end - start + 1;
    vector[chunk_size] a_dyn_vec;
    vector[chunk_size] veff_vec;
    vector[chunk_size] tnd_vec;
    
    int current_s = -1;
    vector[2] Q;
    
    real tnd_s;
    real a_s;
    real v_s;
    real aw_s;
    real al_s;
    real k_dec_s;
    real b_conf_s;
    
    for (i in 1:chunk_size) {
      int orig_i = start + i - 1;
      int s = subj[orig_i];
      if (s != current_s) {
        current_s = s;
        Q = rep_vector(0.5, 2);
        tnd_s = tnd[s];
        a_s = a_base[s];
        v_s = v_ctx[s];
        aw_s = alpha_win[s];
        al_s = alpha_loss[s];
        k_dec_s = k_decay[s];
        b_conf_s = beta_conflict[s];
      }
      
      int ch = resp[orig_i];
      real r = reward[orig_i];
      
      real decay = exp(-iti[orig_i] * k_dec_s);
      Q = 0.5 + (Q - 0.5) * decay;
      
      real Q_diff = Q[1] - Q[2];
      real conflict = 1.0 - sqrt(square(Q_diff) + 1e-8);
      
      a_dyn_vec[i] = a_s + 5.0 * tanh(b_conf_s * conflict);
      tnd_vec[i] = tnd_s;
      
      real veff_raw = v_s * Q_diff;
      veff_vec[i] = (ch == 1) ? veff_raw : -veff_raw;
      
      real pe = r - Q[ch];
      real alpha_eff = (pe > 0) ? aw_s : al_s;
      Q[ch] += alpha_eff * pe;
    }
    
    return wiener_lpdf(rt[start:end] | a_dyn_vec, tnd_vec, 0.5, veff_vec);
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
}
transformed data {
  array[N_trials] int t_idx;
  array[N_trials] real clean_iti;
  for (t in 1:N_trials) {
    t_idx[t] = t;
    clean_iti[t] = (iti[t] < 0) ? 1.0 : iti[t];
  }
}
parameters {
  vector[7] mu_raw;
  vector<lower=0>[7] sigma;
  matrix[7, N_subj] z;
}
transformed parameters {
  real a_max = 5.0;
  real v_max = 20.0;
  real beta_max = 20.0;
  real k_max = 10.0;

  vector[N_subj] a_base = 0.1 + a_max * inv_logit(mu_raw[1] + sigma[1] * z[1]');
  vector[N_subj] tnd;
  vector[N_subj] v_ctx = v_max * inv_logit(mu_raw[3] + sigma[3] * z[3]');
  vector[N_subj] alpha_win = inv_logit(mu_raw[4] + sigma[4] * z[4]');
  vector[N_subj] alpha_loss = inv_logit(mu_raw[5] + sigma[5] * z[5]');
  vector[N_subj] k_decay = k_max * inv_logit(mu_raw[6] + sigma[6] * z[6]');
  vector[N_subj] beta_conflict = beta_max * inv_logit(mu_raw[7] + sigma[7] * z[7]');
  
  for (s in 1:N_subj) {
    real tnd_cap = fmin(min_rt[s] - 0.05, 3.69);
    tnd[s] = 0.01 + (tnd_cap - 0.01) * inv_logit(mu_raw[2] + sigma[2] * z[2, s]);
  }
}
model {
  mu_raw ~ normal(0, 1);
  sigma ~ normal(0, 0.5);
  to_vector(z) ~ std_normal();
  
  target += reduce_sum(partial_sum, t_idx, 1, subj, resp, reward, rt, clean_iti, a_base, tnd, v_ctx, alpha_win, alpha_loss, k_decay, beta_conflict);
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
      
      real decay = exp(-clean_iti[i] * k_decay[s]);
      Q = 0.5 + (Q - 0.5) * decay;
      
      real Q_diff = Q[1] - Q[2];
      real conflict = 1.0 - sqrt(square(Q_diff) + 1e-8);
      real a_dyn = a_base[s] + 5.0 * tanh(beta_conflict[s] * conflict);
      
      real veff_raw = v_ctx[s] * Q_diff;
      real veff = (ch == 1) ? veff_raw : -veff_raw;
      
      log_lik[i] = wiener_lpdf(rt[i] | a_dyn, tnd[s], 0.5, veff);
      
      real pe = r - Q[ch];
      real alpha_eff = (pe > 0) ? alpha_win[s] : alpha_loss[s];
      Q[ch] += alpha_eff * pe;
    }
  }
}
