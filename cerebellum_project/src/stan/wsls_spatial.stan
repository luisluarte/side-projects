functions {
  real partial_sum(array[] int slice_subj, int start, int end, 
                   array[] int start_idx, array[] int end_idx, 
                   array[] int resp, array[] real reward, array[] real rt, 
                   vector a_base_raw, vector tnd, vector v_base, vector v_wsls) {
                   
    real lp = 0;
    
    for (s_idx in 1:size(slice_subj)) {
      int s = slice_subj[s_idx];
      int s_start = start_idx[s];
      int s_end = end_idx[s];
      
      real a_s = 0.11 + 3.0 * a_base_raw[s];
      real tnd_s = tnd[s];
      real vb_s = v_base[s];
      real vw_s = v_wsls[s];
      
      int prev_ch = -999;
      real prev_R = -999.0;
      
      for (t in s_start:s_end) {
        int ch = resp[t];
        
        if (ch > 0 && rt[t] > 0.0 && prev_ch > 0 && prev_R != -999.0) {
            // Predict Right (1) or Left (2)
            int stay_pred = prev_ch;
            int switch_pred = (prev_ch == 1) ? 2 : 1;
            
            int predicted_ch = (prev_R == 1.0) ? stay_pred : switch_pred;
            
            // +1 for Right, -1 for Left
            real S_spatial = (predicted_ch == 1) ? 1.0 : -1.0;
            
            real veff_raw = vb_s + vw_s * S_spatial;
            real veff = (ch == 1) ? veff_raw : -veff_raw;
            
            if (veff == 0.0) {
               lp += log(0.5);
            } else {
               lp += wiener_lpdf(rt[t] | a_s, tnd_s, 0.5, veff);
            }
        }
        
        if (ch > 0) {
            prev_ch = ch;
            prev_R = reward[t];
        }
      }
    }
    return lp;
  }
}

data {
  int<lower=1> N_trials;
  int<lower=1> N_subj;
  array[N_trials] int subj;
  array[N_trials] int resp;
  array[N_trials] real reward;
  array[N_trials] real rt;
  array[N_subj] real min_rt;
  array[N_subj] int start_idx;
  array[N_subj] int end_idx;
  
  vector[4] theta_mean;
  matrix[4, 4] L_Sigma;
  
  int grainsize; 
}

transformed data {
  array[N_subj] int seq_subj;
  for (s in 1:N_subj) seq_subj[s] = s;
}

parameters {
  vector[4] theta_raw;
  vector<lower=0>[4] sigma;
  matrix[4, N_subj] z;
}

transformed parameters {
  vector[4] theta_unc = theta_mean + L_Sigma * theta_raw;
  
  vector[N_subj] a_base_raw;
  vector[N_subj] tnd;
  vector[N_subj] v_base;
  vector[N_subj] v_wsls;

  for (s in 1:N_subj) {
    real tnd_cap = fmin(min_rt[s] - 0.05, 3.69);
    
    a_base_raw[s] = inv_logit(theta_unc[1] + sigma[1] * z[1, s]);
    tnd[s] = 0.01 + (tnd_cap - 0.01) * inv_logit(theta_unc[2] + sigma[2] * z[2, s]);
    
    v_base[s] = 18.51 * inv_logit(theta_unc[3] + sigma[3] * z[3, s]) - 9.255;
    v_wsls[s] = 18.51 * inv_logit(theta_unc[4] + sigma[4] * z[4, s]);
  }
}

model {
  theta_unc ~ normal(0, 2);
  sigma ~ gamma(2, 5);
  to_vector(z) ~ std_normal();

  target += reduce_sum(partial_sum, seq_subj, grainsize,
                       start_idx, end_idx, resp, reward, rt, 
                       a_base_raw, tnd, v_base, v_wsls);
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
      int s_start = start_idx[s];
      int s_end = end_idx[s];
      
      real a_s = 0.11 + 3.0 * a_base_raw[s];
      real tnd_s = tnd[s];
      real vb_s = v_base[s];
      real vw_s = v_wsls[s];
      
      int prev_ch = -999;
      real prev_R = -999.0;
      
      for (t in s_start:s_end) {
        int ch = resp[t];
        
        if (ch > 0 && rt[t] > 0.0 && prev_ch > 0 && prev_R != -999.0) {
            int stay_pred = prev_ch;
            int switch_pred = (prev_ch == 1) ? 2 : 1;
            
            int predicted_ch = (prev_R == 1.0) ? stay_pred : switch_pred;
            real S_spatial = (predicted_ch == 1) ? 1.0 : -1.0;
            
            real veff_raw = vb_s + vw_s * S_spatial;
            real veff = (ch == 1) ? veff_raw : -veff_raw;
            
            if (veff == 0.0) {
               log_lik[t] = log(0.5);
               pred_sw[t] = 0.5;
            } else {
               log_lik[t] = wiener_lpdf(rt[t] | a_s, tnd_s, 0.5, veff);
               real p_left = (exp(-veff_raw * a_s) - 1.0) / (exp(-2.0 * veff_raw * a_s) - 1.0);
               pred_sw[t] = p_left; // Probability of going Right (1)
            }
        }
        
        if (ch > 0) {
            prev_ch = ch;
            prev_R = reward[t];
        }
      }
    }
  }
}
