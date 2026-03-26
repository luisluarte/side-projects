functions {
  real partial_log_lik(array[] int animal_slice,
                       int start, int end,
                       array[] int sessions_per_animal_start,
                       array[] int sessions_per_animal_end,
                       array[] int start_idx,
                       array[] int end_idx,
                       array[] int state_id,
                       array[] int action_id,
                       array[] int next_state_id,
                       array[] int weight,
                       array[] int session_physics_context, 
                       array[] int session_cognitive_context, 
                       array[] int session_drug, 
                       matrix mu_beta, matrix mu_kappa, matrix mu_phi, matrix mu_side,
                       matrix mu_beta_slope,
                       matrix r_animal_beta, matrix r_animal_kappa, matrix r_animal_phi, matrix r_animal_beta_slope,
                       matrix epsilon,
                       int N_actions, int N_states,
                       int ID_IDLE, int ID_WAIT, int ID_LICK1, int ID_LICK2,
                       int ID_REWARD_STATE, int ID_NOREWARD_STATE) {
    
    real lp = 0;
    real dt = 0.025; 
    real log_random_prob = -log(N_actions * 1.0);
    
    for (i in 1:(end - start + 1)) {
      int animal_idx = animal_slice[i];
      real trait_b = r_animal_beta[animal_idx, 1];
      real trait_k = r_animal_kappa[animal_idx, 1];
      real trait_phi = r_animal_phi[animal_idx, 1];
      real trait_b_slope = r_animal_beta_slope[animal_idx, 1];
      
      for (s in sessions_per_animal_start[animal_idx]:sessions_per_animal_end[animal_idx]) {
        int cog_ctx = session_cognitive_context[s]; 
        int d_idx = session_drug[s];
        real current_epsilon = epsilon[d_idx, cog_ctx];
        
        int prev_act = 0; 
        int last_lick_spout = 0;
        real current_wait_time = 0;
        
        // bandit init with flat beta priors
        real alpha1 = 1.0;
        real beta1 = 1.0;
        real alpha2 = 1.0; 
        real beta2 = 1.0;
        
        real b_s = mu_beta[d_idx, cog_ctx] + trait_b;
        real k_s = mu_kappa[d_idx, cog_ctx] + trait_k;
        real phi_s = mu_phi[d_idx, cog_ctx] + trait_phi;
        real side_s = mu_side[d_idx, cog_ctx];
        real b_slope = mu_beta_slope[d_idx, cog_ctx] + trait_b_slope;
        
        // hot loop begins here
        for (t in start_idx[s]:end_idx[s]) {
          int st = state_id[t];
          int act = action_id[t];
          int next_st = next_state_id[t];
          int w = weight[t];
          
          // compute EV and uncertainty (var) for both spouts
          real ev1 = alpha1 / (alpha1 + beta1);
          real ev2 = alpha2 / (alpha2 + beta2);
          
          // guard against log(0)
          real p1 = fmax(fmin(ev1, 0.999999), 0.000001);
          real p2 = fmax(fmin(ev2, 0.999999), 0.000001);
          
          // compute distribution entropy
          real raw_u1 = -p1 * log(p1) - (1.0 - p1) * log(1.0 - p1);
          real raw_u2 = -p2 * log(p2) - (1.0 - p2) * log(1.0 - p2);
          
          real u1 = raw_u1;
          real u2 = raw_u2;
          
          if (st == ID_IDLE) {
              last_lick_spout = 0;
          }
          
          vector[N_actions] Q_base = rep_vector(0.0, N_actions);
          Q_base[ID_LICK1] = (ev1 * 5.0) + side_s;
          Q_base[ID_LICK2] = (ev2 * 5.0);
          
          if (last_lick_spout == 0) {
              Q_base[ID_LICK1] += k_s * u1;
              Q_base[ID_LICK2] += k_s * u2;
          }
          else if (last_lick_spout == ID_LICK1) {
            Q_base[ID_LICK1] += phi_s;
            Q_base[ID_LICK2] += k_s * u2;
          }
          else if (last_lick_spout == ID_LICK2) {
            Q_base[ID_LICK2] += phi_s;
            Q_base[ID_LICK1] += k_s * u1;
          }

          if (act == ID_WAIT) {
              if (w <= 3) {
                  for (k in 1:w) {
                      vector[N_actions] Q_step = Q_base;
                      if (st == ID_IDLE) {
                          real b_eff = b_s + b_slope * log1p(current_wait_time);
                          Q_step[ID_WAIT] += b_eff;
                      }
                      real log_softmax = categorical_logit_lpmf(act | Q_step);
                      lp += log_mix(current_epsilon, log_random_prob, log_softmax);
                      current_wait_time += dt;
                  }
              } else {
                  real t_start = current_wait_time;
                  real t_end   = current_wait_time + (w - 1) * dt;
                  real t_mid   = current_wait_time + ((w - 1) / 2.0) * dt;

                  vector[N_actions] Q_start = Q_base;
                  vector[N_actions] Q_mid   = Q_base;
                  vector[N_actions] Q_end   = Q_base;
                  
                  if (st == ID_IDLE) {
                      Q_start[ID_WAIT] += (b_s + b_slope * log1p(t_start));
                      Q_mid[ID_WAIT]   += (b_s + b_slope * log1p(t_mid));
                      Q_end[ID_WAIT]   += (b_s + b_slope * log1p(t_end));
                  }

                  // <-- ADDED: Mixture to Simpson's rule
                  real lp_start = log_mix(current_epsilon, log_random_prob, categorical_logit_lpmf(act | Q_start));
                  real lp_mid   = log_mix(current_epsilon, log_random_prob, categorical_logit_lpmf(act | Q_mid));
                  real lp_end   = log_mix(current_epsilon, log_random_prob, categorical_logit_lpmf(act | Q_end));
                  
                  lp += w * (lp_start + 4 * lp_mid + lp_end) / 6.0;
                  current_wait_time += w * dt;
              }
          } 
          else {
              if (st == ID_IDLE) {
                  real b_eff = b_s + b_slope * log1p(current_wait_time);
                  Q_base[ID_WAIT] += b_eff;
              }
              real log_softmax = categorical_logit_lpmf(act | Q_base);
              lp += w * log_mix(current_epsilon, log_random_prob, log_softmax);
              current_wait_time = 0;
          }
          
          prev_act = act;
          if (act == ID_LICK1 || act == ID_LICK2) {
              last_lick_spout = act;
          }

          // Pure Integration (No Decay)
          if (next_st == ID_REWARD_STATE || next_st == ID_NOREWARD_STATE) {
            last_lick_spout = 0;
            if (next_st == ID_REWARD_STATE) {
                if (act == ID_LICK1) alpha1 += 1.0;
                else if (act == ID_LICK2) alpha2 += 1.0;
            } else {
                if (act == ID_LICK1) beta1 += 1.0;
                else if (act == ID_LICK2) beta2 += 1.0;
            }
          }
        }
      }
    }
    return lp;
  }
}

data {
  int<lower=1> N_animals;
  int<lower=1> N_sessions_total;
  int<lower=1> N_compressed_steps; 
  int<lower=1> N_drugs;
  int<lower=1> N_physics_contexts; 
  int<lower=1> N_cognitive_contexts; 
  array[N_animals] int<lower=1> sessions_per_animal_start;
  array[N_animals] int<lower=1> sessions_per_animal_end;
  array[N_sessions_total] int<lower=1> session_physics_context;
  array[N_sessions_total] int<lower=1> session_cognitive_context;
  array[N_sessions_total] int<lower=1> session_drug; 
  array[N_compressed_steps] int<lower=1> state_id;    
  array[N_compressed_steps] int<lower=1> action_id;
  array[N_compressed_steps] int<lower=1> next_state_id; 
  array[N_compressed_steps] int<lower=1> weight; 
  array[N_sessions_total] int<lower=1> start_idx; 
  array[N_sessions_total] int<lower=1> end_idx;   
  int<lower=1> N_actions;  
  int<lower=1> N_states;   
  
  int ID_IDLE; int ID_WAIT;
  int ID_LICK1; int ID_LICK2;
  int ID_REWARD_STATE; int ID_NOREWARD_STATE;
  int grainsize; 
}

parameters {
  // BASELINE
  real base_beta;
  real base_kappa;
  real base_phi;
  real base_side;
  real base_beta_slope;
  matrix<lower=0, upper=1>[N_drugs, N_cognitive_contexts] epsilon;

  // VEHICLE SHIFT effect of context + veh over baseline
  vector[N_cognitive_contexts] veh_shift_beta;
  vector[N_cognitive_contexts] veh_shift_kappa;
  vector[N_cognitive_contexts] veh_shift_phi;
  vector[N_cognitive_contexts] veh_shift_side;
  vector[N_cognitive_contexts] veh_shift_beta_slope;

  // DRUG DELTA
  vector[N_cognitive_contexts] drug_delta_beta;
  vector[N_cognitive_contexts] drug_delta_kappa;
  vector[N_cognitive_contexts] drug_delta_phi;
  vector[N_cognitive_contexts] drug_delta_side;
  vector[N_cognitive_contexts] drug_delta_beta_slope;
  
  // animal-level traits
  real<lower=0> sigma_beta_trait;
  real<lower=0> sigma_kappa_trait;
  real<lower=0> sigma_phi_trait;
  real<lower=0> sigma_beta_slope_trait;
  
  vector[N_animals] beta_trait_raw;
  vector[N_animals] kappa_trait_raw;
  vector[N_animals] phi_trait_raw;
  vector[N_animals] beta_slope_trait_raw;
}

transformed parameters {
  matrix[N_drugs, N_cognitive_contexts] mu_beta;
  matrix[N_drugs, N_cognitive_contexts] mu_kappa;
  matrix[N_drugs, N_cognitive_contexts] mu_phi;
  matrix[N_drugs, N_cognitive_contexts] mu_side;
  matrix[N_drugs, N_cognitive_contexts] mu_beta_slope;

  matrix[N_animals, 1] r_animal_beta = to_matrix(sigma_beta_trait * beta_trait_raw, N_animals, 1);
  matrix[N_animals, 1] r_animal_kappa = to_matrix(sigma_kappa_trait * kappa_trait_raw, N_animals, 1);
  matrix[N_animals, 1] r_animal_phi = to_matrix(sigma_phi_trait * phi_trait_raw, N_animals, 1);
  matrix[N_animals, 1] r_animal_beta_slope = to_matrix(sigma_beta_slope_trait * beta_slope_trait_raw, N_animals, 1);

  for (c in 1:N_cognitive_contexts) {
    mu_beta[1, c]         = base_beta;
    mu_kappa[1, c]        = base_kappa;
    mu_phi[1, c]          = base_phi;
    mu_side[1, c]         = base_side;
    mu_beta_slope[1, c]   = base_beta_slope;

    mu_beta[2, c]         = base_beta       + veh_shift_beta[c];
    mu_kappa[2, c]        = base_kappa      + veh_shift_kappa[c];
    mu_phi[2, c]          = base_phi        + veh_shift_phi[c];
    mu_side[2, c]         = base_side       + veh_shift_side[c];
    mu_beta_slope[2, c]   = base_beta_slope + veh_shift_beta_slope[c];

    mu_beta[3, c]         = mu_beta[2, c]       + drug_delta_beta[c];
    mu_kappa[3, c]        = mu_kappa[2, c]      + drug_delta_kappa[c];
    mu_phi[3, c]          = mu_phi[2, c]        + drug_delta_phi[c];
    mu_side[3, c]         = mu_side[2, c]       + drug_delta_side[c];
    mu_beta_slope[3, c]   = mu_beta_slope[2, c] + drug_delta_beta_slope[c];
  }
}

model {
  // 1. Regularized Priors (To prevent Softmax saturation and Treedepth limits)
  base_beta       ~ normal(0, 1.5);
  base_phi        ~ normal(0, 1.5);
  base_side       ~ normal(0, 1.5);
  base_beta_slope ~ normal(0, 1.5);
  base_kappa      ~ normal(0, 1.5);
  for (d in 1:N_drugs) {
    for (c in 1:N_cognitive_contexts){
      epsilon[d, c] ~ beta(1, 19);
    }
  }
  
  veh_shift_beta       ~ normal(0, 1.0);
  veh_shift_kappa      ~ normal(0, 1.5);
  veh_shift_phi        ~ normal(0, 1.0);
  veh_shift_side       ~ normal(0, 1.0);
  veh_shift_beta_slope ~ normal(0, 1.0);

  drug_delta_beta       ~ normal(0, 1.0);
  drug_delta_kappa      ~ normal(0, 1.5);
  drug_delta_phi        ~ normal(0, 1.0);
  drug_delta_side       ~ normal(0, 1.0);
  drug_delta_beta_slope ~ normal(0, 1.0);
  
  // 3. Trait Priors
  beta_trait_raw  ~ std_normal();
  kappa_trait_raw ~ std_normal();
  phi_trait_raw   ~ std_normal();
  beta_slope_trait_raw ~ std_normal();

  sigma_beta_trait  ~ normal(0, 1.0);
  sigma_phi_trait   ~ normal(0, 1.0);
  sigma_beta_slope_trait ~ normal(0, 1.0);
  sigma_kappa_trait ~ normal(0, 1.0);

  array[N_animals] int animal_indices;
  for (i in 1:N_animals) animal_indices[i] = i;

  target += reduce_sum(partial_log_lik, animal_indices, grainsize,
                       sessions_per_animal_start, sessions_per_animal_end,
                       start_idx, end_idx,
                       state_id, action_id, next_state_id, weight,
                       session_physics_context,
                       session_cognitive_context,
                       session_drug,
                       mu_beta, mu_kappa, mu_phi, mu_side,
                       mu_beta_slope,
                       r_animal_beta, r_animal_kappa, r_animal_phi, r_animal_beta_slope,
                       epsilon,
                       N_actions, N_states,
                       ID_IDLE, ID_WAIT, ID_LICK1, ID_LICK2,
                       ID_REWARD_STATE, ID_NOREWARD_STATE);
}