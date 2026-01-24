functions {
  /**
   * partial_log_lik
   * STABILIZED VERSION
   * 1. Scalar Accumulation: Removes 'to_vector' heap allocations (Fixes Double Free).
   * 2. Strict Bounds: Prevents Gamma(0) errors.
   */
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
                       matrix mu_eta_pos, matrix mu_eta_neg, matrix mu_beta_slope,
                       vector sigma_beta_session, vector sigma_kappa_session,
                       matrix r_animal_beta, matrix r_animal_kappa, matrix r_animal_phi,
                       vector beta_session_raw, vector kappa_session_raw,
                       // INPUT: Standard 4D Array (Safe for reduce_sum)
                       array[,,,] real Q_star, 
                       matrix context_probs,
                       real belief_diffusion,
                       int N_physics_contexts, int N_actions,
                       int ID_IDLE, int ID_WAIT, int ID_LICK1, int ID_LICK2,
                       int ID_REWARD_STATE, int ID_NOREWARD_STATE) {
    
    real lp = 0;
    real dt = 0.025; 
    
    vector[N_actions] Q_values;
    vector[N_physics_contexts] belief;
    vector[N_physics_contexts] likelihoods;
    vector[N_physics_contexts] context_p_vec;
    vector[N_physics_contexts] uniform_prior = rep_vector(1.0 / N_physics_contexts, N_physics_contexts);
    
    real H = 0;
    
    for (i in 1:(end - start + 1)) {
      int animal_idx = animal_slice[i];
      belief = uniform_prior;
      
      real trait_b = r_animal_beta[animal_idx, 1];
      real trait_k = r_animal_kappa[animal_idx, 1];
      real trait_phi = r_animal_phi[animal_idx, 1];
      
      for (s in sessions_per_animal_start[animal_idx]:sessions_per_animal_end[animal_idx]) {
        int cog_ctx = session_cognitive_context[s]; 
        int d_idx = session_drug[s]; 
        int prev_act = 0; 
        real current_wait_time = 0;
        
        belief = (1.0 - belief_diffusion) * belief + belief_diffusion * uniform_prior;
        
        H = 0;
        for (c in 1:N_physics_contexts) if (belief[c] > 1e-12) H -= belief[c] * log(belief[c]);
        
        real b_s = mu_beta[d_idx, cog_ctx] + trait_b + sigma_beta_session[cog_ctx] * beta_session_raw[s];
        real k_s = mu_kappa[d_idx, cog_ctx] + trait_k + sigma_kappa_session[cog_ctx] * kappa_session_raw[s];
        real phi_s = mu_phi[d_idx, cog_ctx] + trait_phi;
        real side_s = mu_side[d_idx, cog_ctx];
        real b_slope = mu_beta_slope[d_idx, cog_ctx];
        
        real ep = exp(fmin(mu_eta_pos[d_idx, cog_ctx], 5.0));
        real en = exp(fmin(mu_eta_neg[d_idx, cog_ctx], 5.0));

        for (t in start_idx[s]:end_idx[s]) {
          int st = state_id[t];
          int act = action_id[t];
          int next_st = next_state_id[t];
          real w = weight[t];
          
          // STABILIZED: Scalar Accumulation
          // Avoids 'to_vector' heap allocation inside hot loop.
          // Compiler -O3 will still optimize this to registers.
          for (a in 1:N_actions) {
             real acc = 0;
             for (c in 1:N_physics_contexts) {
                acc += belief[c] * Q_star[c, animal_idx, st, a];
             }
             Q_values[a] = acc;
          }
          
          if (st == ID_IDLE) {
            real b_eff = b_s + b_slope * log1p(current_wait_time);
            Q_values[ID_WAIT] += (b_eff * dt);
          }
          
          Q_values[ID_LICK1] += k_s * H + side_s; 
          Q_values[ID_LICK2] += k_s * H;
          
          if (prev_act == ID_LICK1 || prev_act == ID_LICK2) Q_values[prev_act] += phi_s;
          
          for(a in 1:N_actions) Q_values[a] = fmin(fmax(Q_values[a], -50.0), 50.0);
          
          lp += w * categorical_logit_lpmf(act | Q_values); 
          
          if (act == ID_WAIT) current_wait_time += (w * dt);
          else current_wait_time = 0;
          prev_act = act;
          
          if (next_st == ID_REWARD_STATE || next_st == ID_NOREWARD_STATE) {
            int spout_idx = (act == ID_LICK1) ? 1 : 2;
            context_p_vec = col(context_probs, spout_idx);
            
            if (next_st == ID_REWARD_STATE) {
               for(c in 1:N_physics_contexts) likelihoods[c] = pow(context_p_vec[c], ep);
            } else {
               for(c in 1:N_physics_contexts) likelihoods[c] = pow(1.0 - context_p_vec[c], en);
            }
            
            belief = belief .* likelihoods;
            real sum_b = sum(belief);
            if (sum_b > 1e-15) belief /= sum_b;
            else belief = uniform_prior;
            
            H = 0;
            for (c in 1:N_physics_contexts) if (belief[c] > 1e-12) H -= belief[c] * log(belief[c]);
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
  
  // PRIMITIVE 4D ARRAY: The safest possible container for threading
  // [Context, Animal, State, Action]
  array[N_physics_contexts, N_animals, N_states, N_actions] real Q_star; 
  
  matrix[N_physics_contexts, 2] context_probs; 
  int ID_IDLE; int ID_WAIT; int ID_LICK1; int ID_LICK2;
  int ID_REWARD_STATE; int ID_NOREWARD_STATE;
  int grainsize; 
}

parameters {
  vector[N_cognitive_contexts] base_beta;
  vector[N_cognitive_contexts] base_kappa;
  vector[N_cognitive_contexts] base_phi;
  vector[N_cognitive_contexts] base_side;
  vector[N_cognitive_contexts] base_beta_slope;
  vector[N_cognitive_contexts] base_eta_pos;
  vector[N_cognitive_contexts] base_eta_neg;
  
  matrix[N_drugs, N_cognitive_contexts] delta_beta;
  matrix[N_drugs, N_cognitive_contexts] delta_kappa;
  matrix[N_drugs, N_cognitive_contexts] delta_phi;
  matrix[N_drugs, N_cognitive_contexts] delta_side;
  matrix[N_drugs, N_cognitive_contexts] delta_beta_slope;
  matrix[N_drugs, N_cognitive_contexts] delta_eta_pos;
  matrix[N_drugs, N_cognitive_contexts] delta_eta_neg;
  
  real<lower=0> sigma_beta_trait;
  real<lower=0> sigma_kappa_trait;
  real<lower=0> sigma_phi_trait;
  
  vector[N_animals] beta_trait_raw;
  vector[N_animals] kappa_trait_raw;
  vector[N_animals] phi_trait_raw;

  // FIX: Non-zero lower bound to prevent Gamma(0) error
  vector<lower=1e-6>[N_cognitive_contexts] sigma_beta_session;
  vector<lower=1e-6>[N_cognitive_contexts] sigma_kappa_session;
  
  vector[N_sessions_total] beta_session_raw;
  vector[N_sessions_total] kappa_session_raw;
  
  real<lower=0, upper=0.5> belief_diffusion;
}

transformed parameters {
  matrix[N_drugs, N_cognitive_contexts] mu_beta;
  matrix[N_drugs, N_cognitive_contexts] mu_kappa;
  matrix[N_drugs, N_cognitive_contexts] mu_phi;
  matrix[N_drugs, N_cognitive_contexts] mu_side;
  matrix[N_drugs, N_cognitive_contexts] mu_beta_slope;
  matrix[N_drugs, N_cognitive_contexts] mu_eta_pos;
  matrix[N_drugs, N_cognitive_contexts] mu_eta_neg;
  
  matrix[N_animals, 1] r_animal_beta = to_matrix(sigma_beta_trait * beta_trait_raw, N_animals, 1);
  matrix[N_animals, 1] r_animal_kappa = to_matrix(sigma_kappa_trait * kappa_trait_raw, N_animals, 1);
  matrix[N_animals, 1] r_animal_phi = to_matrix(sigma_phi_trait * phi_trait_raw, N_animals, 1);

  for (d in 1:N_drugs) {
    for (c in 1:N_cognitive_contexts) {
      mu_beta[d, c] = base_beta[c] + (d > 1 ? delta_beta[d, c] : 0);
      mu_kappa[d, c] = base_kappa[c] + (d > 1 ? delta_kappa[d, c] : 0);
      mu_phi[d, c] = base_phi[c] + (d > 1 ? delta_phi[d, c] : 0);
      mu_side[d, c] = base_side[c] + (d > 1 ? delta_side[d, c] : 0);
      mu_beta_slope[d, c] = base_beta_slope[c] + (d > 1 ? delta_beta_slope[d, c] : 0);
      mu_eta_pos[d, c] = base_eta_pos[c] + (d > 1 ? delta_eta_pos[d, c] : 0);
      mu_eta_neg[d, c] = base_eta_neg[c] + (d > 1 ? delta_eta_neg[d, c] : 0);
    }
  }
}

model {
  base_beta ~ normal(1.0, 1.0);
  base_kappa ~ normal(0.1, 0.5);
  base_phi ~ normal(0, 1.0);
  base_side ~ normal(0, 0.5);
  base_beta_slope ~ normal(0, 1.0);
  base_eta_pos ~ normal(0, 0.5);
  base_eta_neg ~ normal(0, 0.5);
  
  to_vector(delta_beta) ~ normal(0, 0.5);
  to_vector(delta_kappa) ~ normal(0, 0.25);
  to_vector(delta_phi) ~ normal(0, 0.5);
  to_vector(delta_side) ~ normal(0, 0.25);
  to_vector(delta_beta_slope) ~ normal(0, 0.5);
  to_vector(delta_eta_pos) ~ normal(0, 0.25);
  to_vector(delta_eta_neg) ~ normal(0, 0.25);
  
  beta_trait_raw ~ std_normal();
  kappa_trait_raw ~ std_normal();
  phi_trait_raw ~ std_normal();
  sigma_beta_trait ~ exponential(1.0);
  sigma_kappa_trait ~ exponential(1.0);
  sigma_phi_trait ~ exponential(1.0);

  sigma_beta_session ~ gamma(2, 10);
  sigma_kappa_session ~ exponential(1.0);
  
  beta_session_raw ~ std_normal();
  kappa_session_raw ~ std_normal();
  
  belief_diffusion ~ beta(1, 10);
  
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
                       mu_eta_pos, mu_eta_neg, mu_beta_slope,
                       sigma_beta_session, sigma_kappa_session, 
                       r_animal_beta, r_animal_kappa, r_animal_phi,
                       beta_session_raw, kappa_session_raw, 
                       Q_star, context_probs,
                       belief_diffusion,
                       N_physics_contexts, N_actions,
                       ID_IDLE, ID_WAIT, ID_LICK1, ID_LICK2,
                       ID_REWARD_STATE, ID_NOREWARD_STATE);
}