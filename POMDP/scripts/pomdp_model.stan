functions {
  /**
   * partial_log_lik
   * STABILIZED & OPTIMIZED VERSION
   * 1. Scalar Accumulation: Removes 'to_vector' heap allocations.
   * 2. Pre-computed Likelihoods: Removes pow() from the hot loop.
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
    vector[N_physics_contexts] uniform_prior = rep_vector(1.0 / N_physics_contexts, N_physics_contexts);
    real H = 0;
    
    // Memory allocation for pre-computed likelihood matrices (Contexts x Spouts)
    matrix[N_physics_contexts, 2] lik_reward;
    matrix[N_physics_contexts, 2] lik_noreward;
    
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
        
        // real ep = exp(fmin(mu_eta_pos[d_idx, cog_ctx], 5.0));
        // real en = exp(fmin(mu_eta_neg[d_idx, cog_ctx], 5.0));
        real ep = exp(mu_eta_pos[d_idx, cog_ctx]);
        real en = exp(mu_eta_neg[d_idx, cog_ctx]);

        // =========================================================================
        // PRE-COMPUTATION BLOCK: Calculate pow() exactly once per session
        // =========================================================================
        for (c in 1:N_physics_contexts) {
          lik_reward[c, 1]   = pow(context_probs[c, 1], ep);
          lik_reward[c, 2]   = pow(context_probs[c, 2], ep);
          lik_noreward[c, 1] = pow(1.0 - context_probs[c, 1], en);
          lik_noreward[c, 2] = pow(1.0 - context_probs[c, 2], en);
        }

        // The hot loop begins here
        for (t in start_idx[s]:end_idx[s]) {
          int st = state_id[t];
          int act = action_id[t];
          int next_st = next_state_id[t];
          real w = weight[t];

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
            
            // =========================================================================
            // FAST LOOKUP: O(1) vector assignment instead of N transcendental functions
            // =========================================================================
            if (next_st == ID_REWARD_STATE) {
               likelihoods = col(lik_reward, spout_idx);
            } else {
               likelihoods = col(lik_noreward, spout_idx);
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
  // ==============================================================================
  // 1. TRUE BASELINE: Pure behavior without injections (Anchored to Context 1)
  // ==============================================================================
  real base_beta;
  real base_kappa;
  real base_phi;
  real base_side;
  real base_beta_slope;
  real base_eta_pos;
  real base_eta_neg;

  // ==============================================================================
  // 2. VEHICLE SHIFT: The effect of (Context 'c' + Vehicle Injection) vs. True Baseline
  // ==============================================================================
  vector[N_cognitive_contexts] veh_shift_beta;
  vector[N_cognitive_contexts] veh_shift_kappa;
  vector[N_cognitive_contexts] veh_shift_phi;
  vector[N_cognitive_contexts] veh_shift_side;
  vector[N_cognitive_contexts] veh_shift_beta_slope;
  vector[N_cognitive_contexts] veh_shift_eta_pos;
  vector[N_cognitive_contexts] veh_shift_eta_neg;

  // ==============================================================================
  // 3. DRUG DELTA: The exact difference between Active Drug and Vehicle within Context 'c'
  // ==============================================================================
  vector[N_cognitive_contexts] drug_delta_beta;
  vector[N_cognitive_contexts] drug_delta_kappa;
  vector[N_cognitive_contexts] drug_delta_phi;
  vector[N_cognitive_contexts] drug_delta_side;
  vector[N_cognitive_contexts] drug_delta_beta_slope;
  vector[N_cognitive_contexts] drug_delta_eta_pos;
  vector[N_cognitive_contexts] drug_delta_eta_neg;
  
  // ==============================================================================
  // 4. SUBJECT-LEVEL TRAITS & SESSION DYNAMICS (Unchanged)
  // ==============================================================================
  real<lower=0> sigma_beta_trait;
  real<lower=0> sigma_kappa_trait;
  real<lower=0> sigma_phi_trait;
  
  vector[N_animals] beta_trait_raw;
  vector[N_animals] kappa_trait_raw;
  vector[N_animals] phi_trait_raw;

  vector<lower=1e-6>[N_cognitive_contexts] sigma_beta_session;
  vector<lower=1e-6>[N_cognitive_contexts] sigma_kappa_session;
  
  vector[N_sessions_total] beta_session_raw;
  vector[N_sessions_total] kappa_session_raw;

  real<lower=0, upper=0.5> belief_diffusion;
}

transformed parameters {
  // The final expected values mapped to [Drug_ID, Context_ID]
  matrix[N_drugs, N_cognitive_contexts] mu_beta;
  matrix[N_drugs, N_cognitive_contexts] mu_kappa;
  matrix[N_drugs, N_cognitive_contexts] mu_phi;
  matrix[N_drugs, N_cognitive_contexts] mu_side;
  matrix[N_drugs, N_cognitive_contexts] mu_beta_slope;
  matrix[N_drugs, N_cognitive_contexts] mu_eta_pos;
  matrix[N_drugs, N_cognitive_contexts] mu_eta_neg;
  
  // Trait construction
  matrix[N_animals, 1] r_animal_beta = to_matrix(sigma_beta_trait * beta_trait_raw, N_animals, 1);
  matrix[N_animals, 1] r_animal_kappa = to_matrix(sigma_kappa_trait * kappa_trait_raw, N_animals, 1);
  matrix[N_animals, 1] r_animal_phi = to_matrix(sigma_phi_trait * phi_trait_raw, N_animals, 1);

  // The Orthogonal Mapping Matrix
  for (c in 1:N_cognitive_contexts) {
    // ID 1: Baseline_NoInj (Only evaluated for c=1 in the data, but defined universally)
    mu_beta[1, c]         = base_beta;
    mu_kappa[1, c]        = base_kappa;
    mu_phi[1, c]          = base_phi;
    mu_side[1, c]         = base_side;
    mu_beta_slope[1, c]   = base_beta_slope;
    mu_eta_pos[1, c]      = base_eta_pos;
    mu_eta_neg[1, c]      = base_eta_neg;

    // ID 2: Vehicle in Context c (Shift from Baseline)
    mu_beta[2, c]         = base_beta       + veh_shift_beta[c];
    mu_kappa[2, c]        = base_kappa      + veh_shift_kappa[c];
    mu_phi[2, c]          = base_phi        + veh_shift_phi[c];
    mu_side[2, c]         = base_side       + veh_shift_side[c];
    mu_beta_slope[2, c]   = base_beta_slope + veh_shift_beta_slope[c];
    mu_eta_pos[2, c]      = base_eta_pos    + veh_shift_eta_pos[c];
    mu_eta_neg[2, c]      = base_eta_neg    + veh_shift_eta_neg[c];

    // ID 3: Active Drug in Context c (Shift from VEHICLE in Context c)
    mu_beta[3, c]         = mu_beta[2, c]       + drug_delta_beta[c];
    mu_kappa[3, c]        = mu_kappa[2, c]      + drug_delta_kappa[c];
    mu_phi[3, c]          = mu_phi[2, c]        + drug_delta_phi[c];
    mu_side[3, c]         = mu_side[2, c]       + drug_delta_side[c];
    mu_beta_slope[3, c]   = mu_beta_slope[2, c] + drug_delta_beta_slope[c];
    mu_eta_pos[3, c]      = mu_eta_pos[2, c]    + drug_delta_eta_pos[c];
    mu_eta_neg[3, c]      = mu_eta_neg[2, c]    + drug_delta_eta_neg[c];
  }
}

model {
  base_beta       ~ normal(1.0, 1.0);
  base_kappa      ~ normal(0.1, 0.5);
  base_phi        ~ normal(0, 1.0);
  base_side       ~ normal(0, 0.5);
  base_beta_slope ~ normal(0, 1.0);
  base_eta_pos    ~ normal(0, 0.5);
  base_eta_neg    ~ normal(0, 0.5);
  
  // Priors: Vehicle Shifts (Context/Injection Effects)
  veh_shift_beta       ~ normal(0, 0.5);
  veh_shift_kappa      ~ normal(0, 0.25);
  veh_shift_phi        ~ normal(0, 0.5);
  veh_shift_side       ~ normal(0, 0.25);
  veh_shift_beta_slope ~ normal(0, 0.5);
  veh_shift_eta_pos    ~ normal(0, 0.25);
  veh_shift_eta_neg    ~ normal(0, 0.25);

  drug_delta_beta       ~ normal(0, 0.5);
  drug_delta_kappa      ~ normal(0, 0.25);
  drug_delta_phi        ~ normal(0, 0.5);
  drug_delta_side       ~ normal(0, 0.25);
  drug_delta_beta_slope ~ normal(0, 0.5);
  drug_delta_eta_pos    ~ normal(0, 0.25);
  drug_delta_eta_neg    ~ normal(0, 0.25);
  
  beta_trait_raw  ~ std_normal();
  kappa_trait_raw ~ std_normal();
  phi_trait_raw   ~ std_normal();
  sigma_beta_trait  ~ normal(0, 1);
  sigma_kappa_trait ~ normal(0, 1);
  sigma_phi_trait   ~ normal(0, 1);

  sigma_beta_session  ~ normal(0, 1);
  sigma_kappa_session ~ normal(0, 1);
  
  beta_session_raw  ~ std_normal();
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
