functions {
  /**
   * partial log lik function
   * processes likelihood for shards of animals in parallel.
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
                       matrix mu_beta, matrix mu_kappa,
                       vector sigma_beta_session, vector sigma_kappa_session,
                       real sigma_beta_trait, real sigma_kappa_trait,
                       vector beta_trait_raw, vector kappa_trait_raw,
                       matrix beta_session_raw, matrix kappa_session_raw,
                       array[,,,] real Q_star, 
                       matrix context_probs,
                       real belief_diffusion,
                       int N_physics_contexts, int N_actions,
                       int ID_IDLE, int ID_WAIT, int ID_LICK1, int ID_LICK2,
                       int ID_REWARD_STATE, int ID_NOREWARD_STATE) {
    
    real lp = 0;
    real dt = 0.025; 
    real tau_fixed = 1.0; 
    
    for (i in 1:(end - start + 1)) {
      int animal_idx = animal_slice[i];
      
      real trait_b = sigma_beta_trait * beta_trait_raw[animal_idx];
      real trait_k = sigma_kappa_trait * kappa_trait_raw[animal_idx];
      
      int s_count = 1;
      for (s in sessions_per_animal_start[animal_idx]:sessions_per_animal_end[animal_idx]) {
        int cog_ctx = session_cognitive_context[s]; 
        int d_idx = session_drug[s]; 
        
        // Belief initialization and Diffusion (Once per session)
        vector[N_physics_contexts] belief = rep_vector(1.0 / N_physics_contexts, N_physics_contexts);
        belief = (1.0 - belief_diffusion) * belief + belief_diffusion * rep_vector(1.0 / N_physics_contexts, N_physics_contexts);
        
        // Recalculate Entropy only when belief changes
        real H = 0;
        for (c in 1:N_physics_contexts) if(belief[c] > 1e-12) H -= belief[c] * log(belief[c]);
        
        real b_s = mu_beta[d_idx, cog_ctx] + trait_b + sigma_beta_session[cog_ctx] * beta_session_raw[animal_idx, s_count];
        real k_s = mu_kappa[d_idx, cog_ctx] + trait_k + sigma_kappa_session[cog_ctx] * kappa_session_raw[animal_idx, s_count];
        
        for (t in start_idx[s]:end_idx[s]) {
          int st = state_id[t];
          int act = action_id[t];
          int next_st = next_state_id[t];
          real w = weight[t];
          
          // Vectorized valuation: Projected from Physics matrix onto current Belief fiber
          vector[N_actions] Q_values = rep_vector(0, N_actions);
          for (c in 1:N_physics_contexts) {
            if (belief[c] > 1e-12) {
                for (a in 1:N_actions) {
                    Q_values[a] += belief[c] * Q_star[c, animal_idx, st, a];
                }
            }
          }
          
          // Add Cognitive Biases
          if (st == ID_IDLE) Q_values[ID_WAIT] += (b_s * dt);
          Q_values[ID_LICK1] += k_s * H;
          Q_values[ID_LICK2] += k_s * H;
          
          lp += w * categorical_logit_lpmf(act | Q_values / tau_fixed);
          
          // Only update belief (and entropy) at outcomes to save computation
          if (next_st == ID_REWARD_STATE || next_st == ID_NOREWARD_STATE) {
            vector[N_physics_contexts] likelihoods;
            int spout_idx = (act == ID_LICK1) ? 1 : 2; 
            for (c in 1:N_physics_contexts) {
              real p = context_probs[c, spout_idx];
              likelihoods[c] = (next_st == ID_REWARD_STATE) ? p : (1.0 - p);
            }
            belief = belief .* likelihoods;
            real sum_b = sum(belief);
            if (sum_b > 1e-15) belief /= sum_b;
            else belief = rep_vector(1.0 / N_physics_contexts, N_physics_contexts);
            
            // Recompute Entropy for the next set of steps
            H = 0;
            for (c in 1:N_physics_contexts) if(belief[c] > 1e-12) H -= belief[c] * log(belief[c]);
          }
        }
        s_count += 1;
      }
    }
    return lp;
  }
}

data {
  int<lower=1> N_animals;
  int<lower=1> N_sessions_total;
  int<lower=1> N_max_sessions_per_animal;
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
  array[N_physics_contexts, N_animals, N_states, N_actions] real Q_star; 
  matrix[N_physics_contexts, 2] context_probs; 
  int ID_IDLE; int ID_WAIT; int ID_LICK1; int ID_LICK2;
  int ID_REWARD_STATE; int ID_NOREWARD_STATE;
  int grainsize; 
}

parameters {
  vector[N_cognitive_contexts] base_beta;
  vector[N_cognitive_contexts] base_kappa;
  
  matrix[N_drugs, N_cognitive_contexts] delta_beta;
  matrix[N_drugs, N_cognitive_contexts] delta_kappa;
  
  vector<lower=0>[N_cognitive_contexts] sigma_beta_session;
  vector<lower=0>[N_cognitive_contexts] sigma_kappa_session;
  real<lower=0> sigma_beta_trait;
  real<lower=0> sigma_kappa_trait;
  
  vector[N_animals] beta_trait_raw;
  vector[N_animals] kappa_trait_raw;
  matrix[N_animals, N_max_sessions_per_animal] beta_session_raw;
  matrix[N_animals, N_max_sessions_per_animal] kappa_session_raw;
  
  real<lower=0, upper=0.5> belief_diffusion;
  
  real<lower=2> nu_beta;
  real<lower=2> nu_kappa;
}

transformed parameters {
  matrix[N_drugs, N_cognitive_contexts] mu_beta;
  matrix[N_drugs, N_cognitive_contexts] mu_kappa;
  
  for (d in 1:N_drugs) {
    for (c in 1:N_cognitive_contexts) {
      if (d == 1) {
        mu_beta[d, c] = base_beta[c];
        mu_kappa[d, c] = base_kappa[c];
      } else {
        mu_beta[d, c] = base_beta[c] + delta_beta[d, c];
        mu_kappa[d, c] = base_kappa[c] + delta_kappa[d, c];
      }
    }
  }
}

model {
  // PRIORS: Regularized for HMC Stability
  base_beta ~ normal(1.0, 1.0); 
  base_kappa ~ normal(0.1, 0.5); 
  to_vector(delta_beta) ~ normal(0, 0.5); 
  to_vector(delta_kappa) ~ normal(0, 0.25); 
  
  nu_beta ~ gamma(2, 0.1); 
  nu_kappa ~ gamma(2, 0.1);
  
  beta_trait_raw ~ student_t(nu_beta, 0, 1);
  kappa_trait_raw ~ student_t(nu_kappa, 0, 1);
  
  to_vector(beta_session_raw) ~ std_normal();
  to_vector(kappa_session_raw) ~ std_normal();
  
  sigma_beta_session ~ exponential(1.0);
  sigma_kappa_session ~ exponential(1.0);
  sigma_beta_trait ~ exponential(1.0);
  sigma_kappa_trait ~ exponential(1.0);
  
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
                       mu_beta, mu_kappa, 
                       sigma_beta_session, sigma_kappa_session, 
                       sigma_beta_trait, sigma_kappa_trait, 
                       beta_trait_raw, kappa_trait_raw, 
                       beta_session_raw, kappa_session_raw, 
                       Q_star, context_probs,
                       belief_diffusion,
                       N_physics_contexts, N_actions,
                       ID_IDLE, ID_WAIT, ID_LICK1, ID_LICK2,
                       ID_REWARD_STATE, ID_NOREWARD_STATE);
}