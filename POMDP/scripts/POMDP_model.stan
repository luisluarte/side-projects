data {
  int<lower=1> N_subjects;
  int<lower=1> N_sessions_total;
  int<lower=1> N_timesteps_total;
  
  // Data Structure
  array[N_timesteps_total] int<lower=1> subject_idx;
  array[N_timesteps_total] int<lower=1> session_idx;
  array[N_timesteps_total] int<lower=1> state_id;    
  array[N_timesteps_total] int<lower=1> action_id;   
  array[N_timesteps_total] int<lower=1> next_state_id; 
  
  // Session Metadata
  array[N_sessions_total] int<lower=1> subj_map; 
  array[N_sessions_total] int<lower=1> start_idx; 
  array[N_sessions_total] int<lower=1> end_idx;   
  array[N_sessions_total] int<lower=1> true_context; 
  
  int<lower=1> N_contexts; 
  int<lower=1> N_actions;  
  int<lower=1> N_states;   
  
  // Pre-Computed Physics (Subject-Specific)
  // Q_star[subject, context, state, action]
  array[N_subjects] matrix[N_states, N_actions] Q_star[N_contexts]; 
  
  // Outcome Probabilities
  matrix[N_contexts, 2] context_probs; 
  
  // IDs
  int<lower=1> ID_IDLE;
  int<lower=1> ID_WAIT;
  int<lower=1> ID_LICK1;
  int<lower=1> ID_LICK2;
  int<lower=1> ID_REWARD_STATE;
  int<lower=1> ID_NOREWARD_STATE;
}

parameters {
  // Group-Level Hyperparameters
  real mu_beta;
  real<lower=0> sigma_beta;
  real mu_kappa;
  real<lower=0> sigma_kappa;
  real mu_log_tau;
  real<lower=0> sigma_log_tau;
  
  // Subject-Level Deviations
  vector[N_subjects] beta_raw;
  vector[N_subjects] kappa_raw;
  vector[N_subjects] log_tau_raw;
}

transformed parameters {
  vector[N_subjects] beta;
  vector[N_subjects] kappa;
  vector[N_subjects] tau;
  
  beta = mu_beta + sigma_beta * beta_raw;
  kappa = mu_kappa + sigma_kappa * kappa_raw;
  tau = exp(mu_log_tau + sigma_log_tau * log_tau_raw);
}

model {
  // --- Priors ---
  mu_beta ~ normal(0, 5);
  mu_kappa ~ normal(0, 5);
  mu_log_tau ~ normal(0, 1);
  sigma_beta ~ cauchy(0, 2);
  sigma_kappa ~ cauchy(0, 2);
  sigma_log_tau ~ cauchy(0, 2);
  
  beta_raw ~ std_normal();
  kappa_raw ~ std_normal();
  log_tau_raw ~ std_normal();
  
  // --- Likelihood ---
  vector[N_contexts] belief; 
  int current_subj = 0;
  
  for (s in 1:N_sessions_total) {
    int subj = subj_map[s];
    
    // 1. Initialize Belief
    if (subj != current_subj) {
      belief = rep_vector(1.0 / N_contexts, N_contexts);
      current_subj = subj;
    } 
    
    // 2. Run Time Steps
    for (t in start_idx[s]:end_idx[s]) {
      
      int st = state_id[t];
      int act = action_id[t];
      int next_st = next_state_id[t];
      
      vector[N_actions] Q_values;
      
      for (a in 1:N_actions) {
        real q_ext = 0;
        real bonus_explore = 0;
        real bonus_info = 0;
        
        // component 1: Extrinsic
        for (c in 1:N_contexts) {
          // INDEXING UPDATE: Q_star now depends on [c][subj]
          q_ext += belief[c] * Q_star[c, subj, st, a];
        }
        
        // component 2: Exploration Beta
        if (st == ID_IDLE && a == ID_WAIT) {
          bonus_explore = beta[subj];
        }
        
        // component 3: Information Kappa
        if (a == ID_LICK1 || a == ID_LICK2) {
           real H = 0;
           for (c in 1:N_contexts) {
             if(belief[c] > 0) H -= belief[c] * log(belief[c]);
           }
           bonus_info = kappa[subj] * H; 
        }
        
        Q_values[a] = q_ext + bonus_explore + bonus_info;
      }
      
      target += categorical_logit_lpmf(act | Q_values / tau[subj]);
      
      // C. Update Belief
      if (next_st == ID_REWARD_STATE || next_st == ID_NOREWARD_STATE) {
        vector[N_contexts] likelihoods;
        int spout_idx = (act == ID_LICK1) ? 1 : 2; 
        
        for (c in 1:N_contexts) {
          real p_reward = context_probs[c, spout_idx];
          if (next_st == ID_REWARD_STATE) {
            likelihoods[c] = p_reward;
          } else {
            likelihoods[c] = 1.0 - p_reward;
          }
        }
        
        belief = belief .* likelihoods;
        belief = belief / sum(belief); 
      }
    }
  }
}