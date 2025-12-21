data {
  int<lower=1> N_animals;
  int<lower=1> N_sessions_total;
  int<lower=1> N_compressed_steps; 
  
  array[N_compressed_steps] int<lower=1> animal_idx;
  array[N_compressed_steps] int<lower=1> state_id;    
  array[N_compressed_steps] int<lower=1> action_id;   
  array[N_compressed_steps] int<lower=1> next_state_id; 
  array[N_compressed_steps] int<lower=1> weight; 
  
  array[N_sessions_total] int<lower=1> animal_map; 
  array[N_sessions_total] int<lower=1> start_idx; 
  array[N_sessions_total] int<lower=1> end_idx;   
  array[N_sessions_total] int<lower=1> true_context; 
  
  int<lower=1> N_contexts; 
  int<lower=1> N_actions;  
  int<lower=1> N_states;   
  
  // Pre-Computed Physics (Animal-Specific)
  // Mapping: [Context, Animal][State, Action]
  array[N_contexts, N_animals] matrix[N_states, N_actions] Q_star; 
  matrix[N_contexts, 2] context_probs; 
  
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
  
  // Animal-Level Deviations
  vector[N_animals] beta_raw;
  vector[N_animals] kappa_raw;
  vector[N_animals] log_tau_raw;
}

transformed parameters {
  vector[N_animals] beta;
  vector[N_animals] kappa;
  vector<lower=0.01>[N_animals] tau; 
  
  beta = mu_beta + sigma_beta * beta_raw;
  kappa = mu_kappa + sigma_kappa * kappa_raw;
  tau = exp(mu_log_tau + sigma_log_tau * log_tau_raw) + 0.01;
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
  for (s in 1:N_sessions_total) {
    int animal = animal_map[s];
    vector[N_contexts] belief; 
    
    // Belief initialization: uniform at start of every session
    belief = rep_vector(1.0 / N_contexts, N_contexts);
    
    for (t in start_idx[s]:end_idx[s]) {
      int st = state_id[t];
      int act = action_id[t];
      int next_st = next_state_id[t];
      real w = weight[t];
      
      vector[N_actions] Q_values;
      
      // Calculate Information Bonus
      real H = 0;
      for (c in 1:N_contexts) {
        if(belief[c] > 1e-12) H -= belief[c] * log(belief[c]);
      }
      real current_info_bonus = kappa[animal] * H;

      for (a in 1:N_actions) {
        real q_ext = 0;
        for (c in 1:N_contexts) {
          q_ext += belief[c] * Q_star[c, animal, st, a];
        }
        
        real bonus_explore = (st == ID_IDLE && a == ID_WAIT) ? beta[animal] : 0;
        real bonus_info = (a == ID_LICK1 || a == ID_LICK2) ? current_info_bonus : 0;
        
        Q_values[a] = q_ext + bonus_explore + bonus_info;
      }
      
      target += w * categorical_logit_lpmf(act | Q_values / tau[animal]);
      
      // Belief Update (Bayesian Posterior)
      if (next_st == ID_REWARD_STATE || next_st == ID_NOREWARD_STATE) {
        vector[N_contexts] likelihoods;
        int spout_idx = (act == ID_LICK1) ? 1 : 2; 
        for (c in 1:N_contexts) {
          real p = context_probs[c, spout_idx];
          likelihoods[c] = (next_st == ID_REWARD_STATE) ? p : (1.0 - p);
        }
        
        belief = belief .* likelihoods;
        real sum_b = sum(belief);
        
        if (sum_b > 1e-15) {
            belief /= sum_b;
        } else {
            belief = rep_vector(1.0 / N_contexts, N_contexts);
        }
      }
    }
  }
}