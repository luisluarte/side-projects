data {
  int<lower=1> N_steps;
  real<lower=0, upper=1> p_reward_1;
  real<lower=0, upper=1> p_reward_2;
}

generated quantities {
  // draw params from the priors
  real prior_beta = normal_rng(10, 1.5);
  real prior_beta_slope = normal_rng(0, 1.5);
  real prior_kappa = abs(normal_rng(0, 1.5));
  real prior_phi = normal_rng(0, 1.5);
  real prior_side = normal_rng(0, 1.5);
  real epsilon = beta_rng(1, 10);
  
  // data to save
  int total_licks_1 = 0;
  int total_licks_2 = 0;
  int total_switches = 0;
  int total_ghost_actions = 0;
  
  // sim
  {
    // set init vars
    real alpha1 = 1.0; real beta1 = 1.0;
    real alpha2 = 1.0; real beta2 = 1.0;
    real current_wait_time = 0.0;
    int last_lick_spout = 0;
    
    // start the session
    for (t in 1:N_steps) {
      
      // compute bounded EV
      real ev1 = fmax(fmin(alpha1 / (alpha1 + beta1), 0.999999), 0.000001);
      real ev2 = fmax(fmin(alpha2 / (alpha2 + beta2), 0.999999), 0.000001);
      
      // compute shannon entropy
      real u1 = -ev1 * log(ev1) - (1 - ev1) * log(1 - ev1);
      real u2 = -ev2 * log(ev2) - (1 - ev2) * log(1 - ev2);
      
      // compute q values
      vector[6] Q = rep_vector(0.0, 6);
      Q[1] = prior_beta + prior_beta_slope * log1p(current_wait_time);
      Q[5] = ev1 + prior_side;
      Q[6] = ev2;
      
      // update q values
      if (last_lick_spout == 0) {
        Q[5] += prior_kappa * u1;
        Q[6] += prior_kappa * u2;
      } else if (last_lick_spout == 1) {
        Q[5] += prior_phi;
        Q[6] += prior_kappa * u2;
      } else if (last_lick_spout == 2) {
        Q[5] += prior_kappa * u1;
        Q[6] += prior_phi;
      }
      
      // softmax and epsilon prob vector
      vector[6] p_soft = softmax(Q);
      vector[6] p_final = epsilon * (1.0/6.0) + (1.0 - epsilon) * p_soft;
      
      // simulate an action
      int action = categorical_rng(p_final);
      
      // add the step
      if (action == 1) {
        current_wait_time += 0.025;
      } else if (action >= 2 && action <= 4) {
        current_wait_time = 0.0;
        total_ghost_actions += 1;
      } else {
        int spout = (action == 5) ? 1 : 2;
        
        // save actions
        if (spout == 1) total_licks_1 += 1;
        if (spout == 2) total_licks_2 += 1;
        
        if (last_lick_spout != 0 && last_lick_spout != spout) {
          total_switches += 1;
        }
        
        // simulate rewards
        int is_rewarded = 0;
        if (spout == 1) is_rewarded = bernoulli_rng(p_reward_1);
        if (spout == 2) is_rewarded = bernoulli_rng(p_reward_2);
        
        // update internal beta dist
        if (is_rewarded == 1) {
          if (spout == 1) alpha1 += 1.0; else alpha2 += 1.0;
        } else {
          if (spout == 1) beta1 += 1.0; else beta2 += 1.0;
        }
        
        // reset timer
        current_wait_time = 0.0;
        last_lick_spout = spout;
      }
    }
  }
}