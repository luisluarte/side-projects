// this a hierarchical bayesian reinforcement learning model
// taking the same principles from the rescorla-wagner, while
// increasing complexity a bit to increase the ability to 'catch'
// the potential of uncertainty driven exploration in a more direct
// way.

// this first part deal with optimization-related stuff
// the bayesian sampler works basically by making a proposal 'step'
// for any given parameter and then looking at the likelihood of such
// step given the data. The native sampler would look over all animals
// to get the likelihood, that's very inefficient, so I made it compute
// the partial likelihood, making it able to generate a parallel process
// so I can get on my ryzen about ~2 animals per thread or so

// the data I feed to the model is the lickometer data in chunks of
// 25 ms, this is a very simple way to discretize data without lossing any detail
// licks, rewards and so on are single event per chunk, however, for waiting periods
// I needed to make some changes because otherwise the model would looking at a huge
// amount of 'waiting' without any reason, because of that waiting periods are rolled
// together and assigned a wait for how long the animal waited, for estimatation of
// parameter that compute the slope of the value of waiting I used a simple 3 point
// estimate using the init, mid and final values, to reduce computational workload
functions {
  real partial_log_lik(
      // this ones are just indices for the data
      array[] int animal_slice,
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
                       // here we start with the global level parameters
                       matrix mu_beta, matrix mu_kappa, matrix mu_phi, matrix mu_side,
                       matrix mu_eta_pos, matrix mu_eta_neg, matrix mu_beta_slope,
                       // the 'random effects' are here
                       matrix r_animal_beta, matrix r_animal_kappa, matrix r_animal_phi, matrix r_animal_beta_slope,
                       array[,] matrix Q_star,
                       matrix context_probs,
                       real belief_diffusion,
                       int N_physics_contexts, int N_actions, int N_states,
                       int ID_IDLE, int ID_WAIT, int ID_LICK1, int ID_LICK2,
                       int ID_REWARD_STATE, int ID_NOREWARD_STATE) {

    real lp = 0;
    real dt = 0.025;

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
      real trait_b_slope = r_animal_beta_slope[animal_idx, 1];


      for (s in sessions_per_animal_start[animal_idx]:sessions_per_animal_end[animal_idx]) {
        int cog_ctx = session_cognitive_context[s];
        int d_idx = session_drug[s];
        int prev_act = 0;
        int last_lick_spout = 0;
        real current_wait_time = 0;

        belief = (1.0 - belief_diffusion) * belief + belief_diffusion * uniform_prior;

        H = 0;
        //for (c in 1:N_physics_contexts) if (belief[c] > 1e-12) H -= belief[c] * log(belief[c]);
        for (c in 1:N_physics_contexts){
          real safe_b = belief[c] + 1e-12;
          H -= safe_b * log(safe_b);
        }

        //real b_s = mu_beta[d_idx, cog_ctx] + trait_b + sigma_beta_session[cog_ctx] * beta_session_raw[s];
        //real k_s = mu_kappa[d_idx, cog_ctx] + trait_k + sigma_kappa_session[cog_ctx] * kappa_session_raw[s];
        real b_s = mu_beta[d_idx, cog_ctx] + trait_b;
        real k_s = mu_kappa[d_idx, cog_ctx] + trait_k;
        real phi_s = mu_phi[d_idx, cog_ctx] + trait_phi;
        real side_s = mu_side[d_idx, cog_ctx];
        real b_slope = mu_beta_slope[d_idx, cog_ctx] + trait_b_slope;

        real ks_H = k_s * H;

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
          int w = weight[t];


          // 1. Calculate the static components ONLY ONCE per compressed block
          vector[N_actions] Q_base = Q_star[animal_idx, st] * belief;

          if (st == ID_IDLE) {
              last_lick_spout = 0;
          }

          Q_base[ID_LICK1] += ks_H + side_s;
          Q_base[ID_LICK2] += ks_H;

          if (last_lick_spout != 0) {
              Q_base[last_lick_spout] += phi_s;
          }

          // 2. Branch: If it's a WAIT, approximate the time-gradient curve
          if (act == ID_WAIT) {
              if (w <= 3) {
                  // For very short waits, exact step-by-step is faster
                  for (k in 1:w) {
                      vector[N_actions] Q_step = Q_base;
                      if (st == ID_IDLE) {
                          real b_eff = b_s + b_slope * log1p(current_wait_time);
                          Q_step[ID_WAIT] += (b_eff * dt);
                      }
                      lp += categorical_logit_lpmf(act | Q_step);
                      current_wait_time += dt;
                  }
              } else {
                  // SIMPSON'S RULE: O(1) Approximation for long waits
                  real t_start = current_wait_time;
                  real t_end   = current_wait_time + (w - 1) * dt;
                  real t_mid   = current_wait_time + ((w - 1) / 2.0) * dt;

                  vector[N_actions] Q_start = Q_base;
                  vector[N_actions] Q_mid   = Q_base;
                  vector[N_actions] Q_end   = Q_base;

                  if (st == ID_IDLE) {
                      Q_start[ID_WAIT] += (b_s + b_slope * log1p(t_start)) * dt;
                      Q_mid[ID_WAIT]   += (b_s + b_slope * log1p(t_mid))   * dt;
                      Q_end[ID_WAIT]   += (b_s + b_slope * log1p(t_end))   * dt;
                  }

                  real lp_start = categorical_logit_lpmf(act | Q_start);
                  real lp_mid   = categorical_logit_lpmf(act | Q_mid);
                  real lp_end   = categorical_logit_lpmf(act | Q_end);

                  // Multiply by weight and apply Simpson's 1/3 weighting
                  lp += w * (lp_start + 4 * lp_mid + lp_end) / 6.0;

                  // Advance the clock by the total block time
                  current_wait_time += w * dt;
              }
          }
          // 3. Branch: If it's a LICK (or anything else), evaluate once and multiply
          else {
              if (st == ID_IDLE) {
                  real b_eff = b_s + b_slope * log1p(current_wait_time);
                  Q_base[ID_WAIT] += (b_eff * dt);
              }

              lp += w * categorical_logit_lpmf(act | Q_base);
              current_wait_time = 0; // Reset wait timer since they acted
          }

          // 4. Update memory trackers
          prev_act = act;
          if (act == ID_LICK1 || act == ID_LICK2) {
              last_lick_spout = act;
          }

          if (next_st == ID_REWARD_STATE || next_st == ID_NOREWARD_STATE) {
            last_lick_spout = 0;
            int spout_idx = (act == ID_LICK1) ? 1 : 2;

            // =========================================================================
            // FAST LOOKUP: O(1) vector assignment instead of N transcendental functions
            // =========================================================================
            if (next_st == ID_REWARD_STATE) {
               for(c in 1:N_physics_contexts) belief[c] *= lik_reward[c, spout_idx];
            } else {
               for(c in 1:N_physics_contexts) belief[c] *= lik_noreward[c, spout_idx];
            }

            for(c in 1:N_physics_contexts) belief[c] += 1e-30;

            real sum_b = sum(belief);
            if (sum_b > 1e-9) belief /= sum_b;
            else belief = uniform_prior;

            H = 0;
            //for (c in 1:N_physics_contexts) if (belief[c] > 1e-9) H -= belief[c] * log(belief[c]);
            for (c in 1:N_physics_contexts){
              real safe_b = fmax(belief[c], 1e-9);
              H -= safe_b * log(safe_b);
            }
            ks_H = k_s * H;
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
  array[N_animals, N_states] matrix[N_actions, N_physics_contexts] Q_star;

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
  real<lower=0> sigma_beta_slope_trait;

  vector[N_animals] beta_trait_raw;
  vector[N_animals] kappa_trait_raw;
  vector[N_animals] phi_trait_raw;
  vector[N_animals] beta_slope_trait_raw;

  //vector<lower=1e-6>[N_cognitive_contexts] sigma_beta_session;
  //vector<lower=1e-6>[N_cognitive_contexts] sigma_kappa_session;

  //vector[N_sessions_total] beta_session_raw;
  //vector[N_sessions_total] kappa_session_raw;

  real<lower=0, upper=1> belief_diffusion;
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
  matrix[N_animals, 1] r_animal_beta_slope = to_matrix(sigma_beta_slope_trait * beta_slope_trait_raw, N_animals, 1);

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
  base_beta       ~ normal(0, 1.0);
  base_phi        ~ normal(0, 1.0);
  base_side       ~ normal(0, 1.0);
  base_beta_slope ~ normal(0, 1.0);

  base_kappa      ~ normal(0, 0.35);
  base_eta_pos    ~ normal(0, 0.35);
  base_eta_neg    ~ normal(0, 0.35);

  // Priors: Vehicle Shifts (Context/Injection Effects)
  veh_shift_beta       ~ normal(0, 0.5);
  veh_shift_kappa      ~ normal(0, 0.5);
  veh_shift_phi        ~ normal(0, 0.5);
  veh_shift_side       ~ normal(0, 0.5);
  veh_shift_beta_slope ~ normal(0, 0.5);
  veh_shift_eta_pos    ~ normal(0, 0.5);
  veh_shift_eta_neg    ~ normal(0, 0.5);

  drug_delta_beta       ~ normal(0, 0.5);
  drug_delta_kappa      ~ normal(0, 0.5);
  drug_delta_phi        ~ normal(0, 0.5);
  drug_delta_side       ~ normal(0, 0.5);
  drug_delta_beta_slope ~ normal(0, 0.5);
  drug_delta_eta_pos    ~ normal(0, 0.5);
  drug_delta_eta_neg    ~ normal(0, 0.5);

  beta_trait_raw  ~ std_normal();
  kappa_trait_raw ~ std_normal();
  phi_trait_raw   ~ std_normal();
  beta_slope_trait_raw ~ std_normal();
  sigma_beta_trait  ~ normal(0, 1.0);
  sigma_phi_trait   ~ normal(0, 1.0);
  sigma_beta_slope_trait ~ normal(0, 1.0);
  sigma_kappa_trait ~ normal(0, 0.1);

  //sigma_beta_session  ~ normal(0, 2.0);
  //sigma_kappa_session ~ normal(0, 1.0);

  //beta_session_raw  ~ std_normal();
  //kappa_session_raw ~ std_normal();

  belief_diffusion ~ beta(100, 1);

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
                       //sigma_beta_session, sigma_kappa_session,
                       r_animal_beta, r_animal_kappa, r_animal_phi, r_animal_beta_slope,
                       //beta_session_raw, kappa_session_raw,
                       Q_star, context_probs,
                       belief_diffusion,
                       N_physics_contexts, N_actions, N_states,
                       ID_IDLE, ID_WAIT, ID_LICK1, ID_LICK2,
                       ID_REWARD_STATE, ID_NOREWARD_STATE);
}
