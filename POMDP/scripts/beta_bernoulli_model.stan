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

// below STAN asks for block the compose the model
// basically it need a function to compute the likelihood
// a definition of the priors
// a definition of the data
// and some definitions to optimize likelihood computation
functions {
  real partial_log_lik(
      // these are just data indices
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
                       // mu parameters are global-level parameters
                       matrix mu_beta, matrix mu_kappa, matrix mu_phi, matrix mu_side,
                       matrix mu_beta_slope,
                       // these are the 'random effects'
                       matrix r_animal_beta, matrix r_animal_kappa, matrix r_animal_phi, matrix r_animal_beta_slope,
                       // this is the global 'pure' randomness parameter
                       real epsilon,
                       // possible states and actions
                       int N_actions, int N_states,
                       int ID_IDLE, int ID_WAIT, int ID_LICK1, int ID_LICK2,
                       int ID_REWARD_STATE, int ID_NOREWARD_STATE) {

    // init stuff

    // likelihood starts at 0
    real lp = 0;
    // this is the step size for data
    real dt = 0.025;
    // without any knowledge we start with a uniform dist
    // over the possible actions
    real log_random_prob = -log(N_actions * 1.0);

    // I called the 'random effects' 'traits'
    // as it is specifically modeled as something each particular
    // animal has as a starting point in its behavior
    for (i in 1:(end - start + 1)) {
      int animal_idx = animal_slice[i];
      // beta is the value of waiting
      real trait_b = r_animal_beta[animal_idx, 1];
      // kappa is the value of uncertainty-driven exploration
      real trait_k = r_animal_kappa[animal_idx, 1];
      // phi is motor perseveration or stickiness
      real trait_phi = r_animal_phi[animal_idx, 1];
      // b_slope is just the value of waiting over time
      // does it go up or down
      real trait_b_slope = r_animal_beta_slope[animal_idx, 1];

      for (s in sessions_per_animal_start[animal_idx]:sessions_per_animal_end[animal_idx]) {
        // a cognitive context is just the set of spout probabilities
        int cog_ctx = session_cognitive_context[s];
        // which drug was used 1: nothing, 2: vehicle, 3: tcs
        int d_idx = session_drug[s];
        // the pure randomness parameter is evaluated per session
        //real current_epsilon = epsilon[d_idx, cog_ctx];

        int last_lick_spout = 0;
        real current_wait_time = 0;

        // here I initialize the beta distribution for both spout
        // beta dist takes two parameter that basically move the mass
        // from left (0) to right (1)
        // with both parameter at 1, we get a uniform distribution
        // that represent no knowledge, then we can represent the
        // change in expectation by changing these params super nice
        // and fast.
        real alpha1 = 1.0;
        real beta1 = 1.0;
        real alpha2 = 1.0;
        real beta2 = 1.0;

        // any given animal is defined by its global parameter
        // plus its trait, super simple like main effect + (1 | ID)
        // sort of idea
        real b_s = mu_beta[d_idx, cog_ctx] + trait_b;
        real k_s = mu_kappa[d_idx, cog_ctx] + trait_k;
        real phi_s = mu_phi[d_idx, cog_ctx] + trait_phi;
        real side_s = mu_side[d_idx, cog_ctx];
        real b_slope = mu_beta_slope[d_idx, cog_ctx] + trait_b_slope;

        // hot loop begins here
        // this is where all the logic is
        // we get the indices for the start and end
        // of the session
        for (t in start_idx[s]:end_idx[s]) {
          // at any given step we look at what defines the model
          // the state where the animal is, the action it takes
          // where that action take the animal to, and the weight
          // for the wait time
          int st = state_id[t];
          int act = action_id[t];
          int next_st = next_state_id[t];
          int w = weight[t];

          // after computing the state of the world
          // here the expected value of each spout is computed
          // alpha are reward counts, beta are non-reward counts
          // nothing fancy here
          real ev1 = alpha1 / (alpha1 + beta1);
          real ev2 = alpha2 / (alpha2 + beta2);

          // just to avoid /0 errors
          real p1 = fmax(fmin(ev1, 0.999999), 0.000001);
          real p2 = fmax(fmin(ev2, 0.999999), 0.000001);

          // compute distribution shannon entropy
          // the flatter the more entropy
          real raw_u1 = -p1 * log(p1) - (1.0 - p1) * log(1.0 - p1);
          real raw_u2 = -p2 * log(p2) - (1.0 - p2) * log(1.0 - p2);

          // I left this here because instead of epsilon
          // I could've set a lower boundary for uncertainty
          // the 'irreducible' uncertainty. However, I decided for
          // the epsilon parameter instead, as (1) is explicitly estimated,
          // (2) easier to interpret, and nicer for the optimizer
          real u1 = raw_u1;
          real u2 = raw_u2;

          // below is the behavioral model

          // first, if the animal is IDLE it means it used the action WAIT
          // thus, the last lick spout is 0 (none) so it is not considered
          // for stickiness
          if (st == ID_IDLE) {
              last_lick_spout = 0;
          }

          // here the Q values vector is generated, the model has all the possible
          // actions, but nosepoke related one are disregarded as they are colineal
          // with spout related actions (animal are super trained)
          vector[N_actions] Q_base = rep_vector(0.0, N_actions);
          // here I used 5.0 to scale the reward, this is just for the model
          // to properly identify smaller differences in EV for both spout
          // given all the the other Q values, also here I added the side
          // bias, as its included only in one spout, a positive value
          // means left side pref, a negative right side pref
          Q_base[ID_LICK1] = (ev1 * 5.0) + side_s;
          Q_base[ID_LICK2] = (ev2 * 5.0);

          // now we get into the task logic more deeply
          // we need to know from where the animal is coming from

          // this check coming from doing nothing
          if (last_lick_spout == 0) {
              // if its coming from nothing it should get 'pulled'
              // by uncertainty-driven exploration by both spouts
              // so we add kappa + the actual shannon entropy of the
              // spouts
              Q_base[ID_LICK1] += k_s * u1;
              Q_base[ID_LICK2] += k_s * u2;
          }
          // this if the animal just licked spout 1
          else if (last_lick_spout == ID_LICK1) {
            // what drives the animal to lick the same spout
            // is the stickiness or phi param, whereas what drives it
            // to change spout is the uncertainty of the other spout
            Q_base[ID_LICK1] += phi_s;
            Q_base[ID_LICK2] += k_s * u2;
          }
          // this is the inverse case, everything is swapped
          else if (last_lick_spout == ID_LICK2) {
            Q_base[ID_LICK2] += phi_s;
            Q_base[ID_LICK1] += k_s * u1;
          }

          // here is where I added the 'continous' element to the model
          // the animal is not head-fixated it roams around a lot
          // so I modeled that as how valuable is for the animal to
          // roam around and do other stuff
          // the equation that defines this value is just
          // baseline wait value + the slope (is like how impatient the animal is)
          // * log(1 + t) which is the current time elapsed with an asymptote
          if (act == ID_WAIT) {
              // if its only the 'steps' waiting
              if (w <= 3) {
                  for (k in 1:w) {
                      vector[N_actions] Q_step = Q_base;
                      if (st == ID_IDLE) {
                          // compute the value of waiting that amount of time
                          real b_eff = b_s + b_slope * log1p(current_wait_time);
                          Q_step[ID_WAIT] += b_eff;
                      }
                      // this is the softmax for generating the log prob vector
                      // and return the likelihood of this particular action
                      // considering all the rest, this is almost the same
                      // as the rescorla wagner, but I removed the tau parameter
                      // as its now explictly modeled, and leads to issues of
                      // model non-identifiable parameters
                      real log_softmax = categorical_logit_lpmf(act | Q_step);
                      lp += log_mix(epsilon, log_random_prob, log_softmax);
                      // this part is to fix super weird results when both spouts are 100%
                      // when the animal does something we ask is it just random behavior
                      // or is some sort of calculation, log_mix takes both of those
                      // paths, current_epsilon is a mixing proportion here between 0 and 1
                      // log_random_prob represent the randomness over all actions,
                      // log_softmax is the softmax output, so now P(action) is
                      // epsilon * P(random) + (1 - epsilon) * P(softmax output)
                      // the function does that avoiding all the issues of working
                      // with logs, also now epsilon is 'true' randomness completely
                      // independent of everything, this allows the model to consider
                      // spout switching in the 100/100 context as epsilon related rather
                      // than a stupid high kappa value
                      //lp += log_mix(current_epsilon, log_random_prob, log_softmax);\
                      //lp += categorical_logit_lpmf(act | Q_step);
                      // we move a step and re-do
                      current_wait_time += dt;
                  }
              } else {
                  // this is a trick I found to avoid the insane computation above
                  // if the animal waited a lot of time, just for consideration
                  // 2 seconds are like ~80 steps

                  // we estimate this getting the start mid and end point
                  // from the weight (# steps waiting)
                  real t_start = current_wait_time;
                  real t_end   = current_wait_time + (w - 1) * dt;
                  real t_mid   = current_wait_time + ((w - 1) / 2.0) * dt;

                  // now we generate three copies of the Q values
                  // basically how thing are going to be at the start, mid and end
                  // of the wait period
                  vector[N_actions] Q_start = Q_base;
                  vector[N_actions] Q_mid   = Q_base;
                  vector[N_actions] Q_end   = Q_base;

                  if (st == ID_IDLE) {
                      // the reasoning here is that as time goes by
                      // the value of waiting is likely to change
                      Q_start[ID_WAIT] += (b_s + b_slope * log1p(t_start));
                      Q_mid[ID_WAIT]   += (b_s + b_slope * log1p(t_mid));
                      Q_end[ID_WAIT]   += (b_s + b_slope * log1p(t_end));
                  }

                  // given our q-values we get the likelihood at each of the
                  // three time points
                  // in other words, how likely was that the animal decided
                  // to wait at these three time points given the particular
                  // parametrization
                  real lp_start = log_mix(epsilon, log_random_prob, categorical_logit_lpmf(act | Q_start));
                  real lp_mid   = log_mix(epsilon, log_random_prob, categorical_logit_lpmf(act | Q_mid));
                  real lp_end   = log_mix(epsilon, log_random_prob, categorical_logit_lpmf(act | Q_end));
                  // real lp_start = categorical_logit_lpmf(act | Q_start);
                  // real lp_mid = categorical_logit_lpmf(act | Q_mid);
                  // real lp_end = categorical_logit_lpmf(act | Q_end);

                  // calculus integration rule (simpson's)
                  // basically we need to get the area (likelihood) under the curve
                  // composed of three points, and this quadratic polynomial is the optimal
                  // to approximate the area under the curve, calculus magic saves LOTS of time
                  lp += w * (lp_start + 4 * lp_mid + lp_end) / 6.0;
                  // now we move all the steps indicated by the weight
                  current_wait_time += w * dt;
              }
          }
          else {
              // act != ID_WAIT
              // so now the animal is going to something else besides waiting
              if (st == ID_IDLE) {
                  // this computation is now saying after all this wait
                  // what's the actual value of waiting
                  real b_eff = b_s + b_slope * log1p(current_wait_time);
                  Q_base[ID_WAIT] += b_eff;
              }
              // and this part compares such value to the others,
              // in other words we know that the animal prefered something else
              // to wait, so the alternative that won over waiting is likely
              // to be highly valued
              real log_softmax = categorical_logit_lpmf(act | Q_base);
              lp += w * log_mix(epsilon, log_random_prob, log_softmax);
              // lp += w * categorical_logit_lpmf(act | Q_base);
              // set the clock 0 the animal is now not-waiting
              current_wait_time = 0;
          }

          // this is to give the licking 'trains' continuity
          // otherwise waiting would break the phi parameter
          if (act == ID_LICK1 || act == ID_LICK2) {
              last_lick_spout = act;
          }

          // this is the learning part
          if (next_st == ID_REWARD_STATE || next_st == ID_NOREWARD_STATE) {
              // if a reward was triggered the 'trial' ends
            last_lick_spout = 0;
            if (next_st == ID_REWARD_STATE) {
                // if a reward was obtained
                // update the beta dist of the spout that gave
                // the rewards
                if (act == ID_LICK1) alpha1 += 1.0;
                else if (act == ID_LICK2) alpha2 += 1.0;
            } else {
                // if no reward was given then update the beta dist
                // to reflect that by adding to the beta parameter
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
    // data part is just that, all the data related definitions
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
  // grainsize is the length of the cue for the processor
  int grainsize;
}

parameters {
  // BASELINE
  real base_beta;
  real<lower=0> base_kappa;
  real<lower=0> base_phi;
  real base_side;
  real base_beta_slope;
  // randomness per drug per context
  real<lower=0, upper=1> epsilon;

  // VEHICLE SHIFT effect of context + veh over baseline
  // we need this because we included baseline data, so we model
  // baseline -> vehicle, to get the potential effect of injection alone
  // and we consider that delta our effect in vehicle
  vector[N_cognitive_contexts] veh_shift_beta;
  vector[N_cognitive_contexts] veh_shift_kappa;
  vector[N_cognitive_contexts] veh_shift_phi;
  vector[N_cognitive_contexts] veh_shift_side;
  vector[N_cognitive_contexts] veh_shift_beta_slope;

  // DRUG DELTA
  // tcs - veh
  vector[N_cognitive_contexts] drug_delta_beta;
  vector[N_cognitive_contexts] drug_delta_kappa;
  vector[N_cognitive_contexts] drug_delta_phi;
  vector[N_cognitive_contexts] drug_delta_side;
  vector[N_cognitive_contexts] drug_delta_beta_slope;

  // animal-level traits
  // this sets the spread of the random effects
  real<lower=0> sigma_beta_trait;
  real<lower=0> sigma_kappa_trait;
  real<lower=0> sigma_phi_trait;
  real<lower=0> sigma_beta_slope_trait;

  // this is used afterwards to compute the random effect
  // as a deviation from the global parameter
  // rather than estimating every single animal by itself
  vector[N_animals] beta_trait_raw;
  vector[N_animals] kappa_trait_raw;
  vector[N_animals] phi_trait_raw;
  vector[N_animals] beta_slope_trait_raw;
}

transformed parameters {
  // here I set everything to match the experimental design
  // with the statistical model
  // we need each parameter per drug per context
  matrix[N_drugs, N_cognitive_contexts] mu_beta;
  matrix[N_drugs, N_cognitive_contexts] mu_kappa;
  matrix[N_drugs, N_cognitive_contexts] mu_phi;
  matrix[N_drugs, N_cognitive_contexts] mu_side;
  matrix[N_drugs, N_cognitive_contexts] mu_beta_slope;

  // set as a matrix if we wanted to test interactions at some point
  // for now they are just the vectors computing the per animal estimate
  // now our model is basically additive kappa + beta + phi + ...
  matrix[N_animals, 1] r_animal_beta = to_matrix(sigma_beta_trait * beta_trait_raw, N_animals, 1);
  matrix[N_animals, 1] r_animal_kappa = to_matrix(sigma_kappa_trait * kappa_trait_raw, N_animals, 1);
  matrix[N_animals, 1] r_animal_phi = to_matrix(sigma_phi_trait * phi_trait_raw, N_animals, 1);
  matrix[N_animals, 1] r_animal_beta_slope = to_matrix(sigma_beta_slope_trait * beta_slope_trait_raw, N_animals, 1);

  for (c in 1:N_cognitive_contexts) {
    // baseline remains constant
    mu_beta[1, c]         = base_beta;
    mu_kappa[1, c]        = base_kappa;
    mu_phi[1, c]          = base_phi;
    mu_side[1, c]         = base_side;
    mu_beta_slope[1, c]   = base_beta_slope;

    // veh is baseline + the shift from injection
    mu_beta[2, c]         = base_beta       + veh_shift_beta[c];
    mu_kappa[2, c]        = base_kappa      + veh_shift_kappa[c];
    mu_phi[2, c]          = base_phi        + veh_shift_phi[c];
    mu_side[2, c]         = base_side       + veh_shift_side[c];
    mu_beta_slope[2, c]   = base_beta_slope + veh_shift_beta_slope[c];

    // tcs is is (baseline + the shift) + the effect of tcs over veh
    mu_beta[3, c]         = mu_beta[2, c]       + drug_delta_beta[c];
    mu_kappa[3, c]        = mu_kappa[2, c]      + drug_delta_kappa[c];
    mu_phi[3, c]          = mu_phi[2, c]        + drug_delta_phi[c];
    mu_side[3, c]         = mu_side[2, c]       + drug_delta_side[c];
    mu_beta_slope[3, c]   = mu_beta_slope[2, c] + drug_delta_beta_slope[c];
  }
}

model {
  // these are the priors
  // the main idea of priors is to represent only the logical
  // boundaries of your parameters, not to include any particular
  // information
  // these are reasonable starting points, deviation cant be to big
  // otherwise softmax can easily break
  array[N_cognitive_contexts] real expected_u = {0.001, 0.346, 0.627};
  base_beta       ~ normal(0, 1.5);
  base_phi        ~ normal(0, 1.5);
  base_side       ~ normal(0, 1.5);
  base_beta_slope ~ normal(0, 1.5);
  base_kappa      ~ normal(0, 1.5);

  // adding the global epsilon
  epsilon ~ beta(1, 1);

  // these are the delta priors
  // similar to a null hypothesis we assume centered at 0
  // kappa needs a slightly higher deviation because
  // kappa is multiplied by uncertainty (k x U)
  // top U = 0.69 for this dist, so assume a U of 0.25
  // to make a pull of +1.0 (like phi could as its additive)
  // kappa would need to be 4
  veh_shift_beta       ~ normal(0, 1.0);
  // veh_shift_kappa      ~ normal(0, 1.5);
  veh_shift_phi        ~ normal(0, 1.0);
  veh_shift_side       ~ normal(0, 1.0);
  veh_shift_beta_slope ~ normal(0, 1.0);

  drug_delta_beta       ~ normal(0, 1.0);
  // drug_delta_kappa      ~ normal(0, 1.5);
  drug_delta_phi        ~ normal(0, 1.0);
  drug_delta_side       ~ normal(0, 1.0);
  drug_delta_beta_slope ~ normal(0, 1.0);

 // kappa is estimated in proportion of the average entropy
 // of the context, the intuition is straightforward in a 100/100
 // there's 0 entropy thus kappa can't be logically relevant
 // so its dragged towards 0, in the 100/50 it can exists
 // but logically 0.5/0.25 > 100/50 if kappa matters at all
  for (c in 1:N_cognitive_contexts){
      veh_shift_kappa[c] ~ normal(0, 2.5 * expected_u[c]);
      drug_delta_kappa[c] ~ normal(0, 2.5 * expected_u[c]);
  }

  // trait priors are just 0, 1 because of non-centered parametrization
  // nothing special needs to be done here
  beta_trait_raw  ~ std_normal();
  kappa_trait_raw ~ std_normal();
  phi_trait_raw   ~ std_normal();
  beta_slope_trait_raw ~ std_normal();

  sigma_beta_trait  ~ normal(0, 0.5);
  sigma_phi_trait   ~ normal(0, 0.5);
  sigma_beta_slope_trait ~ normal(0, 0.5);
  sigma_kappa_trait ~ normal(0, 0.5);

  array[N_animals] int animal_indices;
  for (i in 1:N_animals) animal_indices[i] = i;
  // likelihoods are computed separatedly this function deals
  // with adding them together and talking with the CPU
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