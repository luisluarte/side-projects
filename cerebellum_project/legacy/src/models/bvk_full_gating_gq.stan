functions {
  // Exact Analytical LTI Propagator for the Mossy Fiber Cascade
  vector exact_mf_step(real dt, vector mf, real tau_m, real I_drive, int N_MF) {
    vector[N_MF] mf_next;
    vector[N_MF] d = mf - I_drive;
    real x = dt / tau_m;
    real decay = exp(-x);

    vector[N_MF] w;
    w[1] = 1.0;
    for (i in 2:N_MF) {
      w[i] = w[i-1] * x / (i - 1.0);
    }

    for (k in 1:N_MF) {
      real conv_sum = 0.0;
      for (j in 1:k) {
        conv_sum += d[k - j + 1] * w[j];
      }
      mf_next[k] = I_drive + conv_sum * decay;
    }
    return mf_next;
  }

  // Fully Kernelized Partial Sum (Zero Matrices, Zero Spatial Loops)
  real partial_sum(
    array[] int seq_subj_slice, int start, int end,
    data array[] int choice, data array[] real rt, data array[] real reward,
    data array[] real iti, data array[] real f_dur,
    data array[] int start_idx, data array[] int end_idx,
    data int N_MF,
    vector alpha_ctx, vector tau_m, vector eta_gc, vector lambda_gc,
    vector theta_cb, vector kappa_ctx, vector gamma_suppress, vector a, vector tau_nd, vector beta_a, vector kappa_cb
  ) {
    real target_sum = 0.0;

    for (s_idx in 1:size(seq_subj_slice)) {
      int s = seq_subj_slice[s_idx];
      int start_t = start_idx[s];
      int end_t = end_idx[s];
      int n_trials = end_t - start_t + 1;

      vector[2] Q_ctx = rep_vector(0.5, 2);
      vector[N_MF] mf_state = rep_vector(0.0, N_MF);

      vector[N_MF] w_gc1 = rep_vector(0.0, N_MF);
      vector[N_MF] w_gc2 = rep_vector(0.0, N_MF);
      vector[N_MF] w_mli1 = rep_vector(0.0, N_MF);
      vector[N_MF] w_mli2 = rep_vector(0.0, N_MF);

      array[n_trials] real rt_subj;
      array[n_trials] real w_bias_subj;
      array[n_trials] real a_subj;
      array[n_trials] real v_subj;

      // Hoist subject-level parameters to avoid array indexing and redundant ops in the loop
      real a_s = a[s];
      real tau_nd_s = tau_nd[s];
      real theta_cb_s = theta_cb[s];
      real kappa_ctx_s = kappa_ctx[s];
      real gamma_suppress_s = gamma_suppress[s];
      real beta_a_s = beta_a[s];
      real kappa_cb_s = kappa_cb[s];
      real alpha_ctx_s = alpha_ctx[s];
      real tau_m_s = tau_m[s];
      
      real l_gc_eff = lambda_gc[s] + 1e-8;
      real l_mli_eff = (lambda_gc[s] * 1.5) + 1e-8;
      real eta_gc_s = eta_gc[s];
      real eta_mli_s = eta_gc[s];
      
      real inv_l_gc_eff = 1.0 / l_gc_eff;
      real inv_l_mli_eff = 1.0 / l_mli_eff;

      for (t in start_t:end_t) {
        int idx = t - start_t + 1;
        rt_subj[idx] = rt[t];

        // -----------------------------------------------------
        // Phase 1: Inter-Trial Interval
        // -----------------------------------------------------
        if (iti[t] > 0.01) {
          mf_state = exact_mf_step(iti[t], mf_state, tau_m_s, 0.0, N_MF);

          real decay_gc_iti = exp(-lambda_gc[s] * iti[t]);
          real decay_mli_iti = exp(-(lambda_gc[s] * 1.5) * iti[t]);

          w_gc1 *= decay_gc_iti;
          w_gc2 *= decay_gc_iti;
          w_mli1 *= decay_mli_iti;
          w_mli2 *= decay_mli_iti;
        }

        // -----------------------------------------------------
        // Phase 2: Dual Formulation Readout
        // -----------------------------------------------------
        real Q_cb_1 = dot_product(w_gc1, mf_state) - dot_product(w_mli1, mf_state);
        real Q_cb_2 = dot_product(w_gc2, mf_state) - dot_product(w_mli2, mf_state);

        real delta_Q_ctx = Q_ctx[2] - Q_ctx[1];
        real delta_Q_cb = Q_cb_2 - Q_cb_1;

        real w_bias = 0.5 + 0.45 * tanh(theta_cb_s * delta_Q_cb);
        real conflict = 0.5 * (1.0 - tanh(10.0 * delta_Q_ctx) * tanh(10.0 * delta_Q_cb));
        real v_base = kappa_ctx_s * delta_Q_ctx + kappa_cb_s * delta_Q_cb;
        real v_effective = v_base * exp(-gamma_suppress_s * conflict);

        // Strict Drift Lower Bound (prevents Wiener density from shattering at 0)
        if (abs(v_effective) < 1e-4) {
          v_effective = v_effective >= 0 ? 1e-4 : -1e-4;
        }

        real a_effective = a_s + beta_a_s * conflict;

        if (choice[t] == 1) {
          w_bias_subj[idx] = w_bias;
          v_subj[idx] = v_effective;
          a_subj[idx] = a_effective;
        } else {
          w_bias_subj[idx] = 1.0 - w_bias;
          v_subj[idx] = -v_effective;
          a_subj[idx] = a_effective;
        }

        // -----------------------------------------------------
        // Phase 3: Analytical Plasticity in Dual Space
        // -----------------------------------------------------
        real RPE_ctx = reward[t] - Q_ctx[choice[t] + 1];
        Q_ctx[choice[t] + 1] += alpha_ctx_s * RPE_ctx;

        real cb_pred = (choice[t] == 1) ? Q_cb_2 : Q_cb_1;
        real RPE_cb = reward[t] - cb_pred;

        real E_cb1 = (choice[t] == 0) ? RPE_cb : 0.0;
        real E_cb2 = (choice[t] == 1) ? RPE_cb : 0.0;

        if (f_dur[t] > 0.01) {
          mf_state = exact_mf_step(f_dur[t], mf_state, tau_m_s, reward[t], N_MF);

          real decay_gc_f = exp(-l_gc_eff * f_dur[t]);
          real decay_mli_f = exp(-l_mli_eff * f_dur[t]);

          real int_gc_f = (1.0 - decay_gc_f) * inv_l_gc_eff;
          real int_mli_f = (1.0 - decay_mli_f) * inv_l_mli_eff;

          real scale_gc1 = eta_gc_s * E_cb1 * int_gc_f;
          real scale_gc2 = eta_gc_s * E_cb2 * int_gc_f;
          real scale_mli1 = -eta_mli_s * E_cb1 * int_mli_f;
          real scale_mli2 = -eta_mli_s * E_cb2 * int_mli_f;

          w_gc1 = w_gc1 * decay_gc_f + mf_state * scale_gc1;
          w_gc2 = w_gc2 * decay_gc_f + mf_state * scale_gc2;
          w_mli1 = w_mli1 * decay_mli_f + mf_state * scale_mli1;
          w_mli2 = w_mli2 * decay_mli_f + mf_state * scale_mli2;
        }
      }

      for (idx in 1:n_trials) {
        target_sum += wiener_lpdf(rt_subj[idx] | a_subj[idx], tau_nd[s], w_bias_subj[idx], v_subj[idx]);
      }
    }
    return target_sum;
  }
}

data {
  int<lower=1> N;
  int<lower=1> S;
  array[S] int<lower=1> start_idx;
  array[S] int<lower=1> end_idx;

  array[N] int<lower=0, upper=1> choice;
  array[N] real<lower=0> rt;
  array[N] real<lower=0, upper=1> reward;
  array[N] real<lower=0> iti;
  array[N] real<lower=0> f_dur;
  array[S] real<lower=0> min_rt;

  int<lower=1> N_MF;
  int<lower=1> grainsize;
}

transformed data {
  array[S] int seq_subj;
  for (s in 1:S) seq_subj[s] = s;
}

parameters {
  real mu_alpha_ctx;
  real mu_tau_m;
  real mu_eta_gc;
  real mu_lambda_gc;
  real mu_theta_cb;
  real mu_kappa_ctx;
  real mu_gamma_suppress;
  real mu_a;
  real mu_beta_a;
  real mu_kappa_cb;
  real mu_tau_nd;

  real<lower=0> sigma_alpha_ctx;
  real<lower=0> sigma_tau_m;
  real<lower=0> sigma_eta_gc;
  real<lower=0> sigma_lambda_gc;
  real<lower=0> sigma_theta_cb;
  real<lower=0> sigma_kappa_ctx;
  real<lower=0> sigma_gamma_suppress;
  real<lower=0> sigma_a;
  real<lower=0> sigma_beta_a;
  real<lower=0> sigma_kappa_cb;
  real<lower=0> sigma_tau_nd;

  vector[S] z_alpha_ctx;
  vector[S] z_tau_m;
  vector[S] z_eta_gc;
  vector[S] z_lambda_gc;
  vector[S] z_theta_cb;
  vector[S] z_kappa_ctx;
  vector[S] z_gamma_suppress;
  vector[S] z_a;
  vector[S] z_beta_a;
  vector[S] z_kappa_cb;
  vector[S] z_tau_nd;
}

transformed parameters {
  vector[S] alpha_ctx = inv_logit(mu_alpha_ctx + sigma_alpha_ctx * z_alpha_ctx);
  vector[S] tau_m = 1.0 + 9.0 * inv_logit(mu_tau_m + sigma_tau_m * z_tau_m);
  vector[S] eta_gc = 0.5 + 4.5 * inv_logit(mu_eta_gc + sigma_eta_gc * z_eta_gc);
  vector[S] lambda_gc = 5.0 * inv_logit(mu_lambda_gc + sigma_lambda_gc * z_lambda_gc);
  vector[S] theta_cb = 10.0 * inv_logit(mu_theta_cb + sigma_theta_cb * z_theta_cb);
  vector[S] kappa_ctx = 10.0 * inv_logit(mu_kappa_ctx + sigma_kappa_ctx * z_kappa_ctx);
  vector[S] gamma_suppress = 10.0 * inv_logit(mu_gamma_suppress + sigma_gamma_suppress * z_gamma_suppress);

  vector[S] a = 0.5 + 4.5 * inv_logit(mu_a + sigma_a * z_a);
  vector[S] beta_a = 5.0 * inv_logit(mu_beta_a + sigma_beta_a * z_beta_a);
  vector[S] kappa_cb = 10.0 * inv_logit(mu_kappa_cb + sigma_kappa_cb * z_kappa_cb);
  vector[S] tau_nd = 0.001 + (to_vector(min_rt) - 0.002) .* inv_logit(mu_tau_nd + sigma_tau_nd * z_tau_nd);
}

model {
  mu_alpha_ctx ~ normal(0, 1.5);
  mu_tau_m ~ normal(0, 1.5);
  mu_eta_gc ~ normal(0, 1.5);
  mu_lambda_gc ~ normal(0, 1.5);
  mu_theta_cb ~ normal(0, 1.5);
  mu_kappa_ctx ~ normal(0, 1.5);
  mu_gamma_suppress ~ normal(0, 1.5);
  mu_a ~ normal(0, 1);
  mu_beta_a ~ normal(0, 1.5);
  mu_kappa_cb ~ normal(0, 1.5);
  mu_tau_nd ~ normal(-1, 1);

  sigma_alpha_ctx ~ normal(0, 1);
  sigma_tau_m ~ normal(0, 1);
  sigma_eta_gc ~ normal(0, 1);
  sigma_lambda_gc ~ normal(0, 1);
  sigma_theta_cb ~ normal(0, 1);
  sigma_kappa_ctx ~ normal(0, 1);
  sigma_gamma_suppress ~ normal(0, 1);
  sigma_a ~ normal(0, 1);
  sigma_beta_a ~ normal(0, 1);
  sigma_kappa_cb ~ normal(0, 1);
  sigma_tau_nd ~ normal(0, 1);

  z_alpha_ctx ~ std_normal();
  z_tau_m ~ std_normal();
  z_eta_gc ~ std_normal();
  z_lambda_gc ~ std_normal();
  z_theta_cb ~ std_normal();
  z_kappa_ctx ~ std_normal();
  z_gamma_suppress ~ std_normal();
  z_a ~ std_normal();
  z_beta_a ~ std_normal();
  z_kappa_cb ~ std_normal();
  z_tau_nd ~ std_normal();

  }


generated quantities {
  vector[N] log_lik;
  for (s in 1:S) {
    int start_t = start_idx[s];
    int end_t = end_idx[s];
    
    vector[N_MF] mf_state = rep_vector(0.0, N_MF);
    vector[N_MF] w_gc1 = rep_vector(0.0, N_MF);
    vector[N_MF] w_gc2 = rep_vector(0.0, N_MF);
    vector[N_MF] w_mli1 = rep_vector(0.0, N_MF);
    vector[N_MF] w_mli2 = rep_vector(0.0, N_MF);
    vector[2] Q_ctx = rep_vector(0.5, 2);

    real a_s = a[s];
    real tau_nd_s = tau_nd[s];
    real theta_cb_s = theta_cb[s];
    real kappa_ctx_s = kappa_ctx[s];
    real gamma_suppress_s = gamma_suppress[s];
    real beta_a_s = beta_a[s];
    real kappa_cb_s = kappa_cb[s];
    real alpha_ctx_s = alpha_ctx[s];
    real tau_m_s = tau_m[s];
    
    real l_gc_eff = lambda_gc[s] + 1e-8;
    real l_mli_eff = (lambda_gc[s] * 1.5) + 1e-8;
    real eta_gc_s = eta_gc[s];
    real eta_mli_s = eta_gc[s];
    
    real inv_l_gc_eff = 1.0 / l_gc_eff;
    real inv_l_mli_eff = 1.0 / l_mli_eff;
    
    for (t in start_t:end_t) {
      if (iti[t] > 0.01) {
        mf_state = exact_mf_step(iti[t], mf_state, tau_m_s, 0.0, N_MF);
        real decay_gc_iti = exp(-lambda_gc[s] * iti[t]);
        real decay_mli_iti = exp(-(lambda_gc[s] * 1.5) * iti[t]);
        w_gc1 *= decay_gc_iti;
        w_gc2 *= decay_gc_iti;
        w_mli1 *= decay_mli_iti;
        w_mli2 *= decay_mli_iti;
      }
      
      real Q_cb_1 = dot_product(w_gc1, mf_state) - dot_product(w_mli1, mf_state);
      real Q_cb_2 = dot_product(w_gc2, mf_state) - dot_product(w_mli2, mf_state);
      real delta_Q_ctx = Q_ctx[2] - Q_ctx[1];
      real delta_Q_cb = Q_cb_2 - Q_cb_1;
      
      real w_bias = 0.5 + 0.45 * tanh(theta_cb_s * delta_Q_cb);
      real conflict = 0.5 * (1.0 - tanh(10.0 * delta_Q_ctx) * tanh(10.0 * delta_Q_cb));
      real v_base = kappa_ctx_s * delta_Q_ctx + kappa_cb_s * delta_Q_cb;
      real v_effective = v_base * exp(-gamma_suppress_s * conflict);
      
      if (abs(v_effective) < 1e-4) {
        v_effective = v_effective >= 0 ? 1e-4 : -1e-4;
      }
      
      real a_effective = a_s + beta_a_s * conflict;
      
      real w_bias_subj = choice[t] == 1 ? w_bias : 1.0 - w_bias;
      real v_subj = choice[t] == 1 ? v_effective : -v_effective;
      
      real log_uniform_dens = log(1.0 / 5.8);
      real wiener_lp;
      if (rt[t] - tau_nd_s < 1e-4) {
        log_lik[t] = log_uniform_dens;
      } else {
        wiener_lp = wiener_lpdf(rt[t] | a_effective, tau_nd_s, w_bias_subj, v_subj);
        log_lik[t] = log_mix(0.98, wiener_lp, log_uniform_dens);
      }
      
      real RPE_ctx = reward[t] - Q_ctx[choice[t] + 1];
      Q_ctx[choice[t] + 1] += alpha_ctx_s * RPE_ctx;
      real cb_pred = (choice[t] == 1) ? Q_cb_2 : Q_cb_1;
      real RPE_cb = reward[t] - cb_pred;
      real E_cb1 = (choice[t] == 0) ? RPE_cb : 0.0;
      real E_cb2 = (choice[t] == 1) ? RPE_cb : 0.0;
      
      if (f_dur[t] > 0.01) {
        mf_state = exact_mf_step(f_dur[t], mf_state, tau_m_s, reward[t], N_MF);
        real decay_gc_f = exp(-l_gc_eff * f_dur[t]);
        real decay_mli_f = exp(-l_mli_eff * f_dur[t]);
        real int_gc_f = (1.0 - decay_gc_f) * inv_l_gc_eff;
        real int_mli_f = (1.0 - decay_mli_f) * inv_l_mli_eff;
        real scale_gc1 = eta_gc_s * E_cb1 * int_gc_f;
        real scale_gc2 = eta_gc_s * E_cb2 * int_gc_f;
        real scale_mli1 = -eta_mli_s * E_cb1 * int_mli_f;
        real scale_mli2 = -eta_mli_s * E_cb2 * int_mli_f;
        w_gc1 = w_gc1 * decay_gc_f + mf_state * scale_gc1;
        w_gc2 = w_gc2 * decay_gc_f + mf_state * scale_gc2;
        w_mli1 = w_mli1 * decay_mli_f + mf_state * scale_mli1;
        w_mli2 = w_mli2 * decay_mli_f + mf_state * scale_mli2;
      }
    }
  }
}
