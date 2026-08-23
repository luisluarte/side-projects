functions {
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
  real mu_eta_mli;
  real mu_lambda_mli;
  real mu_theta_cb;
  real mu_kappa_ctx;
  real mu_gamma_suppress;
  real mu_a;
  real mu_tau_nd;

  real<lower=0> sigma_alpha_ctx;
  real<lower=0> sigma_tau_m;
  real<lower=0> sigma_eta_gc;
  real<lower=0> sigma_lambda_gc;
  real<lower=0> sigma_eta_mli;
  real<lower=0> sigma_lambda_mli;
  real<lower=0> sigma_theta_cb;
  real<lower=0> sigma_kappa_ctx;
  real<lower=0> sigma_gamma_suppress;
  real<lower=0> sigma_a;
  real<lower=0> sigma_tau_nd;

  vector[S] z_alpha_ctx;
  vector[S] z_tau_m;
  vector[S] z_eta_gc;
  vector[S] z_lambda_gc;
  vector[S] z_eta_mli;
  vector[S] z_lambda_mli;
  vector[S] z_theta_cb;
  vector[S] z_kappa_ctx;
  vector[S] z_gamma_suppress;
  vector[S] z_a;
  vector[S] z_tau_nd;
}

transformed parameters {
  vector[S] alpha_ctx = inv_logit(mu_alpha_ctx + sigma_alpha_ctx * z_alpha_ctx);
  vector[S] tau_m = 0.1 + 9.9 * inv_logit(mu_tau_m + sigma_tau_m * z_tau_m);
  vector[S] eta_gc = 5.0 * inv_logit(mu_eta_gc + sigma_eta_gc * z_eta_gc);
  vector[S] lambda_gc = 5.0 * inv_logit(mu_lambda_gc + sigma_lambda_gc * z_lambda_gc);
  vector[S] eta_mli = 5.0 * inv_logit(mu_eta_mli + sigma_eta_mli * z_eta_mli);
  vector[S] lambda_mli = 5.0 * inv_logit(mu_lambda_mli + sigma_lambda_mli * z_lambda_mli);
  vector[S] theta_cb = 10.0 * inv_logit(mu_theta_cb + sigma_theta_cb * z_theta_cb);
  vector[S] kappa_ctx = 10.0 * inv_logit(mu_kappa_ctx + sigma_kappa_ctx * z_kappa_ctx);
  vector[S] gamma_suppress = 10.0 * inv_logit(mu_gamma_suppress + sigma_gamma_suppress * z_gamma_suppress);

  vector[S] a = 0.5 + 4.5 * inv_logit(mu_a + sigma_a * z_a);
  vector[S] tau_nd = 0.001 + (to_vector(min_rt) - 0.002) .* inv_logit(mu_tau_nd + sigma_tau_nd * z_tau_nd);
}

generated quantities {
  vector[N] log_lik; // The strict predictive tensor for PSIS-LOO

  { // Local scope restricts memory output exclusively to log_lik
    for (s in 1:S) {
      int start_t = start_idx[s];
      int end_t = end_idx[s];

      vector[2] Q_ctx = rep_vector(0.5, 2);
      vector[N_MF] mf_state = rep_vector(0.0, N_MF);

      vector[N_MF] w_gc1 = rep_vector(0.0, N_MF);
      vector[N_MF] w_gc2 = rep_vector(0.0, N_MF);
      vector[N_MF] w_mli1 = rep_vector(0.0, N_MF);
      vector[N_MF] w_mli2 = rep_vector(0.0, N_MF);

      for (t in start_t:end_t) {
        // Phase 1: ITI Decay
        if (iti[t] > 0.01) {
          mf_state = exact_mf_step(iti[t], mf_state, tau_m[s], 0.0, N_MF);
          real decay_gc_iti = exp(-lambda_gc[s] * iti[t]);
          real decay_mli_iti = exp(-lambda_mli[s] * iti[t]);
          w_gc1 *= decay_gc_iti;
          w_gc2 *= decay_gc_iti;
          w_mli1 *= decay_mli_iti;
          w_mli2 *= decay_mli_iti;
        }

        // Phase 2: Coupled Kinematic Probability Readout
        real Q_cb_1 = dot_product(w_gc1, mf_state) - dot_product(w_mli1, mf_state);
        real Q_cb_2 = dot_product(w_gc2, mf_state) - dot_product(w_mli2, mf_state);

        real delta_Q_ctx = Q_ctx[2] - Q_ctx[1];
        real delta_Q_cb = Q_cb_2 - Q_cb_1;

        real w_bias = 0.5 + 0.45 * tanh(theta_cb[s] * delta_Q_cb);
        real delta_CC = abs(delta_Q_ctx - delta_Q_cb);
        real v_base = kappa_ctx[s] * delta_Q_ctx;
        real v_effective = v_base * exp(-gamma_suppress[s] * delta_CC);

        // Strict Drift Lower Bound
        if (abs(v_effective) < 1e-4) {
          v_effective = v_effective >= 0 ? 1e-4 : -1e-4;
        }

        real log_uniform_dens = log(1.0 / 5.8);
        real wiener_lp;

        // Temporal Impossibility Violation Check
        if (rt[t] - tau_nd[s] < 1e-4) {
          log_lik[t] = log_uniform_dens; // Bypass wiener entirely
        } else {
          // Compute Pointwise Log-Likelihood
          if (choice[t] == 1) {
            wiener_lp = wiener_lpdf(rt[t] | a[s], tau_nd[s], w_bias, v_effective);
          } else {
            wiener_lp = wiener_lpdf(rt[t] | a[s], tau_nd[s], 1.0 - w_bias, -v_effective);
          }
          log_lik[t] = log_mix(0.98, wiener_lp, log_uniform_dens);
        }

        // Phase 3: Exact Analytical Plasticity
        real RPE_ctx = reward[t] - Q_ctx[choice[t] + 1];
        Q_ctx[choice[t] + 1] += alpha_ctx[s] * RPE_ctx;

        real cb_pred = (choice[t] == 1) ? Q_cb_2 : Q_cb_1;
        real RPE_cb = reward[t] - cb_pred;

        real E_cb1 = (choice[t] == 0) ? RPE_cb : 0.0;
        real E_cb2 = (choice[t] == 1) ? RPE_cb : 0.0;

        if (f_dur[t] > 0.01) {
          mf_state = exact_mf_step(f_dur[t], mf_state, tau_m[s], reward[t], N_MF);
          real l_gc_eff = lambda_gc[s] + 1e-8;
          real l_mli_eff = lambda_mli[s] + 1e-8;
          real decay_gc_f = exp(-l_gc_eff * f_dur[t]);
          real decay_mli_f = exp(-l_mli_eff * f_dur[t]);
          real int_gc_f = (1.0 - decay_gc_f) / l_gc_eff;
          real int_mli_f = (1.0 - decay_mli_f) / l_mli_eff;

          real scale_gc1 = eta_gc[s] * E_cb1 * int_gc_f;
          real scale_gc2 = eta_gc[s] * E_cb2 * int_gc_f;
          real scale_mli1 = -eta_mli[s] * E_cb1 * int_mli_f;
          real scale_mli2 = -eta_mli[s] * E_cb2 * int_mli_f;

          w_gc1 = w_gc1 * decay_gc_f + mf_state * scale_gc1;
          w_gc2 = w_gc2 * decay_gc_f + mf_state * scale_gc2;
          w_mli1 = w_mli1 * decay_mli_f + mf_state * scale_mli1;
          w_mli2 = w_mli2 * decay_mli_f + mf_state * scale_mli2;
        }
      }
    }
  }
}