import re

with open("src/models/bvk_full_gating.stan", "r") as f:
    code = f.read()

# Remove target +=
code = re.sub(
    r'target \+= reduce_sum\([^;]+;\n',
    '',
    code
)

# Add generated quantities block
gq_code = """
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
      
      log_lik[t] = wiener_lpdf(rt[t] | a_effective, tau_nd_s, w_bias_subj, v_subj);
      
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
"""

with open("src/models/bvk_full_gating_gq.stan", "w") as f:
    f.write(code + "\n" + gq_code)

print("Generated src/models/bvk_full_gating_gq.stan")
