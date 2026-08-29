import re

def generate(is_gq=False):
    suffix = '_gq' if is_gq else ''
    filename = f'bvk_continuous{suffix}.stan'
    
    with open('old_bvk.stan', 'r', encoding='ascii') as f:
        text = f.read()

    # 1. Update partial_sum signature
    text = re.sub(r'vector lambda_gc, vector eta_mli, vector lambda_mli,\s*vector theta_cb, vector kappa_ctx',
                  r'vector lambda_gc, vector kappa_ctx', text)

    # 2. Update parameters, transformed parameters, and model blocks
    text = re.sub(r'\s*real mu_eta_mli;\s*real mu_lambda_mli;\s*real mu_theta_cb;', '', text)
    text = re.sub(r'\s*real<lower=0> sigma_eta_mli;\s*real<lower=0> sigma_lambda_mli;\s*real<lower=0> sigma_theta_cb;', '', text)
    text = re.sub(r'\s*vector\[S\] z_eta_mli;\s*vector\[S\] z_lambda_mli;\s*vector\[S\] z_theta_cb;', '', text)

    text = re.sub(r'\s*vector\[S\] eta_mli = 5.0 \* inv_logit\(mu_eta_mli \+ sigma_eta_mli \* z_eta_mli\);', '', text)
    text = re.sub(r'\s*vector\[S\] lambda_mli = 5.0 \* inv_logit\(mu_lambda_mli \+ sigma_lambda_mli \* z_lambda_mli\);', '', text)
    text = re.sub(r'\s*vector\[S\] theta_cb = 10.0 \* inv_logit\(mu_theta_cb \+ sigma_theta_cb \* z_theta_cb\);', '', text)

    text = re.sub(r'\s*mu_eta_mli ~ normal\(0, 1.5\);\s*mu_lambda_mli ~ normal\(0, 1.5\);\s*mu_theta_cb ~ normal\(0, 1.5\);', '', text)
    text = re.sub(r'\s*sigma_eta_mli ~ normal\(0, 1\);\s*sigma_lambda_mli ~ normal\(0, 1\);\s*sigma_theta_cb ~ normal\(0, 1\);', '', text)
    text = re.sub(r'\s*z_eta_mli ~ std_normal\(\);\s*z_lambda_mli ~ std_normal\(\);\s*z_theta_cb ~ std_normal\(\);', '', text)

    text = re.sub(r'eta_gc, lambda_gc, eta_mli, lambda_mli, theta_cb, kappa_ctx',
                  r'eta_gc, lambda_gc, kappa_ctx', text)

    # 3. Update the inner loop physics
    old_loop = r'''        real Q_cb_1 = dot_product\(w_gc1, mf_state\) - dot_product\(w_mli1, mf_state\);.*?w_mli2 = w_mli2 \* decay_mli_f \+ \(eta_mli\[s\] \* E_c2 \* mf_state\) \* int_mli_f;
        \}'''
    
    s_idx = '[s]' if not is_gq else '[s]'
    
    new_loop = f'''        real Q_cb_1 = dot_product(w_gc1, mf_state) - dot_product(w_mli1, mf_state);
        real Q_cb_2 = dot_product(w_gc2, mf_state) - dot_product(w_mli2, mf_state);

        real delta_Q_ctx = Q_ctx[2] - Q_ctx[1];
        real delta_Q_cb = Q_cb_2 - Q_cb_1;

        // Static Bias (Parametric Ablation)
        real w_bias = 0.5;
        
        // Normalized Topological Alignment (1 = perfect agreement, -1 = perfect conflict)
        real alignment = tanh(10.0 * delta_Q_ctx) * tanh(10.0 * delta_Q_cb);
        
        real v_base = kappa_ctx{s_idx} * delta_Q_ctx;
        real v_effective = v_base * exp(gamma_suppress{s_idx} * alignment);

        if (choice[t] == 1) {{
          {'target_sum += ' if not is_gq else 'wiener_lp = '}wiener_lpdf(rt[t] | a{s_idx}, tau_nd{s_idx}, w_bias, v_effective);
        }} else {{
          {'target_sum += ' if not is_gq else 'wiener_lp = '}wiener_lpdf(rt[t] | a{s_idx}, tau_nd{s_idx}, 1.0 - w_bias, -v_effective);
        }}
        {'log_lik[t] = log_mix(0.98, wiener_lp, log_uniform_dens);' if is_gq else ''}

        // Phase 3: Analytical Plasticity in Dual Space
        real RPE = reward[t] - Q_ctx[choice[t] + 1];
        Q_ctx[choice[t] + 1] += alpha_ctx{s_idx} * RPE;

        // Localized Cerebellar RPE
        real Q_cb_chosen = (choice[t] == 0) ? Q_cb_1 : Q_cb_2;
        real RPE_cb = reward[t] - Q_cb_chosen;

        real E_c1 = (choice[t] == 0) ? RPE_cb : 0.0;
        real E_c2 = (choice[t] == 1) ? RPE_cb : 0.0;

        if (f_dur[t] > 0.01) {{
          mf_state = exact_mf_step(f_dur[t], mf_state, tau_m{s_idx}, reward[t], N_MF);

          // Coupled MLI Parameters (Parametric Ablation)
          real eta_mli_s = eta_gc{s_idx};
          real lambda_mli_s = lambda_gc{s_idx} * 1.5;

          real l_gc_eff = lambda_gc{s_idx} + 1e-8;
          real l_mli_eff = lambda_mli_s + 1e-8;

          real decay_gc_f = exp(-l_gc_eff * f_dur[t]);
          real decay_mli_f = exp(-l_mli_eff * f_dur[t]);

          real int_gc_f = (1.0 - decay_gc_f) / l_gc_eff;
          real int_mli_f = (1.0 - decay_mli_f) / l_mli_eff;

          w_gc1 = w_gc1 * decay_gc_f + (eta_gc{s_idx} * E_c1 * mf_state) * int_gc_f;
          w_gc2 = w_gc2 * decay_gc_f + (eta_gc{s_idx} * E_c2 * mf_state) * int_gc_f;
          w_mli1 = w_mli1 * decay_mli_f + (eta_mli_s * E_c1 * mf_state) * int_mli_f;
          w_mli2 = w_mli2 * decay_mli_f + (eta_mli_s * E_c2 * mf_state) * int_mli_f;
        }}'''

    text = re.sub(old_loop, new_loop, text, flags=re.DOTALL)
    
    if is_gq:
        text = text.replace('real partial_sum(', 'vector partial_sum(')
        text = text.replace('real target_sum = 0.0;', 'vector[size(seq_subj_slice)] log_lik;\n    real log_uniform_dens = log(1.0 / 5.8);\n    real wiener_lp;')
        text = text.replace('return target_sum;', 'return log_lik;')

    with open(filename, 'w', encoding='ascii', newline='\n') as f:
        f.write(text)

generate(is_gq=False)
generate(is_gq=True)
