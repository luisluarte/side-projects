text = open('good_gq.stan', 'r').read()

# 1. Strip out the 3 parameters from the block (mu_, sigma_, z_)
text = text.replace('real mu_eta_mli;\n  real mu_lambda_mli;\n  real mu_theta_cb;\n', '')
text = text.replace('real<lower=0> sigma_eta_mli;\n  real<lower=0> sigma_lambda_mli;\n  real<lower=0> sigma_theta_cb;\n', '')
text = text.replace('vector[S] z_eta_mli;\n  vector[S] z_lambda_mli;\n  vector[S] z_theta_cb;\n', '')

# 2. Strip from transformed parameters
text = text.replace('vector[S] eta_mli = 5.0 * inv_logit(mu_eta_mli + sigma_eta_mli * z_eta_mli);\n', '')
text = text.replace('vector[S] lambda_mli = 5.0 * inv_logit(mu_lambda_mli + sigma_lambda_mli * z_lambda_mli);\n', '')
text = text.replace('vector[S] theta_cb = 10.0 * inv_logit(mu_theta_cb + sigma_theta_cb * z_theta_cb);\n', '')

# 3. Replace usage in generated quantities loop
old_block = '''        real w_bias = 0.5 + 0.45 * tanh(theta_cb[s] * delta_Q_cb);
        real delta_CC = abs(delta_Q_ctx - delta_Q_cb);
        real v_base = kappa_ctx[s] * delta_Q_ctx;
        real v_effective = v_base * exp(-gamma_suppress[s] * delta_CC);'''

new_block = '''        real w_bias = 0.5;
        real alignment = tanh(10.0 * delta_Q_ctx) * tanh(10.0 * delta_Q_cb);
        real v_base = kappa_ctx[s] * delta_Q_ctx;
        real v_effective = v_base * exp(gamma_suppress[s] * alignment);'''

text = text.replace(old_block, new_block)

text = text.replace('lambda_mli[s]', '(lambda_gc[s] * 1.5)')
text = text.replace('eta_mli[s]', 'eta_gc[s]')

open('bvk_continuous_gq.stan', 'w', newline='\n').write(text)
