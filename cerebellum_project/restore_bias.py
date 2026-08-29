import re

def restore_theta(filename, is_gq=False):
    with open(filename, 'r', encoding='ascii') as f:
        text = f.read()

    text = re.sub(r'(real mu_tau_nd;)', r'\1\n  real mu_theta_cb;', text)
    text = re.sub(r'(real<lower=0> sigma_tau_nd;)', r'\1\n  real<lower=0> sigma_theta_cb;', text)
    text = re.sub(r'(vector\[S\] z_tau_nd;)', r'\1\n  vector[S] z_theta_cb;', text)

    text = re.sub(
        r'(vector\[S\] tau_nd = [^;]+;)', 
        r'\1\n  vector[S] theta_cb = 10.0 * inv_logit(mu_theta_cb + sigma_theta_cb * z_theta_cb);', 
        text
    )

    if not is_gq:
        text = re.sub(r'(mu_tau_nd ~ [^;]+;)', r'\1\n  mu_theta_cb ~ normal(0, 1.5);', text)
        text = re.sub(r'(sigma_tau_nd ~ [^;]+;)', r'\1\n  sigma_theta_cb ~ normal(0, 1);', text)
        text = re.sub(r'(z_tau_nd ~ [^;]+;)', r'\1\n  z_theta_cb ~ std_normal();', text)

    text = re.sub(
        r'(vector gamma_suppress, vector a, vector tau_nd)',
        r'\1, vector theta_cb',
        text
    )

    text = re.sub(
        r'(kappa_ctx, gamma_suppress, a, tau_nd)',
        r'\1, theta_cb',
        text
    )

    text = text.replace(
        'real w_bias = 0.5;',
        'real w_bias = 0.5 + 0.45 * tanh(theta_cb[s] * delta_Q_cb);'
    )

    with open(filename, 'w', encoding='ascii', newline='\n') as f:
        f.write(text)

restore_theta('bvk_continuous.stan', is_gq=False)
restore_theta('bvk_continuous_gq.stan', is_gq=True)
