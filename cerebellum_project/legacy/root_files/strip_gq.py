import re

text = open('bvk_continuous_gq.stan', 'r').read()

to_remove = [
    'mu_eta_mli', 'mu_lambda_mli', 'mu_theta_cb',
    'sigma_eta_mli', 'sigma_lambda_mli', 'sigma_theta_cb',
    'z_eta_mli', 'z_lambda_mli', 'z_theta_cb',
    'eta_mli', 'lambda_mli', 'theta_cb'
]

for var in to_remove:
    # parameters block
    text = re.sub(r'real(?:<lower=0>)?\s+'+var+r'\s*;\s*', '', text)
    text = re.sub(r'vector\[S\]\s+'+var+r'\s*;\s*', '', text)
    # transformed parameters assignments
    text = re.sub(r'vector\[S\]\s+'+var+r'\s*=[^;]+;\s*', '', text)
    # model priors
    text = re.sub(r''+var+r'\s*~[^;]+;\s*', '', text)

open('bvk_continuous_gq.stan', 'w', newline='\n').write(text)
