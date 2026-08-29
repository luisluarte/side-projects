import re

def process_file(in_name, out_name):
    text = open(in_name, 'r', encoding='ascii').read()

    # Prevent tau_m from decaying to 0 during the ITI and zeroing the Cerebellar gradients
    old_tau = 'vector[S] tau_m = 0.1 + 9.9 * inv_logit(mu_tau_m + sigma_tau_m * z_tau_m);'
    new_tau = 'vector[S] tau_m = 1.0 + 9.0 * inv_logit(mu_tau_m + sigma_tau_m * z_tau_m);'
    
    text = text.replace(old_tau, new_tau)

    open(out_name, 'w', encoding='ascii', newline='\n').write(text)

process_file('bvk_continuous.stan', 'bvk_continuous.stan')
process_file('bvk_continuous_gq.stan', 'bvk_continuous_gq.stan')
