import re

def process_file(in_name, out_name):
    text = open(in_name, 'r', encoding='ascii').read()

    # Apply the biological minimum bound to the Cerebellar learning rate
    old_eta = 'vector[S] eta_gc = 5.0 * inv_logit(mu_eta_gc + sigma_eta_gc * z_eta_gc);'
    new_eta = 'vector[S] eta_gc = 0.5 + 4.5 * inv_logit(mu_eta_gc + sigma_eta_gc * z_eta_gc);'
    
    text = text.replace(old_eta, new_eta)

    open(out_name, 'w', encoding='ascii', newline='\n').write(text)

process_file('bvk_continuous.stan', 'bvk_continuous.stan')
process_file('bvk_continuous_gq.stan', 'bvk_continuous_gq.stan')
