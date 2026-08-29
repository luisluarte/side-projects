import re

def process_file(in_name, out_name):
    try:
        text = open(in_name, 'r', encoding='ascii').read()
    except UnicodeDecodeError:
        text = open(in_name, 'r', encoding='utf-16').read()

    # 1. Remove MLI hierarchical parameters
    to_remove = [
        'mu_eta_mli', 'mu_lambda_mli',
        'sigma_eta_mli', 'sigma_lambda_mli',
        'z_eta_mli', 'z_lambda_mli',
        'eta_mli', 'lambda_mli'
    ]

    for var in to_remove:
        # parameters block
        text = re.sub(r'real(?:<lower=0>)?\s+'+var+r'\s*;\s*', '', text)
        text = re.sub(r'vector\[S\]\s+'+var+r'\s*;\s*', '', text)
        # transformed parameters assignments
        text = re.sub(r'vector\[S\]\s+'+var+r'\s*=[^;]+;\s*', '', text)
        # model priors
        text = re.sub(r''+var+r'\s*~[^;]+;\s*', '', text)
        
    # 2. Update partial_sum signature and reduce_sum call
    text = re.sub(r',\s*vector\s+eta_mli\s*,\s*vector\s+lambda_mli', '', text)
    text = re.sub(r',\s*eta_mli\s*,\s*lambda_mli', '', text)

    # 3. Inside the loop, replace the scalar extractions
    text = re.sub(r'real\s+eta_mli_s\s*=\s*eta_mli\[s\];', 'real eta_mli_s = eta_gc[s];', text)
    text = re.sub(r'real\s+lambda_mli_s\s*=\s*lambda_mli\[s\];', 'real lambda_mli_s = lambda_gc[s] * 1.5;', text)
    
    # 4. Topological Alignment replacement
    old_cc = 'real delta_CC = abs(delta_Q_ctx - delta_Q_cb);'
    new_align = 'real alignment = tanh(10.0 * delta_Q_ctx) * tanh(10.0 * delta_Q_cb);'
    text = text.replace(old_cc, new_align)

    old_v = 'real v_effective = v_base * exp(-gamma_suppress_s * delta_CC);'
    new_v = 'real v_effective = v_base * exp(gamma_suppress_s * alignment);'
    text = text.replace(old_v, new_v)

    # For GQ file, it might not have the _s variables!
    old_v_gq = 'real v_effective = v_base * exp(-gamma_suppress[s] * delta_CC);'
    new_v_gq = 'real v_effective = v_base * exp(gamma_suppress[s] * alignment);'
    text = text.replace(old_v_gq, new_v_gq)

    # Also for GQ file, MLI scalars
    text = text.replace('lambda_mli[s]', '(lambda_gc[s] * 1.5)')
    text = text.replace('eta_mli[s]', 'eta_gc[s]')

    open(out_name, 'w', encoding='ascii', newline='\n').write(text)

process_file('head_bvk.stan', 'bvk_continuous.stan')
process_file('good_gq.stan', 'bvk_continuous_gq.stan')
