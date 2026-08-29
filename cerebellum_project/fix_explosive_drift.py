import re

def process_file(in_name, out_name):
    text = open(in_name, 'r', encoding='ascii').read()

    # Replace the explosive exponential alignment with bounded topological conflict
    old_align = 'real alignment = tanh(10.0 * delta_Q_ctx) * tanh(10.0 * delta_Q_cb);'
    new_conflict = 'real conflict = 0.5 * (1.0 - tanh(10.0 * delta_Q_ctx) * tanh(10.0 * delta_Q_cb));'
    
    text = text.replace(old_align, new_conflict)

    old_v = 'real v_effective = v_base * exp(gamma_suppress_s * alignment);'
    new_v = 'real v_effective = v_base * exp(-gamma_suppress_s * conflict);'
    text = text.replace(old_v, new_v)

    old_v_gq = 'real v_effective = v_base * exp(gamma_suppress[s] * alignment);'
    new_v_gq = 'real v_effective = v_base * exp(-gamma_suppress[s] * conflict);'
    text = text.replace(old_v_gq, new_v_gq)

    open(out_name, 'w', encoding='ascii', newline='\n').write(text)

process_file('bvk_continuous.stan', 'bvk_continuous.stan')
process_file('bvk_continuous_gq.stan', 'bvk_continuous_gq.stan')
