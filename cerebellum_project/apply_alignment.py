import re

def patch(file):
    with open(file, 'r', encoding='ascii') as f:
        text = f.read()

    old_block = r'''        real w_bias = 0.5 \+ 0.45 \* tanh\(theta_cb.*? \* delta_Q_cb\);
        real delta_CC = abs\(delta_Q_ctx - delta_Q_cb\);
        real v_base = kappa_ctx.*? \* delta_Q_ctx;
        real v_effective = v_base \* exp\(-gamma_suppress.*? \* delta_CC\);'''

    suffix = '[s]' if '_gq' in file else '[s]'

    new_block = f'''        real w_bias = 0.5 + 0.45 * tanh(theta_cb{suffix} * delta_Q_cb);
        
        real alignment = tanh(10.0 * delta_Q_ctx) * tanh(10.0 * delta_Q_cb);
        
        real v_base = kappa_ctx{suffix} * delta_Q_ctx;
        real v_effective = v_base * exp(gamma_suppress{suffix} * alignment);'''

    text = re.sub(old_block, new_block, text, flags=re.DOTALL)
    
    with open(file, 'w', encoding='ascii', newline='\n') as f:
        f.write(text)

patch('bvk_continuous.stan')
patch('bvk_continuous_gq.stan')
