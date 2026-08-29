import re
text = open('bvk_continuous_gq.stan', 'r').read()

text = re.sub(r'real w_bias = 0\.5 \+ 0\.45 \* tanh\(theta_cb\[s\] \* delta_Q_cb\);', 'real w_bias = 0.5;', text)
text = re.sub(r'real delta_CC = abs\(delta_Q_ctx - delta_Q_cb\);', 'real alignment = tanh(10.0 * delta_Q_ctx) * tanh(10.0 * delta_Q_cb);', text)
text = re.sub(r'exp\(-gamma_suppress\[s\] \* delta_CC\)', 'exp(gamma_suppress[s] * alignment)', text)

open('bvk_continuous_gq.stan', 'w', newline='\n').write(text)
