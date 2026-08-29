import re

def mutate(filepath):
    with open(filepath, 'r') as f:
        text = f.read()

    # 1. Change delta_CC to CC_interaction
    text = text.replace('real delta_CC = abs(delta_Q_ctx - delta_Q_cb);', 'real CC_interaction = delta_Q_ctx * delta_Q_cb;')

    # 2. Change v_effective calculation
    if '_gq' in filepath:
        text = text.replace('real v_effective = v_base * exp(-gamma_suppress[s] * delta_CC);', 'real v_effective = v_base * exp(gamma_suppress[s] * CC_interaction);')
    else:
        text = text.replace('real v_effective = v_base * exp(-gamma_suppress_s * delta_CC);', 'real v_effective = v_base * exp(gamma_suppress_s * CC_interaction);')

    # 3. Soften the scalars (theta_cb to 2.0, gamma_suppress to 5.0)
    text = text.replace('10.0 * inv_logit(mu_theta_cb', '2.0 * inv_logit(mu_theta_cb')
    text = text.replace('10.0 * inv_logit(mu_gamma_suppress', '5.0 * inv_logit(mu_gamma_suppress')

    with open(filepath, 'w') as f:
        f.write(text)

mutate('src/models/bvk_continuous.stan')
mutate('src/models/bvk_continuous_gq.stan')
