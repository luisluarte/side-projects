import sys

# FIX M012
with open('src/stan/m012_ss3.stan', 'r') as f:
    content = f.read()

old_block_partial = """             if (ch > 0 && prev_ch > 0 && rt[t] > 0.0) {
                 int is_switch = (ch != prev_ch) ? 1 : 0;
                 real Q_switch = Q[3 - prev_ch]; 
                 real Q_stay = Q[prev_ch];
                 real veff_raw = v_s * (Q_switch - Q_stay);
                 real veff = (is_switch == 1) ? veff_raw : -veff_raw;
                 real w_eff = (is_switch == 1) ? w_start : (1.0 - w_start);
                 pt += wiener_lpdf(rt[t] | a_dyn, tnd_s, w_eff, veff);
             }"""

new_block_partial = """             if (ch > 0 && rt[t] > 0.0) {
                 real veff_raw = v_s * Q_diff;
                 real veff = (ch == 1) ? veff_raw : -veff_raw;
                 real w_eff = (ch == 1) ? w_start : (1.0 - w_start);
                 pt += wiener_lpdf(rt[t] | a_dyn, tnd_s, w_eff, veff);
             }"""

old_block_gq = """             if (ch > 0 && prev_ch > 0 && rt[t] > 0.0) {
                 int is_switch = (ch != prev_ch) ? 1 : 0;
                 real Q_switch = Q[3 - prev_ch]; 
                 real Q_stay = Q[prev_ch];
                 real veff_raw = v_s * (Q_switch - Q_stay);
                 real veff = (is_switch == 1) ? veff_raw : -veff_raw;
                 real w_eff = (is_switch == 1) ? w_start : (1.0 - w_start);
                 log_lik[t] = wiener_lpdf(rt[t] | a_dyn, tnd_s, w_eff, veff);
                 
                 if (veff_raw == 0.0) {
                     pred_sw[t] = w_start;
                 } else {
                     pred_sw[t] = (exp(-2.0 * veff_raw * a_dyn * w_start) - 1.0) / (exp(-2.0 * veff_raw * a_dyn) - 1.0);
                 }
             }"""

new_block_gq = """             if (ch > 0 && rt[t] > 0.0) {
                 real veff_raw = v_s * Q_diff;
                 real veff = (ch == 1) ? veff_raw : -veff_raw;
                 real w_eff = (ch == 1) ? w_start : (1.0 - w_start);
                 log_lik[t] = wiener_lpdf(rt[t] | a_dyn, tnd_s, w_eff, veff);
                 
                 if (veff_raw == 0.0) {
                     pred_sw[t] = w_start;
                 } else {
                     real p_left = (exp(-2.0 * veff_raw * a_dyn * w_start) - 1.0) / (exp(-2.0 * veff_raw * a_dyn) - 1.0);
                     if (t > start_idx[s] && prev_ch > 0) {
                         pred_sw[t] = (prev_ch == 1) ? (1.0 - p_left) : p_left;
                     } else {
                         pred_sw[t] = 0.5;
                     }
                 }
             }"""

content = content.replace(old_block_partial, new_block_partial)
content = content.replace(old_block_gq, new_block_gq)
with open('src/stan/m012_ss3.stan', 'w') as f:
    f.write(content)

# FIX VOPT
with open('src/stan/vopt_ss3.stan', 'r') as f:
    content = f.read()

content = content.replace(old_block_partial, new_block_partial)
content = content.replace(old_block_gq, new_block_gq)
with open('src/stan/vopt_ss3.stan', 'w') as f:
    f.write(content)

print("Replaced both files successfully")
