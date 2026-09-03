import sys
import re

for filename in ['src/stan/m012_ss3.stan', 'src/stan/vopt_ss3.stan']:
    with open(filename, 'r') as f:
        content = f.read()
    
    # Replace partial sum block
    old_partial = re.compile(r'if \(t > start_idx\[s\]\) \{\s*int prev_ch = resp\[t-1\];\s*if \(ch > 0 && prev_ch > 0 && rt\[t\] > 0\.0\) \{\s*int is_switch = \(ch != prev_ch\) \? 1 : 0;\s*real Q_switch = Q\[3 - prev_ch\];\s*real Q_stay = Q\[prev_ch\];\s*real veff_raw = v_s \* \(Q_switch - Q_stay\);\s*real veff = \(is_switch == 1\) \? veff_raw : -veff_raw;\s*real w_eff = \(is_switch == 1\) \? w_start : \(1\.0 - w_start\);\s*pt \+= wiener_lpdf\(rt\[t\] \| a_dyn, tnd_s, w_eff, veff\);\s*\}\s*\}', re.MULTILINE)
    
    new_partial = """if (t > start_idx[s]) {
             if (ch > 0 && rt[t] > 0.0) {
                 real veff_raw = v_s * Q_diff;
                 real veff = (ch == 1) ? veff_raw : -veff_raw;
                 real w_eff = (ch == 1) ? w_start : (1.0 - w_start);
                 pt += wiener_lpdf(rt[t] | a_dyn, tnd_s, w_eff, veff);
             }
          }"""
          
    content = old_partial.sub(new_partial, content)

    # Replace generated quantities block
    old_gq = re.compile(r'if \(t > start_idx\[s\]\) \{\s*int prev_ch = resp\[t-1\];\s*if \(ch > 0 && prev_ch > 0 && rt\[t\] > 0\.0\) \{\s*int is_switch = \(ch != prev_ch\) \? 1 : 0;\s*real Q_switch = Q\[3 - prev_ch\];\s*real Q_stay = Q\[prev_ch\];\s*real veff_raw = v_s \* \(Q_switch - Q_stay\);\s*real veff = \(is_switch == 1\) \? veff_raw : -veff_raw;\s*real w_eff = \(is_switch == 1\) \? w_start : \(1\.0 - w_start\);\s*log_lik\[t\] = wiener_lpdf\(rt\[t\] \| a_dyn, tnd_s, w_eff, veff\);\s*if \(veff_raw == 0\.0\) \{\s*pred_sw\[t\] = w_start;\s*\} else \{\s*pred_sw\[t\] = \(exp\(-2\.0 \* veff_raw \* a_dyn \* w_start\) - 1\.0\) / \(exp\(-2\.0 \* veff_raw \* a_dyn\) - 1\.0\);\s*\}\s*\}\s*\}', re.MULTILINE)

    new_gq = """if (t > start_idx[s]) {
             int prev_ch = resp[t-1];
             if (ch > 0 && rt[t] > 0.0) {
                 real veff_raw = v_s * Q_diff;
                 real veff = (ch == 1) ? veff_raw : -veff_raw;
                 real w_eff = (ch == 1) ? w_start : (1.0 - w_start);
                 log_lik[t] = wiener_lpdf(rt[t] | a_dyn, tnd_s, w_eff, veff);
                 
                 if (veff_raw == 0.0) {
                     pred_sw[t] = w_start;
                 } else {
                     real p_left = (exp(-2.0 * veff_raw * a_dyn * w_start) - 1.0) / (exp(-2.0 * veff_raw * a_dyn) - 1.0);
                     if (prev_ch > 0) {
                         pred_sw[t] = (prev_ch == 1) ? (1.0 - p_left) : p_left;
                     } else {
                         pred_sw[t] = 0.5;
                     }
                 }
             }
          }"""
          
    content = old_gq.sub(new_gq, content)

    with open(filename, 'w') as f:
        f.write(content)
        
print("Replaced with regex!")
