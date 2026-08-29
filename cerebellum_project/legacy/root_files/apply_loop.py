import re

with open("src/models/bvk_full_gating.stan", "r") as f:
    code = f.read()

# Replace the vectorized wiener_lpdf with a loop
code = re.sub(
    r'target_sum \+= wiener_lpdf\(rt_subj \| a_subj, tau_nd\[s\], w_bias_subj, v_subj\);', 
    'for (idx in 1:n_trials) {\n        target_sum += wiener_lpdf(rt_subj[idx] | a_subj[idx], tau_nd[s], w_bias_subj[idx], v_subj[idx]);\n      }', 
    code
)

with open("src/models/bvk_full_gating.stan", "w") as f:
    f.write(code)

print("Loop applied.")
