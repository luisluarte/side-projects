import re

with open("src/models/bvk_continuous.stan", "r") as f:
    code = f.read()

# 1. Add parameters
code = re.sub(
    r'real mu_a;', 
    'real mu_a;\n  real mu_beta_a;\n  real mu_kappa_cb;', 
    code
)
code = re.sub(
    r'real<lower=0> sigma_a;', 
    'real<lower=0> sigma_a;\n  real<lower=0> sigma_beta_a;\n  real<lower=0> sigma_kappa_cb;', 
    code
)
code = re.sub(
    r'vector\[S\] z_a;', 
    'vector[S] z_a;\n  vector[S] z_beta_a;\n  vector[S] z_kappa_cb;', 
    code
)

# 2. Add transformed parameters
code = re.sub(
    r'vector\[S\] a = 0.5 \+ 4.5 \* inv_logit\(mu_a \+ sigma_a \* z_a\);', 
    'vector[S] a = 0.5 + 4.5 * inv_logit(mu_a + sigma_a * z_a);\n  vector[S] beta_a = 5.0 * inv_logit(mu_beta_a + sigma_beta_a * z_beta_a);\n  vector[S] kappa_cb = 10.0 * inv_logit(mu_kappa_cb + sigma_kappa_cb * z_kappa_cb);', 
    code
)

# 3. Add model priors
code = re.sub(
    r'mu_a ~ normal\(0, 1\);', 
    'mu_a ~ normal(0, 1);\n  mu_beta_a ~ normal(0, 1.5);\n  mu_kappa_cb ~ normal(0, 1.5);', 
    code
)
code = re.sub(
    r'sigma_a ~ normal\(0, 1\);', 
    'sigma_a ~ normal(0, 1);\n  sigma_beta_a ~ normal(0, 1);\n  sigma_kappa_cb ~ normal(0, 1);', 
    code
)
code = re.sub(
    r'z_a ~ std_normal\(\);', 
    'z_a ~ std_normal();\n  z_beta_a ~ std_normal();\n  z_kappa_cb ~ std_normal();', 
    code
)

# 4. Modify partial_sum signature
code = re.sub(
    r'vector gamma_suppress, vector a, vector tau_nd', 
    'vector gamma_suppress, vector a, vector tau_nd, vector beta_a, vector kappa_cb', 
    code
)

# 5. Modify partial_sum body
code = re.sub(
    r'array\[n_trials\] real w_bias_subj;', 
    'array[n_trials] real w_bias_subj;\n      array[n_trials] real a_subj;', 
    code
)
code = re.sub(
    r'real gamma_suppress_s = gamma_suppress\[s\];', 
    'real gamma_suppress_s = gamma_suppress[s];\n      real beta_a_s = beta_a[s];\n      real kappa_cb_s = kappa_cb[s];', 
    code
)

code = re.sub(
    r'real v_base = kappa_ctx_s \* delta_Q_ctx;', 
    'real v_base = kappa_ctx_s * delta_Q_ctx + kappa_cb_s * delta_Q_cb;', 
    code
)

code = re.sub(
    r'if \(choice\[t\] == 1\) \{', 
    'real a_effective = a_s + beta_a_s * conflict;\n\n        if (choice[t] == 1) {', 
    code
)
code = re.sub(
    r'w_bias_subj\[idx\] = w_bias;\n          v_subj\[idx\] = v_effective;', 
    'w_bias_subj[idx] = w_bias;\n          v_subj[idx] = v_effective;\n          a_subj[idx] = a_effective;', 
    code
)
code = re.sub(
    r'w_bias_subj\[idx\] = 1.0 - w_bias;\n          v_subj\[idx\] = -v_effective;', 
    'w_bias_subj[idx] = 1.0 - w_bias;\n          v_subj[idx] = -v_effective;\n          a_subj[idx] = a_effective;', 
    code
)
code = re.sub(
    r'target_sum \+= wiener_lpdf\(rt_subj \| a\[s\], tau_nd\[s\], w_bias_subj, v_subj\);', 
    'target_sum += wiener_lpdf(rt_subj | a_subj, tau_nd[s], w_bias_subj, v_subj);', 
    code
)

# 6. Update reduce_sum call
code = re.sub(
    r'gamma_suppress, a, tau_nd\n  \);', 
    'gamma_suppress, a, tau_nd, beta_a, kappa_cb\n  );', 
    code
)

with open("src/models/bvk_full_gating.stan", "w") as f:
    f.write(code)

print("Generated src/models/bvk_full_gating.stan")
