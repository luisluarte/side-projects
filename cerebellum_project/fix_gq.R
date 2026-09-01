lines <- readLines("src/models/bvk_full_gating_gq.stan")
# Let's just use string replacement in R
text <- paste(lines, collapse="\n")
old_str <- "log_lik[t] = wiener_lpdf(rt[t] | a_effective, tau_nd_s, w_bias_subj, v_subj);"
new_str <- "real log_uniform_dens = log(1.0 / 5.8);\n      real wiener_lp;\n      if (rt[t] - tau_nd_s < 1e-4) {\n        log_lik[t] = log_uniform_dens;\n      } else {\n        wiener_lp = wiener_lpdf(rt[t] | a_effective, tau_nd_s, w_bias_subj, v_subj);\n        log_lik[t] = log_mix(0.98, wiener_lp, log_uniform_dens);\n      }"
text <- gsub(old_str, new_str, text, fixed=TRUE)
writeLines(text, "src/models/bvk_full_gating_gq.stan")
