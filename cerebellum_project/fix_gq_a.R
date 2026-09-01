lines <- readLines("src/models/q_learning_ddm_dynamic_gq.stan")
text <- paste(lines, collapse="\n")
text <- gsub("real a_t = a_s + beta_a_s \\* tanh\\(theta_ctx_s \\* iti\\[t\\]\\);", "real a_t = a_s + beta_a_s * tanh(theta_ctx_s * iti[t]);\n      if (a_t < 0.01) a_t = 0.01;", text)
writeLines(text, "src/models/q_learning_ddm_dynamic_gq.stan")

lines <- readLines("src/models/q_learning_ddm_dynamic.stan")
text <- paste(lines, collapse="\n")
text <- gsub("real a_t = a_s \\+ beta_a_s \\* tanh\\(theta_ctx_s \\* iti\\[t\\]\\);", "real a_t = a_s + beta_a_s * tanh(theta_ctx_s * iti[t]);\n        if (a_t < 0.01) a_t = 0.01;", text)
writeLines(text, "src/models/q_learning_ddm_dynamic.stan")
