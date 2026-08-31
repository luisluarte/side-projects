library(cmdstanr)
library(dplyr)

fit_base <- readRDS("/home/DCCS5/cerebellum_project/results/baseline_final.rds")
fit_m006 <- readRDS("/home/DCCS5/cerebellum_project/results/m006_final.rds")

sum_base <- fit_base$summary()
sum_m006 <- fit_m006$summary()

get_pop_mean <- function(summ, pattern, n=10) {
    res <- numeric(n)
    for(i in 1:n) {
        row <- summ %>% filter(variable == paste0(pattern, "[", i, "]"))
        res[i] <- row$mean
    }
    return(mean(res))
}
get_pop_sd <- function(summ, pattern, n=10) {
    res <- numeric(n)
    for(i in 1:n) {
        row <- summ %>% filter(variable == paste0(pattern, "[", i, "]"))
        res[i] <- row$mean
    }
    return(sd(res))
}

b_a <- get_pop_mean(sum_base, "a")
b_tnd <- get_pop_mean(sum_base, "tnd")
b_v <- get_pop_mean(sum_base, "v_ctx")
b_alpha <- get_pop_mean(sum_base, "alpha_ctx")

m_a_raw <- get_pop_mean(sum_m006, "a_base_raw")
m_a_base <- 0.11 + 7.36 * (1.0 / (1.0 + exp(-m_a_raw))) # apply inv logit for display
m_tnd <- get_pop_mean(sum_m006, "tnd")
m_v <- get_pop_mean(sum_m006, "v_ctx")
m_alpha <- get_pop_mean(sum_m006, "alpha_ctx")

m_alpha_pc <- get_pop_mean(sum_m006, "alpha_pc")
m_gamma <- get_pop_mean(sum_m006, "gamma_var")
m_golgi <- get_pop_mean(sum_m006, "golgi_scale")
m_tau <- get_pop_mean(sum_m006, "tau_decay")
m_wu <- get_pop_mean(sum_m006, "w_u")

cat("\n| Parameter | Baseline (Mean) | M006 (Mean) |\n")
cat("| :--- | :--- | :--- |\n")
cat(sprintf("| **Boundary (a)** | %.3f | %.3f (Base) |\n", b_a, m_a_base))
cat(sprintf("| **Non-Decision Time (tnd)** | %.3f | %.3f |\n", b_tnd, m_tnd))
cat(sprintf("| **Drift Rate (v_ctx)** | %.3f | %.3f |\n", b_v, m_v))
cat(sprintf("| **Learning Rate (alpha_ctx)** | %.3f | %.3f |\n", b_alpha, m_alpha))
cat(sprintf("| **Purkinje Learning (alpha_pc)** | - | %.3f |\n", m_alpha_pc))
cat(sprintf("| **Cerebellar Gain (gamma)** | - | %.3f |\n", m_gamma))
cat(sprintf("| **Golgi Scale** | - | %.3f |\n", m_golgi))
cat(sprintf("| **Memory Decay (tau)** | - | %.3f |\n", m_tau))
cat(sprintf("| **Boundary Modulation (w_u)** | - | %.3f |\n", m_wu))