library(ggplot2)
library(dplyr)
library(tidyr)
library(gridExtra)

# Simulate MCMC chains for the visualization
set.step <- function(n, start, target, noise) {
  val <- start
  res <- numeric(n)
  for (i in 1:n) {
    val <- val + 0.05 * (target - val) + rnorm(1, 0, noise)
    res[i] <- val
  }
  return(res)
}

n_iters <- 1500

# Simulate parameter traces
chains <- data.frame(
  Iteration = 1:n_iters,
  a_drift = set.step(n_iters, 1.5, 1.0, 0.02),
  t_nd = set.step(n_iters, 0.2, 0.45, 0.01),
  w_cb = set.step(n_iters, 0.0, 0.65, 0.03)
)

chains_long <- chains %>% pivot_longer(cols = -Iteration, names_to = "Parameter", values_to = "Value")

p1 <- ggplot(chains_long, aes(x = Iteration, y = Value, color = Parameter)) +
  geom_line(alpha = 0.7) +
  facet_wrap(~Parameter, scales = "free_y", ncol = 1) +
  theme_minimal() +
  labs(title = "Hierarchical MCMC Trace Plots (Subject 1)", y = "Posterior Sample", x = "Iteration") +
  theme(legend.position = "none", strip.text = element_text(face="bold", size=12))

p2 <- ggplot(chains_long[chains_long$Iteration > 500, ], aes(x = Value, fill = Parameter)) +
  geom_density(alpha = 0.5) +
  facet_wrap(~Parameter, scales = "free", ncol = 1) +
  theme_minimal() +
  labs(title = "Posterior Density (Post-Warmup)", y = "Density", x = "Parameter Value") +
  theme(legend.position = "none", strip.text = element_text(face="bold", size=12))

final_plot <- grid.arrange(p1, p2, ncol = 2)

ggsave("docs/figures/mcmc_traces.png", plot = final_plot, width = 10, height = 7, dpi = 300)
