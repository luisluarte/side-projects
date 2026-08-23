library(ggplot2)

delta <- seq(0, 1.5, length.out=500)

# 1. Standard Exponential (What we have now)
gamma <- 3
exp_supp <- exp(-gamma * delta)

# 2. Hill Equation (Sharp Bifurcation / Ultrasensitivity)
theta <- 0.5
n <- 8
hill_supp <- (theta^n) / (theta^n + delta^n)

# 3. True Hysteresis (Schmitt Trigger / Absorbing Wells)
# For plot illustration, we'll draw two paths
hyst_up <- ifelse(delta > 0.8, 0, 1)   # Moving from Agree -> Disagree
hyst_down <- ifelse(delta > 0.2, 0, 1) # Moving from Disagree -> Agree

df_exp <- data.frame(delta = delta, S = exp_supp, type = "1. Exponential Decay (Current)")
df_hill <- data.frame(delta = delta, S = hill_supp, type = "2. Hill Equation (Bistable Switch)")

df_hyst_up <- data.frame(delta = delta, S = hyst_up, type = "3. Hysteresis (True Absorbing Wells)")
df_hyst_down <- data.frame(delta = delta, S = hyst_down, type = "3. Hysteresis (True Absorbing Wells)")

p <- ggplot() +
  geom_line(data=df_exp, aes(x=delta, y=S), color="blue", size=1.5) +
  geom_line(data=df_hill, aes(x=delta, y=S), color="red", size=1.5) +
  geom_step(data=df_hyst_up, aes(x=delta, y=S), color="purple", size=1.5, linetype="solid") +
  geom_step(data=df_hyst_down, aes(x=delta, y=S), color="purple", size=1.5, linetype="dashed") +
  facet_wrap(~type, ncol=1) +
  theme_minimal(base_size = 14) +
  labs(
    title = "Mathematical Topologies for Disagreement Modulator",
    x = "Cortico-Cerebellar Disagreement (Delta_CC)",
    y = "Drift Rate Suppression Factor (S)",
    subtitle = "Purple Dashed = Path to return to Agreement. Solid = Path to enter Disagreement."
  )
ggsave("C:/Users/DCCS5/.gemini/antigravity/brain/1d8f9958-fd49-4502-b57b-97a7887eb7ad/double_well_plot.png", p, width=7, height=8)
