# libs --------------------------------------------------------------------
pacman::p_load(
  tidyverse,
  ggplot2,
  glmmTMB,
  lme4,
  ggpubr
)

setwd(this.path::here())

# data --------------------------------------------------------------------
dat_rt <- read_csv("../../results/tables/magi_raw_trial_simulations_N30_CMAES.csv")
dat_w1 <- read_csv("../../results/tables/magi_raw_subject_W1_N30_CMAES.csv")


# models ------------------------------------------------------------------

# plots -------------------------------------------------------------------

theme_physics <- function(base_size = 12, base_family = "") {
  theme_bw(base_size = base_size, base_family = base_family) %+replace% #
    theme(
      # Remove background grids for a clean look
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),

      # Solid bounding box around the plot
      panel.border = element_rect(colour = "black", fill = NA, linewidth = 1),

      # Adjust axis ticks to be clean and visible
      axis.tick.length = unit(-0.15, "cm"), # Inward ticks (optional, use positive for outward)
      axis.text.x = element_text(margin = margin(t = 0.2, unit = "cm")), # Add spacing for inward ticks
      axis.text.y = element_text(margin = margin(r = 0.2, unit = "cm")),
      axis.ticks = element_line(colour = "black", linewidth = 0.8),

      # Bold and clear text for readability
      axis.title = element_text(size = rel(1.1), face = "bold"),
      axis.text = element_text(size = rel(0.95), colour = "black"),

      # Legend formatting
      legend.background = element_blank(),
      legend.box.background = element_blank(),
      legend.key = element_blank(),
      legend.text = element_text(size = rel(0.9)),
      legend.title = element_text(size = rel(0.9), face = "bold")
    )
}

p1 <- dat_rt %>%
  select(SubjectID, Empirical_RT, Baseline_Wald_Sim_RT, Terminal_Hybrid_Sim_RT) %>%
  pivot_longer(-SubjectID, names_to = "model", values_to = "rt") %>%
  ggplot(aes(
    rt,
    fill = model
  )) +
  geom_density(alpha = 0.5) +
  coord_cartesian(xlim = c(0, 5)) +
  theme_physics() +
  scale_fill_viridis_d()
p1

p2 <- dat_w1 %>%
  pivot_longer(-SubjectID, names_to = "model", values_to = "W1") %>%
  ggplot(aes(
    model, W1,
    fill = model
  )) +
  geom_boxplot(outlier.shape = NA, fill = "white") +
  geom_point(
    shape = 21,
    size = 3,
    alpha = 0.2
  ) +
  geom_line(aes(group = SubjectID), alpha = 0.1) +
  theme_physics() +
  scale_fill_viridis_d()
p2
