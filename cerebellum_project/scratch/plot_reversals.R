library(ggplot2)

df <- read.csv("results/tables/reversal_kinematics_summary.csv")

# Create a bar plot
p <- ggplot(df, aes(x = group, y = disagreement, fill = group)) +
  geom_bar(stat = "identity", width = 0.6, color = "black") +
  theme_minimal() +
  scale_fill_manual(values = c("Pause_Volatile" = "#e74c3c", 
                               "Continuous_Reversal" = "#f39c12", 
                               "Pause_Stationary" = "#3498db")) +
  labs(title = "Cortico-Cerebellar Disagreement Following Events",
       x = "Event Type",
       y = "Mean Disagreement (|Q_CTX - Q_CB|)") +
  theme(legend.position = "none",
        text = element_text(size = 14)) +
  scale_x_discrete(labels = c("Continuous Reversal", "Stationary Pause", "Volatile Pause"))

ggsave("results/figures/reversal_kinematics_plot.png", plot = p, width = 8, height = 6)
