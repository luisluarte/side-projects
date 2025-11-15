# === MAIN EXECUTION PIPELINE ===

# 1. Load Libraries
pacman::p_load(
  tidyverse,
  ggplot2,
  svglite,
  this.path
)

args <- commandArgs(trailingOnly = TRUE)
options(browser = "firefox")

# 2. Source All HMM Morphisms
setwd(this.path::here()) # Set working directory if needed
source("hmm_funcs.R")

# get data
dat <- read_csv("final_probabilities_with_dates_prices.csv") %>%
  pivot_longer(
    cols = c(bull, bear, sideways)
  ) %>%
  group_by(Date) %>%
  filter(value == max(value))
print(dat)

p1 <- dat %>%
  ggplot(aes(
    Date, log(Close)
  )) +
  geom_line(aes(color = name, group = 1))
ggsave(filename = "p1.svg",
       plot = p1)
browseURL("p1.svg")
