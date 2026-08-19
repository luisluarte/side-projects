dataset_path <- "C:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv"
dat <- read.csv(dataset_path)

dt <- numeric()
for (p in unique(dat$participant_id)) {
  p_data <- dat[dat$participant_id == p, ]
  p_data <- p_data[order(p_data$ttp), ]
  dt <- c(dt, diff(p_data$ttp))
}

dt <- dt[dt > 0 & dt < 100000] # remove extremely absurd values if any
cat(sprintf("Median dt: %.2f ms\n", median(dt)))
cat(sprintf("Mean dt: %.2f ms\n", mean(dt)))
cat(sprintf("Q3 dt: %.2f ms\n", quantile(dt, 0.75)))
cat(sprintf("99th percentile: %.2f ms\n", quantile(dt, 0.99)))
cat(sprintf("Proportion > 10s: %.4f\n", mean(dt > 10000)))
cat(sprintf("Proportion > 5s: %.4f\n", mean(dt > 5000)))
