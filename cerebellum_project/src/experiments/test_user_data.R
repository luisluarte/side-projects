
pacman::p_load(cmdstanr, posterior, jsonlite)
fit_q <- readRDS('../../results/fit_q_complete.rds')
data_path <- fit_q[['data_file']]()
cat('Data path:', data_path, '\n')
if (file.exists(data_path)) {
  cat('Data file exists!\n')
} else {
  cat('Data file missing.\n')
}

