# install packages
pacman::p_load(
  tidyquant,
  tidyverse,
  this.path
)

# go to file path
setwd(this.path::here())
cat(paste0("Current dir is: ", getwd(), "\n"))


# metadata stuff ----
current_date <- Sys.Date()
cat(paste0("Current date is: ", current_date, "\n"))

filename <- paste0("./data/", "sp500_data_raw_", current_date, ".csv")
cat(paste0("Filename is: ", filename, "\n"))

# data ----
if (file.exists(filename)) {
  stop("File already exists skipping download...\n")
}

cat("fetching sp500 data...\n")
sp500_data_raw <- tq_get(
  "^GSPC",
  from = "2000-01-01",
  to = Sys.Date()
)
cat("data downloaded...\n")
print(head(sp500_data_raw))

# save data ----
write_csv(sp500_data_raw, file = filename)
