# setup -------------------------------------------------------------------
local_lib <- Sys.getenv("R_LIBS_USER")

if (!dir.exists(local_lib)) {
  dir.create(local_lib, recursive = TRUE)
}
.libPaths(c(local_lib, .libPaths()))

# libs --------------------------------------------------------------------
cat("LIBS\n")
if (!require("pacman", character.only = TRUE)) {
  install.packages("pacman",
                   lib = local_lib)
  library("pacman", character.only = TRUE)
}
pacman::p_load(
  tidyverse,
  cmdstanr,
  posterior,
  this.path
)

if (!dir.exists(cmdstan_path())) {
  install_cmdstan()
} else {
  message("cmdstan installed at: ", cmdstan_path())
}

setwd(here())

# data --------------------------------------------------------------------





