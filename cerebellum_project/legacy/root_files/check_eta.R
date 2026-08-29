file_path <- "results/tables/thermo_sudoku_sweep.csv"
total_evals <- 100 * 128

if(!file.exists(file_path)) {
    cat("Sweep not started.\n")
    quit()
}

info <- file.info(file_path)
# ctime is creation/change time. If it's constantly modified, mtime updates. 
# We'll use the log file creation time instead for absolute start.
log_path <- "thermo_sudoku.log"
if(file.exists(log_path)) {
    start_time <- file.info(log_path)$ctime
} else {
    start_time <- info$ctime
}

current_time <- Sys.time()
df <- tryCatch(read.csv(file_path), error=function(e) data.frame())
current_evals <- nrow(df)

if(current_evals == 0) {
    cat("0 evaluations completed.\n")
    quit()
}

elapsed_secs <- as.numeric(difftime(current_time, start_time, units="secs"))
if (elapsed_secs <= 0) elapsed_secs <- 1 # safety

rate <- current_evals / elapsed_secs
remaining_evals <- total_evals - current_evals
eta_secs <- remaining_evals / rate

cat("=== THERMODYNAMIC SWEEP ETA ===\n")
cat(sprintf("Participants Processed: %d / %d (%.2f%%)\n", current_evals, total_evals, 100 * current_evals / total_evals))
cat(sprintf("Throughput: %.2f optimizations/minute\n", rate * 60))
cat(sprintf("Elapsed Time: %.2f mins\n", elapsed_secs / 60))
cat(sprintf("Estimated Time Remaining: %.2f mins (%.2f hours)\n", eta_secs / 60, eta_secs / 3600))
