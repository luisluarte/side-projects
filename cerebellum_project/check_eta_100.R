# ETA Calculator for 100-Model WFPT Sweep
# Reads the incremental output CSV and estimates remaining time

args <- commandArgs(trailingOnly=TRUE)
out_file <- ifelse(length(args)>0, args[1], "results/tables/wfpt_100_terminal.csv")

if(!file.exists(out_file)) {
    cat("Output file not yet created. Sweep has not started writing results.\n")
    q(save="no")
}

df <- tryCatch(read.csv(out_file), error=function(e) data.frame())
if(nrow(df)==0) {
    cat("Output file exists but is empty (header only). Baseline still running.\n")
    q(save="no")
}

total_models <- 100
completed <- nrow(df)
remaining <- total_models - completed

# Get file modification timestamps to estimate per-model time
finfo <- file.info(out_file)
elapsed_sec <- as.numeric(difftime(Sys.time(), finfo$ctime, units="secs"))

if(completed > 0) {
    per_model_sec <- elapsed_sec / completed
    eta_sec <- per_model_sec * remaining
    eta_min <- eta_sec / 60
    
    cat(sprintf("==========================================\n"))
    cat(sprintf("WFPT 100-Model Sweep Progress\n"))
    cat(sprintf("==========================================\n"))
    cat(sprintf("Models completed : %d / %d (%.0f%%)\n", completed, total_models, 100*completed/total_models))
    cat(sprintf("Elapsed time     : %.1f minutes\n", elapsed_sec/60))
    cat(sprintf("Per-model time   : %.1f seconds\n", per_model_sec))
    cat(sprintf("Remaining models : %d\n", remaining))
    cat(sprintf("Estimated ETA    : %.1f minutes (%.1f hours)\n", eta_min, eta_min/60))
    cat(sprintf("==========================================\n"))
    
    # Show last 3 completed models
    cat("\nLast 3 completed:\n")
    tail_df <- tail(df, 3)
    for(i in 1:nrow(tail_df)) {
        r <- tail_df[i,]
        cat(sprintf("  %s: NLL=%.2f | ROC=%.3f | RMSE=%.3f\n", r$ModelID, r$Mean_NLL, r$Mean_ROC, r$Mean_RMSE))
    }
} else {
    cat("No models completed yet.\n")
}
