source("src/experiments/run_mega_subject_cmaes.R")
C <- res[["C"]]
sigma_vec <- sqrt(diag(C))
Corr <- C / (sigma_vec %o% sigma_vec)
eig_res <- eigen(Corr, symmetric=TRUE)

cat("\n=== FULL HESSIAN EIGENDECOMPOSITION ===\n")
param_names <- c("a_base", "tnd_raw", "v_ctx", "alpha_ctx", "alpha_pc", "gamma", "golgi", "tau", "w_u")

for (idx in 1:9) {
    cat(sprintf("\n[Eigenvector %d] Lambda = %.4e\n", idx, eig_res[["values"]][idx]))
    vec <- eig_res[["vectors"]][, idx]
    df <- data.frame(Param = param_names, Loading = vec)
    df <- df[order(abs(df[["Loading"]]), decreasing=TRUE), ]
    for (j in 1:9) {
        cat(sprintf("  %12s : %8.4f\n", df[["Param"]][j], df[["Loading"]][j]))
    }
}
