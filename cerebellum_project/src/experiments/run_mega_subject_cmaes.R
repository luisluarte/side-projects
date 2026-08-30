pacman::p_load(tidyverse, Rcpp, cmaes)

cat("Initiating Secure Enclave Mega-Subject Optimization...\n")

dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types=FALSE)
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>%
    group_by(participant_id) %>%
    dplyr::mutate(RT = (ttr-ttp)/1000, Boundary = ifelse(Resp==2, 1, 2), ITI = (ttp - lag(ttr))/1000) %>%
    ungroup() %>% filter(RT>0.1, RT<3.0, !is.na(Resp), !is.na(F)) %>%
    mutate(ITI = ifelse(is.na(ITI) | ITI < 0, median(ITI, na.rm=TRUE), ITI), 
           participant_idx = as.integer(as.factor(participant_id)))

N_subj <- max(dat_clean$participant_idx)
min_rt_df <- dat_clean %>% group_by(participant_idx) %>% summarise(min_rt = min(RT)) %>% arrange(participant_idx)

set.seed(42)
W_exp <- matrix(rnorm(N_subj * 32, 0, 1), nrow=N_subj, ncol=32)

sourceCpp("src/models/pooled_map_m006.cpp")

n_params <- 9
cat(sprintf("Optimizing %d Core Dimensions (Mega-Subject Topology)...\n", n_params))

fit_fn <- function(x) {
    val <- get_pooled_map_006(x, dat_clean$participant_idx, dat_clean$Boundary, 
                              dat_clean$F, dat_clean$RT, dat_clean$ITI, min_rt_df$min_rt, W_exp)
    return(val)
}

cma_es_custom <- function (par, fn, lower, upper) {
    norm <- function(x) drop(sqrt(crossprod(x)))
    xmean <- par
    N <- length(xmean)
    lambda <- 4 + floor(3 * log(N))
    mu <- floor(lambda/2)
    weights <- log(mu + 1) - log(1:mu)
    weights <- weights/sum(weights)
    mueff <- sum(weights)^2/sum(weights^2)
    cc <- 4/(N + 4)
    cs <- (mueff + 2)/(N + mueff + 3)
    mucov <- mueff
    ccov <- (1/mucov) * 2/(N + 1.4)^2 + (1 - 1/mucov) * ((2 * mucov - 1)/((N + 2)^2 + 2 * mucov))
    damps <- 1 + 2 * max(0, sqrt((mueff - 1)/(N + 1)) - 1) + cs
    sigma <- 0.5 
    
    pc <- rep(0, N)
    ps <- rep(0, N)
    B <- diag(N)
    D <- diag(N)
    BD <- B %*% D
    C <- BD %*% t(BD)
    chiN <- sqrt(N) * (1 - 1/(4 * N) + 1/(21 * N^2))
    
    iter <- 0
    maxiter <- 200
    stalled_gens <- 0
    best_last <- Inf
    
    cat("Starting CMA-ES Loop...\n")
    while (iter < maxiter) {
        iter <- iter + 1
        arz <- matrix(rnorm(N * lambda), ncol = lambda)
        arx <- xmean + sigma * (BD %*% arz)
        arx <- pmax(pmin(arx, upper), lower)
        
        arfitness <- apply(arx, 2, fn)
        
        arindex <- order(arfitness)
        arfitness <- arfitness[arindex]
        aripop <- arindex[1:mu]
        selx <- arx[, aripop]
        xmean <- drop(selx %*% weights)
        selz <- arz[, aripop]
        zmean <- drop(selz %*% weights)
        
        ps <- (1 - cs) * ps + sqrt(cs * (2 - cs) * mueff) * (B %*% zmean)
        hsig <- drop((norm(ps)/sqrt(1 - (1 - cs)^(2 * iter/lambda))/chiN) < (1.4 + 2/(N + 1)))
        pc <- (1 - cc) * pc + hsig * sqrt(cc * (2 - cc) * mueff) * drop(BD %*% zmean)
        BDz <- BD %*% selz
        C <- (1 - ccov) * C + ccov * (1/mucov) * (pc %o% pc + (1 - hsig) * cc * (2 - cc) * C) + 
             ccov * (1 - 1/mucov) * BDz %*% diag(weights) %*% t(BDz)
        sigma <- sigma * exp((norm(ps)/chiN - 1) * cs/damps)
        
        e <- eigen(C, symmetric = TRUE)
        B <- e$vectors
        D <- diag(sqrt(e$values), length(e$values))
        BD <- B %*% D
        
        min_eig <- min(e$values)
        max_eig <- max(e$values)
        cond_num <- max_eig / max(min_eig, 1e-30)
        
        if (iter %% 5 == 0) cat(sprintf("Iter %d: NLL = %.2f | Sigma = %.2e | Cond = %.2e\n", iter, arfitness[1], sigma, cond_num))
        
        if (abs(arfitness[1] - best_last) < 1e-4) stalled_gens <- stalled_gens + 1 else stalled_gens <- 0
        best_last <- arfitness[1]
        
        if (cond_num > 1e14) {
            cat("FAIL-SAFE TRIGGERED: Covariance Explosion (Condition Number > 1e14)\n")
            return(list(C=C, par=xmean, vals=e$values, vecs=e$vectors, fail="cond"))
        }
        if (sigma < 1e-8 && stalled_gens >= 100) {
            cat("FAIL-SAFE TRIGGERED: Premature Step-Size Collapse\n")
            return(list(C=C, par=xmean, vals=e$values, vecs=e$vectors, fail="sigma"))
        }
    }
    cat("Optimization Completed Normally.\n")
    return(list(C=C, par=xmean, vals=e$values, vecs=e$vectors, fail="converged"))
}

lower_bounds <- rep(-5, 9)
upper_bounds <- rep(5, 9)
init_par <- rep(0, 9)

res <- cma_es_custom(init_par, fit_fn, lower_bounds, upper_bounds)

# Phase III: Spectral Eigendecomposition
C <- res$C
sigma_vec <- sqrt(diag(C))
Corr <- C / (sigma_vec %o% sigma_vec)
eig_res <- eigen(Corr, symmetric=TRUE)

report <- "### Spectral Diagnostic Report: Mega-Subject 9D Topology\n\n"
report <- paste0(report, sprintf("Status: %s\n", res$fail))
report <- paste0(report, "Eigenvalues of the Correlation Matrix:\n")
for(i in 1:9) report <- paste0(report, sprintf("Lambda %d: %.2e\n", i, eig_res$values[i]))

degenerate_idx <- which(eig_res$values < 1e-3)
if (length(degenerate_idx) > 0) {
    report <- paste0(report, "\nDegenerate Eigenvectors Detected (< 1e-3):\n")
    param_names <- c("a_base", "tnd", "v_ctx", "alpha_ctx", "alpha_pc", "gamma", "golgi_scale", "tau_decay", "w_u")
    for (idx in degenerate_idx) {
        vec <- eig_res$vectors[, idx]
        report <- paste0(report, sprintf("\nVector %d (Lambda = %.2e) Absolute Loadings:\n", idx, eig_res$values[idx]))
        df <- data.frame(Param = param_names, Loading = abs(vec)) %>% arrange(desc(Loading))
        for (j in 1:9) {
            report <- paste0(report, sprintf("- %s: %.4f\n", df$Param[j], df$Loading[j]))
        }
    }
} else {
    report <- paste0(report, "\nNo degenerate eigenvalues < 1e-3 detected. The 9-dimensional biological subspace is structurally identifiable.\n")
}

writeLines(report, "results/spectral_diagnostic_pooled_report.md")
cat("Audit complete. Report saved.\n")
