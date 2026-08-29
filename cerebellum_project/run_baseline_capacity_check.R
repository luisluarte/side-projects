pacman::p_load(tidyverse, Rcpp, cmaes, parallel)

Rcpp::sourceCpp("src/models/ql_baseline_extended.cpp")
Rcpp::sourceCpp("src/models/eccm_magi_poly.cpp")

dat_raw <- read_csv("data/raw/behavioral_compilate.csv", show_col_types = FALSE)
dat_clean <- dat_raw %>% arrange(participant_id, ttp) %>% 
    group_by(participant_id) %>% 
    mutate(RT = (ttr - ttp) / 1000, 
           ITI = (ttp - lag(ttF)) / 1000, 
           F_dur = (ttF - ttr) / 1000, 
           Boundary = ifelse(Resp == 2, 1, 0)) %>% 
    mutate(ITI = ifelse(is.na(ITI), mean(ITI, na.rm = TRUE), ITI), 
           F_dur = ifelse(is.na(F_dur), 0.5, F_dur)) %>% 
    ungroup() %>% 
    filter(RT > 0.1, RT < 3.0, !is.na(Resp), !is.na(`F`))

dat_clean <- dat_clean %>% mutate(participant_idx = as.integer(as.factor(participant_id)))
S <- length(unique(dat_clean$participant_idx))
lambda <- 1.0 
num_cores <- 30
cmaes_maxit <- 100

# Biological Champion Architecture: Topology 3, Euclidean Conflict, HFV Boundary, PH Volatility
genes_champ <- c(3, 1, 0, 3, 1, 1, 20)

cat("Fitting Extended Algorithmic Baseline and Biological Champion...\n")
res_list <- mclapply(1:S, function(s_idx) {
    d <- dat_clean %>% filter(participant_idx == s_idx)
    resp <- d$Boundary + 1
    out <- d$`F`
    rt <- d$RT
    iti <- d$ITI
    f_dur <- d$F_dur
    ttp <- d$ttp
    
    # --- Extended Algorithmic Baseline (12 parameters) ---
    obj_base_ext <- function(phi) { 
        ll <- eval_ql_baseline_extended(phi, resp, out, rt)
        if(is.na(ll) || is.infinite(ll)) return(1e9)
        return(ll + lambda * sum(phi^2)) 
    }
    fit_base <- cma_es(rep(0, 12), obj_base_ext, lower=rep(-5, 12), upper=rep(5, 12), control=list(maxit=cmaes_maxit))
    rt_base <- extract_rt_ql_baseline_extended(fit_base$par, resp, out, rt, FALSE)
    
    # --- Biological Champion (19 parameters) ---
    mask <- rep(1, 19)
    if(genes_champ[3] == 0) mask[13] <- 0
    if(genes_champ[4] == 0) mask[9] <- 0
    if(genes_champ[4] == 1) mask[12] <- 0
    if(genes_champ[4] == 2) mask[16] <- 0
    if(genes_champ[5] == 0) mask[14] <- 0
    if(genes_champ[6] == 0) mask[15] <- 0
    
    obj_champ <- function(phi) {
        p <- phi * mask
        ll <- eval_magi_topo_poly(p, genes_champ, resp, out, rt, iti, f_dur, ttp)
        if(is.na(ll) || is.infinite(ll)) return(1e9)
        return(ll + lambda * sum(p^2))
    }
    fit_champ <- cma_es(rep(0, 19), obj_champ, lower=rep(-5, 19), upper=rep(5, 19), control=list(maxit=cmaes_maxit))
    rt_champ <- extract_topology_poly(fit_champ$par * mask, genes_champ, resp, out, rt, iti, f_dur, ttp, FALSE)
    
    return(data.frame(
        Subject = s_idx, Trial = 1:nrow(d), 
        RT_Emp = rt, RT_BaseExt = rt_base, RT_Champ = rt_champ,
        NLL_BaseExt = fit_base$value, NLL_Champ = fit_champ$value
    ))
}, mc.cores = num_cores)

df_all <- bind_rows(res_list)
write_csv(df_all, "capacity_comparison_results.csv")

summ <- df_all %>% group_by(Subject) %>% 
    summarize(NLL_BaseExt = first(NLL_BaseExt), NLL_Champ = first(NLL_Champ))
cat(sprintf("Mean NLL Baseline Extended: %.2f\n", mean(summ$NLL_BaseExt)))
cat(sprintf("Mean NLL Biological Champion: %.2f\n", mean(summ$NLL_Champ)))

df_lmm <- df_all %>% filter(is.finite(RT_BaseExt) & RT_BaseExt > 0 & RT_BaseExt < 10) %>%
                     filter(is.finite(RT_Champ) & RT_Champ > 0 & RT_Champ < 10)

lm_base <- lm(RT_Emp ~ RT_BaseExt, data=df_lmm)
lm_champ <- lm(RT_Emp ~ RT_Champ, data=df_lmm)
cat(sprintf("Beta Baseline Extended: %.4f\n", coef(lm_base)["RT_BaseExt"]))
cat(sprintf("Beta Biological Champion: %.4f\n", coef(lm_champ)["RT_Champ"]))

# Generate plot
png("rt_predicted_vs_empirical_capacity_check.png", width=800, height=400)
par(mfrow=c(1,2))
plot(df_lmm$RT_BaseExt, df_lmm$RT_Emp, pch=16, col=rgb(0,0,1,0.1),
     xlab="Predicted RT", ylab="Empirical RT", main="Extended Algorithmic Baseline")
abline(a=0, b=1, col="red", lwd=2)

plot(df_lmm$RT_Champ, df_lmm$RT_Emp, pch=16, col=rgb(0,0.5,0,0.1),
     xlab="Predicted RT", ylab="Empirical RT", main="Biological Champion")
abline(a=0, b=1, col="red", lwd=2)
dev.off()
cat("Analysis Complete.\n")
