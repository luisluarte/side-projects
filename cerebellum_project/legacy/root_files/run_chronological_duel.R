pacman::p_load(tidyverse, dtw, infotheo, zoo, lmerTest)

d <- read_csv("results/tables/magi_raw_trial_simulations_N30.csv", show_col_types = FALSE)

res_dtw <- data.frame()
res_ccf <- data.frame()
res_mi <- data.frame()
d_smooth <- data.frame()
d_quant <- data.frame()

for(s in 1:30) {
    ds <- d %>% filter(SubjectID == s)
    emp <- ds$Empirical_RT
    base_exp <- ds$Baseline_Wald_Expected_RT
    term_exp <- ds$Terminal_Hybrid_Expected_RT
    
    # 1. DTW
    dtw_base <- dtw(emp, base_exp, keep.internals=FALSE)$distance
    dtw_term <- dtw(emp, term_exp, keep.internals=FALSE)$distance
    res_dtw <- bind_rows(res_dtw, data.frame(SubjectID=s, DTW_Base=dtw_base, DTW_Term=dtw_term))
    
    # 2. CCF (max cross-correlation within lag 5)
    ccf_base <- max(ccf(emp, base_exp, lag.max=5, plot=FALSE)$acf)
    ccf_term <- max(ccf(emp, term_exp, lag.max=5, plot=FALSE)$acf)
    res_ccf <- bind_rows(res_ccf, data.frame(SubjectID=s, CCF_Base=ccf_base, CCF_Term=ccf_term))
    
    # 3. Mutual Information (Discretized)
    nbins <- 10
    emp_bin <- discretize(emp, disc="equalfreq", nbins=nbins)
    base_bin <- discretize(base_exp, disc="equalfreq", nbins=nbins)
    term_bin <- discretize(term_exp, disc="equalfreq", nbins=nbins)
    mi_base <- mutinformation(emp_bin, base_bin)
    mi_term <- mutinformation(emp_bin, term_bin)
    res_mi <- bind_rows(res_mi, data.frame(SubjectID=s, MI_Base=mi_base, MI_Term=mi_term))
    
    # Smooth data for regression
    sm_emp <- rollmean(emp, k=5, fill=NA)
    sm_base <- rollmean(base_exp, k=5, fill=NA)
    sm_term <- rollmean(term_exp, k=5, fill=NA)
    d_smooth <- bind_rows(d_smooth, data.frame(SubjectID=s, emp=sm_emp, base=sm_base, term=sm_term))
    
    # Quantile bins
    ds <- ds %>% mutate(
        emp_q = ntile(Empirical_RT, 10),
        base_q = ntile(Baseline_Wald_Expected_RT, 10),
        term_q = ntile(Terminal_Hybrid_Expected_RT, 10)
    )
    d_quant <- bind_rows(d_quant, ds)
}

cat("=== 1. Spectral Low-Pass Regression (Smoothed) ===\n")
mod_sm_base <- lmer(emp ~ base + (1|SubjectID), data=d_smooth %>% drop_na())
mod_sm_term <- lmer(emp ~ term + (1|SubjectID), data=d_smooth %>% drop_na())
print(summary(mod_sm_base)$coefficients)
print(summary(mod_sm_term)$coefficients)

cat("\n=== 2. Quantile State Calibration ===\n")
mod_q_base <- lmer(emp_q ~ base_q + (1|SubjectID), data=d_quant)
mod_q_term <- lmer(emp_q ~ term_q + (1|SubjectID), data=d_quant)
print(summary(mod_q_base)$coefficients)
print(summary(mod_q_term)$coefficients)

cat("\n=== 3. Dynamic Time Warping (DTW Distance) ===\n")
cat("Mean DTW Base:", mean(res_dtw$DTW_Base), "| Mean DTW Term:", mean(res_dtw$DTW_Term), "\n")
t_dtw <- t.test(res_dtw$DTW_Base, res_dtw$DTW_Term, paired=TRUE)
print(t_dtw)

cat("\n=== 4. Cross-Correlation Function (CCF Peak) ===\n")
cat("Mean CCF Base:", mean(res_ccf$CCF_Base), "| Mean CCF Term:", mean(res_ccf$CCF_Term), "\n")
t_ccf <- t.test(res_ccf$CCF_Base, res_ccf$CCF_Term, paired=TRUE)
print(t_ccf)

cat("\n=== 5. Mutual Information ===\n")
cat("Mean MI Base:", mean(res_mi$MI_Base), "| Mean MI Term:", mean(res_mi$MI_Term), "\n")
t_mi <- t.test(res_mi$MI_Base, res_mi$MI_Term, paired=TRUE)
print(t_mi)
