library(cmaes)
library(Rcpp)
library(dplyr)

# 1. Load the exact same 50 participants
dataset_path <- "data/raw/behavioral_compilate.csv"
if (!file.exists(dataset_path)) {
  dataset_path <- "c:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv"
}

dat_all <- read.csv(dataset_path)
dat_all[['RT']] <- (as.numeric(dat_all[['ttr']]) - as.numeric(dat_all[['ttp']])) / 1000.0
dat_all <- dat_all[!is.na(dat_all[['RT']]) & dat_all[['RT']] >= 0.1 & dat_all[['RT']] <= 3.0 & dat_all[['Resp']] %in% c(1, 2), ]

set.seed(456) 
sample_participants <- sample(unique(dat_all[['participant_id']]), 50)
dat_all <- dat_all[dat_all[['participant_id']] %in% sample_participants, ]
dat_all$participant_factor <- as.integer(as.factor(dat_all$participant_id))

dat_all <- dat_all %>%
  group_by(participant_id) %>%
  arrange(ttp) %>%
  mutate(
    trial_idx = row_number(),
    n_trials = n(),
    is_test = ifelse(trial_idx > 0.7 * n_trials, 1, 0)
  ) %>%
  ungroup() %>%
  arrange(participant_factor, ttp)

# 2. C++ Engine for M1 and M2 70/30
cpp_code <- '
#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace Rcpp;

inline double wiener_pdf(double t_raw, int choice, double v, double a, double t_nd, double eps = 1e-9) {
  double t = t_raw - t_nd;
  if (t <= 0.001) return 1e-12;
  double sign = (choice == 1) ? 1.0 : -1.0;
  double w = 0.50; 
  double x0 = (choice == 1) ? (1.0 - w) : w;
  double drift_term = std::exp(sign * v * a * w - 0.5 * v * v * t);
  double tt = t / (a * a);
  double sum = 0.0;
  if (tt >= 0.08) {
    for (int k = 1; k <= 30; ++k) {
      double term = (double)k * std::sin((double)k * M_PI * x0) * std::exp(-0.5 * k * k * M_PI * M_PI * tt);
      sum += term;
      if (std::abs(term) < eps) break;
    }
    sum *= M_PI;
  } else {
    double sqrt_tt = std::sqrt(tt);
    for (int k = -15; k <= 15; ++k) {
      double num = (x0 + 2.0 * k);
      double term = num * std::exp(-0.5 * (num * num) / tt);
      sum += term;
    }
    sum /= (std::sqrt(2.0 * M_PI) * tt * sqrt_tt);
  }
  double pdf_val = (drift_term / (a * a)) * sum;
  return std::max(1e-12, pdf_val);
}

// [[Rcpp::export]]
NumericVector evaluate_m1_cv_cpp(const NumericVector& phi, const IntegerVector& resp, const IntegerVector& is_test, const NumericVector& rt, const IntegerVector& subj_idx) {
    int N_t = resp.size();
    double b_v=phi[0], a_0=phi[1], t_nd=phi[2], p_sw=phi[3];
    double train_nll = 0.0;
    std::vector<double> test_nlls(51, 0.0);
    
    for(int t=0; t<N_t; ++t) {
        int ch = resp[t];
        int prev_ch = (t>0 && subj_idx[t]==subj_idx[t-1]) ? resp[t-1] : 1;
        
        double v_t = 0.0;
        if(ch == prev_ch) { v_t = b_v * (1.0 - p_sw); }
        else { v_t = -b_v * p_sw; }
        
        double dens = wiener_pdf(rt[t], ch, v_t, a_0, t_nd);
        double nll_t = -std::log(dens);
        
        if (is_test[t] == 0) train_nll += nll_t;
        else test_nlls[subj_idx[t]] += nll_t;
    }
    if (std::isnan(train_nll) || std::isinf(train_nll)) train_nll = 1e9;
    test_nlls[0] = train_nll; 
    return wrap(test_nlls);
}

// [[Rcpp::export]]
NumericVector evaluate_m2_cv_cpp(const NumericVector& phi, const IntegerVector& resp, const IntegerVector& out, const IntegerVector& is_test, const NumericVector& rt, const IntegerVector& subj_idx) {
    int N_t = resp.size();
    double b_v=phi[0], a_0=phi[1], t_nd=phi[2], alpha=phi[3];
    double train_nll = 0.0;
    std::vector<double> test_nlls(51, 0.0);
    
    double q1 = 0.5, q2 = 0.5;
    
    for(int t=0; t<N_t; ++t) {
        if(t>0 && subj_idx[t]!=subj_idx[t-1]) { q1=0.5; q2=0.5; }
        int ch = resp[t], o = out[t];
        
        double v_t = b_v * (q1 - q2);
        double dens = wiener_pdf(rt[t], ch, v_t, a_0, t_nd);
        double nll_t = -std::log(dens);
        
        if (is_test[t] == 0) train_nll += nll_t;
        else test_nlls[subj_idx[t]] += nll_t;
        
        if (is_test[t] == 0) {
            double target = (o == 1) ? 1.0 : -1.0;
            if(ch == 1) { q1 += alpha*(target - q1); q2 += alpha*(-target - q2); }
            else { q2 += alpha*(target - q2); q1 += alpha*(-target - q1); }
        }
    }
    if (std::isnan(train_nll) || std::isinf(train_nll)) train_nll = 1e9;
    test_nlls[0] = train_nll; 
    return wrap(test_nlls);
}
'
sourceCpp(code = cpp_code)

# 3. Load Combined Results
results_df <- read.csv("results/tables/eccm_topological_ablation_50_subj.csv")

# 4. Optimize M1
cat("Training M1 (WSLS)...\\n")
lb_m1 <- c(0.0, 0.3, 0.1, 0.0)
ub_m1 <- c(3.0, 2.5, 0.9, 1.0)
obj_m1 <- function(phi) {
    if (any(phi < lb_m1) || any(phi > ub_m1)) return(1e9)
    res <- evaluate_m1_cv_cpp(phi, as.integer(dat_all$Resp), as.integer(dat_all$is_test), dat_all$RT, dat_all$participant_factor)
    return(res[1])
}
cma_m1 <- cma_es(lb_m1 + (ub_m1 - lb_m1)/2, obj_m1, lower=lb_m1, upper=ub_m1, control=list(maxit=50, trace=FALSE))
m1_test <- evaluate_m1_cv_cpp(cma_m1$par, as.integer(dat_all$Resp), as.integer(dat_all$is_test), dat_all$RT, dat_all$participant_factor)[2:51]

# 5. Optimize M2
cat("Training M2 (RWCF)...\\n")
lb_m2 <- c(0.0, 0.3, 0.1, 0.0)
ub_m2 <- c(3.0, 2.5, 0.9, 1.0)
obj_m2 <- function(phi) {
    if (any(phi < lb_m2) || any(phi > ub_m2)) return(1e9)
    res <- evaluate_m2_cv_cpp(phi, as.integer(dat_all$Resp), as.integer(dat_all$F), as.integer(dat_all$is_test), dat_all$RT, dat_all$participant_factor)
    return(res[1])
}
cma_m2 <- cma_es(lb_m2 + (ub_m2 - lb_m2)/2, obj_m2, lower=lb_m2, upper=ub_m2, control=list(maxit=50, trace=FALSE))
m2_test <- evaluate_m2_cv_cpp(cma_m2$par, as.integer(dat_all$Resp), as.integer(dat_all$F), as.integer(dat_all$is_test), dat_all$RT, dat_all$participant_factor)[2:51]

results_df$M1_NLL <- m1_test
results_df$M2_NLL <- m2_test

cat("\\n======================================\\n")
cat("AGGREGATE TEST NLL ACROSS 50 PARTICIPANTS\\n")
cat(sprintf("M1 (WSLS) : %.2f\\n", sum(results_df$M1_NLL)))
cat(sprintf("M2 (RWCF) : %.2f\\n", sum(results_df$M2_NLL)))
cat(sprintf("Combined  : %.2f\\n", sum(results_df$Combined_NLL)))

cat("\\n--- PAIRED T-TESTS (Participant Level) ---\\n")
t1 <- t.test(results_df$M1_NLL, results_df$Combined_NLL, paired=TRUE)
cat(sprintf("M1 vs Combined: t = %.2f, p = %.3e (Mean Diff: %.2f)\\n", t1$statistic, t1$p.value, t1$estimate))

t2 <- t.test(results_df$M2_NLL, results_df$Combined_NLL, paired=TRUE)
cat(sprintf("M2 vs Combined: t = %.2f, p = %.3e (Mean Diff: %.2f)\\n", t2$statistic, t2$p.value, t2$estimate))

write.csv(results_df, "results/tables/eccm_topological_ablation_50_subj.csv", row.names=FALSE)
