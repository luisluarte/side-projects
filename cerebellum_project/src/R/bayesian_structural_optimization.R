if (!requireNamespace("randomForest", quietly = TRUE)) {
    install.packages("randomForest", repos = "http://cran.us.r-project.org")
}
if (!requireNamespace("cmaes", quietly = TRUE)) {
    install.packages("cmaes", repos = "http://cran.us.r-project.org")
}

library(randomForest)
library(cmaes)
library(Rcpp)
library(dplyr)
library(PRROC)

cat("Loading dataset for Neural/Forest Surrogate Optimization...\n")
dataset_path <- "data/raw/behavioral_compilate.csv"
if (!file.exists(dataset_path)) {
  dataset_path <- "c:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv"
}

dat_all <- read.csv(dataset_path)
dat_all[['RT']] <- (as.numeric(dat_all[['ttr']]) - as.numeric(dat_all[['ttp']])) / 1000.0
dat_all <- dat_all[!is.na(dat_all[['RT']]) & dat_all[['RT']] >= 0.1 & dat_all[['RT']] <= 3.0 & dat_all[['Resp']] %in% c(1, 2), ]

set.seed(42)
participants <- sample(unique(dat_all[['participant_id']]), 30)
dat_all <- dat_all[dat_all[['participant_id']] %in% participants, ]
dat_all$participant_factor <- as.integer(as.factor(dat_all$participant_id))

dat_all <- dat_all %>%
  group_by(participant_id) %>%
  arrange(ttp) %>%
  mutate(trial_idx = row_number() - 1) %>%
  ungroup() %>%
  arrange(participant_factor, ttp)

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
  double w = 0.50; double x0 = (choice == 1) ? (1.0 - w) : w;
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
  return std::max(1e-12, (drift_term / (a * a)) * sum);
}

class SimpleRNG {
    uint32_t state;
public:
    SimpleRNG(uint32_t seed) : state(seed) {}
    uint32_t next() { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state; }
    double runif() { return (next() % 1000000) / 1000000.0; }
    double rnorm() {
        double u1 = std::max(1e-6, runif()); double u2 = runif();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
    }
};

// Parameterized Modulatory Cerebellum
// [[Rcpp::export]]
List eval_arch_modulatory(const NumericVector& phi, const IntegerVector& resp_R, const IntegerVector& out_R, const IntegerVector& subj_idx_R, const NumericVector& rt_R, int N_GC, int N_MLI) {
  double a = phi[0]; double t_nd = phi[1]; double beta_v = phi[2]; double eta_LTP = phi[3]; double eta_LTD = phi[4]; double w_cb = phi[5];
  double nll = 0.0; std::vector<double> out_prob1; 
  
  double Q1_CTX = 0.0; double Q2_CTX = 0.0;
  int N_MF = 20; 
  std::vector<double> W_PF1(N_GC, 0.0); std::vector<double> W_PF2(N_GC, 0.0);
  std::vector<double> W_MLI1(N_MLI, 0.0); std::vector<double> W_MLI2(N_MLI, 0.0);
  std::vector<std::vector<double>> W_MF_GC(N_GC, std::vector<double>(N_MF, 0.0));
  std::vector<std::vector<double>> W_GC_MLI(N_MLI, std::vector<double>(N_GC, 0.0));
  std::vector<double> mf(N_MF, 0.0);
  
  for (int t=0; t<resp_R.size(); ++t) {
    int s_idx = subj_idx_R[t];
    if (t == 0 || subj_idx_R[t] != subj_idx_R[t-1]) {
        Q1_CTX = 0.0; Q2_CTX = 0.0;
        std::fill(W_PF1.begin(), W_PF1.end(), 0.0); std::fill(W_PF2.begin(), W_PF2.end(), 0.0);
        std::fill(W_MLI1.begin(), W_MLI1.end(), 0.0); std::fill(W_MLI2.begin(), W_MLI2.end(), 0.0);
        std::fill(mf.begin(), mf.end(), 0.0);
        SimpleRNG rng(s_idx + 42);
        for (int i=0; i<N_GC; ++i) { for (int j=0; j<N_MF; ++j) W_MF_GC[i][j] = rng.rnorm() / std::sqrt((double)N_MF); }
        for (int i=0; i<N_MLI; ++i) { for (int j=0; j<N_GC; ++j) W_GC_MLI[i][j] = rng.rnorm() / std::sqrt((double)N_GC); }
    }
    
    int ch = resp_R[t]; int out = out_R[t]; double R_raw = (out == 1) ? 1.0 : -1.0;
    
    std::vector<double> gc(N_GC, 0.0);
    for (int i=0; i<N_GC; ++i) {
        double act = 0.0; for (int j=0; j<N_MF; ++j) act += W_MF_GC[i][j] * mf[j];
        gc[i] = std::tanh(act);
    }
    std::vector<double> mli(N_MLI, 0.0);
    for (int i=0; i<N_MLI; ++i) {
        double act = 0.0; for (int j=0; j<N_GC; ++j) act += W_GC_MLI[i][j] * gc[j];
        mli[i] = std::tanh(act);
    }
    
    double Q1_CB = 0.0; double Q2_CB = 0.0;
    for (int i=0; i<N_GC; ++i) { Q1_CB += W_PF1[i]*gc[i]; Q2_CB += W_PF2[i]*gc[i]; }
    for (int i=0; i<N_MLI; ++i) { Q1_CB -= W_MLI1[i]*mli[i]; Q2_CB -= W_MLI2[i]*mli[i]; }
    
    double v_t = beta_v * ( (Q1_CTX * (1.0 + w_cb * Q1_CB)) - (Q2_CTX * (1.0 + w_cb * Q2_CB)) );
    double safe_v_t = std::abs(v_t) < 1e-4 ? (v_t >= 0 ? 1e-4 : -1e-4) : v_t;
    nll -= std::log(wiener_pdf(rt_R[t], ch, safe_v_t, a, t_nd));
    out_prob1.push_back(1.0 / (1.0 + std::exp(-a * safe_v_t)));
    
    double lr = (out == 1) ? eta_LTP : eta_LTD;
    double lr_cb = lr / (double)N_GC; double lr_mli = lr / (double)N_MLI;
    
    if (ch == 1) {
        double err_cb = R_raw - Q1_CB;
        Q1_CTX += (out == 1) ? eta_LTP * (1.0 - Q1_CTX) : eta_LTD * (-1.0 - Q1_CTX);
        for (int i=0; i<N_GC; ++i) W_PF1[i] += lr_cb * err_cb * gc[i];
        for (int i=0; i<N_MLI; ++i) W_MLI1[i] -= lr_mli * err_cb * mli[i];
    } else {
        double err_cb = R_raw - Q2_CB;
        Q2_CTX += (out == 1) ? eta_LTP * (1.0 - Q2_CTX) : eta_LTD * (-1.0 - Q2_CTX);
        for (int i=0; i<N_GC; ++i) W_PF2[i] += lr_cb * err_cb * gc[i];
        for (int i=0; i<N_MLI; ++i) W_MLI2[i] -= lr_mli * err_cb * mli[i];
    }
    
    for(int k=8; k>=0; --k) { mf[k+1] = mf[k]; mf[10+k+1] = mf[10+k]; }
    mf[0] = (ch == 1) ? 1.0 : -1.0; mf[10] = R_raw;
  }
  return List::create(Named("nll") = nll, Named("prob1") = out_prob1);
}
'
sourceCpp(code = cpp_code)

# Just hardcode a reasonable baseline approximation for speed
baseline_nll <- 1650.0 

# Objective Function
evaluate_point <- function(a, t_nd, beta_v, eta_LTP, eta_LTD, w_cb, N_GC, N_MLI) {
    phi <- c(a, t_nd, beta_v, eta_LTP, eta_LTD, w_cb)
    
    res <- tryCatch(eval_arch_modulatory(phi, dat_all$Resp, dat_all$F, dat_all$participant_factor, dat_all$RT, N_GC, N_MLI), error=function(e) NULL)
    if (is.null(res)) return(-1000.0)
    
    dat_all$p1_temp <- res$prob1
    subj_res <- dat_all %>% group_by(participant_id) %>% summarize(
        nll = -sum(log(pmax(1e-15, pmin(1 - 1e-15, ifelse(Resp==1, p1_temp, 1-p1_temp))))),
        prauc = pr.curve(scores.class0 = p1_temp, weights.class0 = ifelse(Resp==1, 1, 0))$auc.integral
    )
    
    Total_NLL <- sum(subj_res$nll)
    Mean_PRAUC <- mean(subj_res$prauc, na.rm=TRUE)
    
    if (is.na(Total_NLL) || is.infinite(Total_NLL)) Total_NLL <- 1e6
    if (is.na(Mean_PRAUC)) Mean_PRAUC <- 0.5
    
    penalty <- max(0, (Total_NLL - baseline_nll) * 0.001)
    score <- Mean_PRAUC - penalty
    return(score)
}

# 1. Generate Random Initial Set (Latin Hypercube style)
cat("Sampling Initial 30 Random Points for Surrogate Training...\n")
set.seed(42)
N_init <- 30
history_df <- data.frame(
    a = runif(N_init, 0.1, 5.0),
    t_nd = runif(N_init, 0.01, 1.0),
    beta_v = runif(N_init, 0.0, 10.0),
    eta_LTP = runif(N_init, 0.0, 1.0),
    eta_LTD = runif(N_init, 0.0, 1.0),
    w_cb = runif(N_init, -5.0, 5.0),
    N_GC = sample(20:1000, N_init, replace=TRUE),
    N_MLI = sample(10:300, N_init, replace=TRUE),
    Score = NA
)

for (i in 1:N_init) {
    history_df$Score[i] <- evaluate_point(history_df$a[i], history_df$t_nd[i], history_df$beta_v[i], 
                                          history_df$eta_LTP[i], history_df$eta_LTD[i], history_df$w_cb[i], 
                                          history_df$N_GC[i], history_df$N_MLI[i])
}

# 2. Surrogate-Assisted Optimization Loop
N_iterations <- 10
for (iter in 1:N_iterations) {
    cat(sprintf("Surrogate Iteration %d/%d... Training Random Forest...\n", iter, N_iterations))
    
    # Train Surrogate Model
    rf_model <- randomForest(Score ~ a + t_nd + beta_v + eta_LTP + eta_LTD + w_cb + N_GC + N_MLI, data = history_df)
    
    # Generate Massive Random Grid to predict
    N_candidates <- 10000
    candidate_df <- data.frame(
        a = runif(N_candidates, 0.1, 5.0),
        t_nd = runif(N_candidates, 0.01, 1.0),
        beta_v = runif(N_candidates, 0.0, 10.0),
        eta_LTP = runif(N_candidates, 0.0, 1.0),
        eta_LTD = runif(N_candidates, 0.0, 1.0),
        w_cb = runif(N_candidates, -5.0, 5.0),
        N_GC = sample(20:1000, N_candidates, replace=TRUE),
        N_MLI = sample(10:300, N_candidates, replace=TRUE)
    )
    
    # Predict scores using the surrogate
    candidate_df$Pred_Score <- predict(rf_model, candidate_df)
    
    # Take top 3 most promising points
    top_candidates <- candidate_df %>% arrange(desc(Pred_Score)) %>% head(3)
    
    for (i in 1:nrow(top_candidates)) {
        true_score <- evaluate_point(top_candidates$a[i], top_candidates$t_nd[i], top_candidates$beta_v[i], 
                                     top_candidates$eta_LTP[i], top_candidates$eta_LTD[i], top_candidates$w_cb[i], 
                                     top_candidates$N_GC[i], top_candidates$N_MLI[i])
        
        new_row <- data.frame(
            a = top_candidates$a[i], t_nd = top_candidates$t_nd[i], beta_v = top_candidates$beta_v[i],
            eta_LTP = top_candidates$eta_LTP[i], eta_LTD = top_candidates$eta_LTD[i], w_cb = top_candidates$w_cb[i],
            N_GC = top_candidates$N_GC[i], N_MLI = top_candidates$N_MLI[i], Score = true_score
        )
        history_df <- rbind(history_df, new_row)
        cat(sprintf("   Evaluated Candidate: GC=%d, MLI=%d, True Score=%.4f (Surrogate predicted: %.4f)\n", 
                    top_candidates$N_GC[i], top_candidates$N_MLI[i], true_score, top_candidates$Pred_Score[i]))
    }
}

cat("\n=======================================================\n")
cat("          SURROGATE OPTIMIZATION RESULTS               \n")
cat("=======================================================\n")

best_idx <- which.max(history_df$Score)
best_params <- history_df[best_idx, ]

cat(sprintf("Best Score Found: %.4f\n", best_params$Score))
cat(sprintf("Optimal Network Structure -> N_GC: %d | N_MLI: %d\n", best_params$N_GC, best_params$N_MLI))
cat(sprintf("Optimal Parameters -> a: %.2f, beta_v: %.2f, w_cb: %.2f\n", best_params$a, best_params$beta_v, best_params$w_cb))
cat("=======================================================\n")

# Re-run best to get detailed NLL and PRAUC
phi_best <- c(best_params$a, best_params$t_nd, best_params$beta_v, best_params$eta_LTP, best_params$eta_LTD, best_params$w_cb)
res_best <- eval_arch_modulatory(phi_best, dat_all$Resp, dat_all$F, dat_all$participant_factor, dat_all$RT, best_params$N_GC, best_params$N_MLI)

dat_all$p1_best <- res_best$prob1
subj_best <- dat_all %>% group_by(participant_id) %>% summarize(
    nll = -sum(log(pmax(1e-15, pmin(1 - 1e-15, ifelse(Resp==1, p1_best, 1-p1_best))))),
    prauc = pr.curve(scores.class0 = p1_best, weights.class0 = ifelse(Resp==1, 1, 0))$auc.integral
)
cat(sprintf("Final Performance -> Total NLL: %.2f | Mean PR-AUC: %.4f\n", sum(subj_best$nll), mean(subj_best$prauc, na.rm=TRUE)))
cat("=======================================================\n")
