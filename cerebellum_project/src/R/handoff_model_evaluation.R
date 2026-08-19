library(cmaes)
library(Rcpp)
library(dplyr)
library(ggplot2)

cat("Loading dataset...\n")
dataset_path <- "data/raw/behavioral_compilate.csv"
if (!file.exists(dataset_path)) {
  dataset_path <- "c:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv"
}

dat_all <- read.csv(dataset_path)
dat_all[['RT']] <- (as.numeric(dat_all[['ttr']]) - as.numeric(dat_all[['ttp']])) / 1000.0
dat_all <- dat_all[!is.na(dat_all[['RT']]) & dat_all[['RT']] >= 0.1 & dat_all[['RT']] <= 3.0 & dat_all[['Resp']] %in% c(1, 2), ]

set.seed(42)
participants <- unique(dat_all[['participant_id']])
num_participants <- length(participants)
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

class SimpleRNG {
    uint32_t state;
public:
    SimpleRNG(uint32_t seed) : state(seed) {}
    uint32_t next() { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state; }
    double runif() { return (next() % 1000000) / 1000000.0; }
    double rnorm() {
        double u1 = std::max(1e-6, runif());
        double u2 = runif();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
    }
};

// Pure CFMR (Cortex)
// [[Rcpp::export]]
List evaluate_cfmr_full(const NumericVector& phi, const IntegerVector& resp_R, const IntegerVector& out_R, const IntegerVector& subj_idx_R, const NumericVector& rt_R, const IntegerVector& trial_idx_R) {
  double a = phi[0]; double t_nd = phi[1]; double beta_v = phi[2]; double eta_LTP = phi[3]; double eta_LTD = phi[4];
  double nll = 0.0; std::vector<double> out_prob1; 
  double Q1 = 0.0; double Q2 = 0.0;
  for (int t=0; t<resp_R.size(); ++t) {
    if (t > 0 && subj_idx_R[t] != subj_idx_R[t-1]) { Q1 = 0.0; Q2 = 0.0; }
    int ch = resp_R[t]; int out = out_R[t];
    double v_t = beta_v * (Q1 - Q2);
    double safe_v_t = std::abs(v_t) < 1e-4 ? (v_t >= 0 ? 1e-4 : -1e-4) : v_t;
    double dens = wiener_pdf(rt_R[t], ch, safe_v_t, a, t_nd);
    nll -= std::log(dens);
    out_prob1.push_back(1.0 / (1.0 + std::exp(-a * safe_v_t)));
    if (ch == 1) { Q1 += (out == 1) ? eta_LTP * (1.0 - Q1) : eta_LTD * (-1.0 - Q1); } 
    else         { Q2 += (out == 1) ? eta_LTP * (1.0 - Q2) : eta_LTD * (-1.0 - Q2); }
  }
  return List::create(Named("nll") = nll, Named("prob1") = out_prob1);
}

// Pure Expansion-Compression Cerebellum (20:400:80 MF:GC:MLI)
// [[Rcpp::export]]
List evaluate_expcomp_full(const NumericVector& phi, const IntegerVector& resp_R, const IntegerVector& out_R, const IntegerVector& subj_idx_R, const NumericVector& rt_R, const IntegerVector& trial_idx_R) {
  double a = phi[0]; double t_nd = phi[1]; double beta_v = phi[2]; double eta_LTP = phi[3]; double eta_LTD = phi[4];
  double nll = 0.0; std::vector<double> out_prob1; 
  
  int N_MF = 20; int N_GC = 400; int N_MLI = 80;
  std::vector<double> W_PF1(N_GC, 0.0); std::vector<double> W_PF2(N_GC, 0.0);
  std::vector<double> W_MLI1(N_MLI, 0.0); std::vector<double> W_MLI2(N_MLI, 0.0);
  
  std::vector<std::vector<double>> W_MF_GC(N_GC, std::vector<double>(N_MF, 0.0));
  std::vector<std::vector<double>> W_GC_MLI(N_MLI, std::vector<double>(N_GC, 0.0));
  std::vector<double> mf(N_MF, 0.0);
  
  for (int t=0; t<resp_R.size(); ++t) {
    int s_idx = subj_idx_R[t];
    if (t == 0 || subj_idx_R[t] != subj_idx_R[t-1]) {
        std::fill(W_PF1.begin(), W_PF1.end(), 0.0); std::fill(W_PF2.begin(), W_PF2.end(), 0.0);
        std::fill(W_MLI1.begin(), W_MLI1.end(), 0.0); std::fill(W_MLI2.begin(), W_MLI2.end(), 0.0);
        std::fill(mf.begin(), mf.end(), 0.0);
        
        SimpleRNG rng(s_idx + 42);
        for (int i=0; i<N_GC; ++i) { for (int j=0; j<N_MF; ++j) W_MF_GC[i][j] = rng.rnorm() / std::sqrt((double)N_MF); }
        for (int i=0; i<N_MLI; ++i) { for (int j=0; j<N_GC; ++j) W_GC_MLI[i][j] = rng.rnorm() / std::sqrt((double)N_GC); }
    }
    int ch = resp_R[t]; int out = out_R[t];
    
    std::vector<double> gc(N_GC, 0.0);
    for (int i=0; i<N_GC; ++i) {
        double act = 0.0;
        for (int j=0; j<N_MF; ++j) act += W_MF_GC[i][j] * mf[j];
        gc[i] = std::tanh(act);
    }
    
    std::vector<double> mli(N_MLI, 0.0);
    for (int i=0; i<N_MLI; ++i) {
        double act = 0.0;
        for (int j=0; j<N_GC; ++j) act += W_GC_MLI[i][j] * gc[j];
        mli[i] = std::tanh(act);
    }
    
    double Q1 = 0.0; double Q2 = 0.0;
    for (int i=0; i<N_GC; ++i) { Q1 += W_PF1[i]*gc[i]; Q2 += W_PF2[i]*gc[i]; }
    for (int i=0; i<N_MLI; ++i) { Q1 -= W_MLI1[i]*mli[i]; Q2 -= W_MLI2[i]*mli[i]; } // Inhibitory
    
    double v_t = beta_v * (Q1 - Q2);
    double safe_v_t = std::abs(v_t) < 1e-4 ? (v_t >= 0 ? 1e-4 : -1e-4) : v_t;
    double dens = wiener_pdf(rt_R[t], ch, safe_v_t, a, t_nd);
    nll -= std::log(dens);
    out_prob1.push_back(1.0 / (1.0 + std::exp(-a * safe_v_t)));
    
    double err1 = (out == 1) ? (1.0 - Q1) : (-1.0 - Q1);
    double err2 = (out == 1) ? (1.0 - Q2) : (-1.0 - Q2);
    
    if (ch == 1) {
        double lr = (out == 1) ? eta_LTP : eta_LTD;
        for (int i=0; i<N_GC; ++i) W_PF1[i] += lr * err1 * gc[i];
        for (int i=0; i<N_MLI; ++i) W_MLI1[i] -= lr * err1 * mli[i]; // inhibitory update
    } else {
        double lr = (out == 1) ? eta_LTP : eta_LTD;
        for (int i=0; i<N_GC; ++i) W_PF2[i] += lr * err2 * gc[i];
        for (int i=0; i<N_MLI; ++i) W_MLI2[i] -= lr * err2 * mli[i]; // inhibitory update
    }
    
    for(int k=8; k>=0; --k) { mf[k+1] = mf[k]; mf[10+k+1] = mf[10+k]; }
    mf[0] = (ch == 1) ? 1.0 : -1.0;
    mf[10] = (out == 1) ? 1.0 : -1.0;
  }
  return List::create(Named("nll") = nll, Named("prob1") = out_prob1);
}

// Hand-Off: Exponential Decay (20:400:80 to CFMR)
// [[Rcpp::export]]
List evaluate_handoff_exp_full(const NumericVector& phi, const IntegerVector& resp_R, const IntegerVector& out_R, const IntegerVector& subj_idx_R, const NumericVector& rt_R, const IntegerVector& trial_idx_R) {
  double a = phi[0]; double t_nd = phi[1]; double beta_v = phi[2]; double eta_LTP = phi[3]; double eta_LTD = phi[4]; double tau = phi[5];
  double nll = 0.0; std::vector<double> out_prob1, out_lambda; 
  
  double Q1_CTX = 0.0; double Q2_CTX = 0.0;
  
  int N_MF = 20; int N_GC = 400; int N_MLI = 80;
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
    
    int ch = resp_R[t]; int out = out_R[t]; int tr = trial_idx_R[t];
    
    // Cerebellar Module
    std::vector<double> gc(N_GC, 0.0);
    for (int i=0; i<N_GC; ++i) {
        double act = 0.0;
        for (int j=0; j<N_MF; ++j) act += W_MF_GC[i][j] * mf[j];
        gc[i] = std::tanh(act);
    }
    std::vector<double> mli(N_MLI, 0.0);
    for (int i=0; i<N_MLI; ++i) {
        double act = 0.0;
        for (int j=0; j<N_GC; ++j) act += W_GC_MLI[i][j] * gc[j];
        mli[i] = std::tanh(act);
    }
    
    double Q1_CB = 0.0; double Q2_CB = 0.0;
    for (int i=0; i<N_GC; ++i) { Q1_CB += W_PF1[i]*gc[i]; Q2_CB += W_PF2[i]*gc[i]; }
    for (int i=0; i<N_MLI; ++i) { Q1_CB -= W_MLI1[i]*mli[i]; Q2_CB -= W_MLI2[i]*mli[i]; }
    
    // Transition
    double lambda_t = std::exp(-tau * (double)tr);
    double v_t = beta_v * (lambda_t * (Q1_CB - Q2_CB) + (1.0 - lambda_t) * (Q1_CTX - Q2_CTX));
    
    double safe_v_t = std::abs(v_t) < 1e-4 ? (v_t >= 0 ? 1e-4 : -1e-4) : v_t;
    double dens = wiener_pdf(rt_R[t], ch, safe_v_t, a, t_nd);
    nll -= std::log(dens);
    out_prob1.push_back(1.0 / (1.0 + std::exp(-a * safe_v_t)));
    out_lambda.push_back(lambda_t);
    
    // Updates
    if (ch == 1) {
        Q1_CTX += (out == 1) ? eta_LTP * (1.0 - Q1_CTX) : eta_LTD * (-1.0 - Q1_CTX);
        
        double err1 = (out == 1) ? (1.0 - Q1_CB) : (-1.0 - Q1_CB);
        double lr = (out == 1) ? eta_LTP : eta_LTD;
        for (int i=0; i<N_GC; ++i) W_PF1[i] += lr * err1 * gc[i];
        for (int i=0; i<N_MLI; ++i) W_MLI1[i] -= lr * err1 * mli[i];
    } else {
        Q2_CTX += (out == 1) ? eta_LTP * (1.0 - Q2_CTX) : eta_LTD * (-1.0 - Q2_CTX);
        
        double err2 = (out == 1) ? (1.0 - Q2_CB) : (-1.0 - Q2_CB);
        double lr = (out == 1) ? eta_LTP : eta_LTD;
        for (int i=0; i<N_GC; ++i) W_PF2[i] += lr * err2 * gc[i];
        for (int i=0; i<N_MLI; ++i) W_MLI2[i] -= lr * err2 * mli[i];
    }
    
    for(int k=8; k>=0; --k) { mf[k+1] = mf[k]; mf[10+k+1] = mf[10+k]; }
    mf[0] = (ch == 1) ? 1.0 : -1.0;
    mf[10] = (out == 1) ? 1.0 : -1.0;
  }
  return List::create(Named("nll") = nll, Named("prob1") = out_prob1, Named("lambda") = out_lambda);
}

// Hand-Off: Uncertainty-Driven (20:400:80 to CFMR)
// [[Rcpp::export]]
List evaluate_handoff_unc_full(const NumericVector& phi, const IntegerVector& resp_R, const IntegerVector& out_R, const IntegerVector& subj_idx_R, const NumericVector& rt_R, const IntegerVector& trial_idx_R) {
  double a = phi[0]; double t_nd = phi[1]; double beta_v = phi[2]; double eta_LTP = phi[3]; double eta_LTD = phi[4]; double alpha_u = phi[5];
  double nll = 0.0; std::vector<double> out_prob1, out_lambda; 
  
  double Q1_CTX = 0.0; double Q2_CTX = 0.0; double U_t = 2.0;
  
  int N_MF = 20; int N_GC = 400; int N_MLI = 80;
  std::vector<double> W_PF1(N_GC, 0.0); std::vector<double> W_PF2(N_GC, 0.0);
  std::vector<double> W_MLI1(N_MLI, 0.0); std::vector<double> W_MLI2(N_MLI, 0.0);
  
  std::vector<std::vector<double>> W_MF_GC(N_GC, std::vector<double>(N_MF, 0.0));
  std::vector<std::vector<double>> W_GC_MLI(N_MLI, std::vector<double>(N_GC, 0.0));
  std::vector<double> mf(N_MF, 0.0);
  
  for (int t=0; t<resp_R.size(); ++t) {
    int s_idx = subj_idx_R[t];
    if (t == 0 || subj_idx_R[t] != subj_idx_R[t-1]) {
        Q1_CTX = 0.0; Q2_CTX = 0.0; U_t = 2.0;
        std::fill(W_PF1.begin(), W_PF1.end(), 0.0); std::fill(W_PF2.begin(), W_PF2.end(), 0.0);
        std::fill(W_MLI1.begin(), W_MLI1.end(), 0.0); std::fill(W_MLI2.begin(), W_MLI2.end(), 0.0);
        std::fill(mf.begin(), mf.end(), 0.0);
        
        SimpleRNG rng(s_idx + 42);
        for (int i=0; i<N_GC; ++i) { for (int j=0; j<N_MF; ++j) W_MF_GC[i][j] = rng.rnorm() / std::sqrt((double)N_MF); }
        for (int i=0; i<N_MLI; ++i) { for (int j=0; j<N_GC; ++j) W_GC_MLI[i][j] = rng.rnorm() / std::sqrt((double)N_GC); }
    }
    
    int ch = resp_R[t]; int out = out_R[t];
    
    // Cerebellar Module
    std::vector<double> gc(N_GC, 0.0);
    for (int i=0; i<N_GC; ++i) {
        double act = 0.0;
        for (int j=0; j<N_MF; ++j) act += W_MF_GC[i][j] * mf[j];
        gc[i] = std::tanh(act);
    }
    std::vector<double> mli(N_MLI, 0.0);
    for (int i=0; i<N_MLI; ++i) {
        double act = 0.0;
        for (int j=0; j<N_GC; ++j) act += W_GC_MLI[i][j] * gc[j];
        mli[i] = std::tanh(act);
    }
    
    double Q1_CB = 0.0; double Q2_CB = 0.0;
    for (int i=0; i<N_GC; ++i) { Q1_CB += W_PF1[i]*gc[i]; Q2_CB += W_PF2[i]*gc[i]; }
    for (int i=0; i<N_MLI; ++i) { Q1_CB -= W_MLI1[i]*mli[i]; Q2_CB -= W_MLI2[i]*mli[i]; }
    
    // Transition
    double lambda_t = std::min(1.0, std::max(0.0, U_t / 2.0));
    double v_t = beta_v * (lambda_t * (Q1_CB - Q2_CB) + (1.0 - lambda_t) * (Q1_CTX - Q2_CTX));
    
    double safe_v_t = std::abs(v_t) < 1e-4 ? (v_t >= 0 ? 1e-4 : -1e-4) : v_t;
    double dens = wiener_pdf(rt_R[t], ch, safe_v_t, a, t_nd);
    nll -= std::log(dens);
    out_prob1.push_back(1.0 / (1.0 + std::exp(-a * safe_v_t)));
    out_lambda.push_back(lambda_t);
    
    // Updates
    double rpe = (out == 1 ? 1.0 : -1.0) - (ch == 1 ? Q1_CTX : Q2_CTX);
    U_t = (1.0 - alpha_u) * U_t + alpha_u * std::abs(rpe);
    
    if (ch == 1) {
        Q1_CTX += (out == 1) ? eta_LTP * (1.0 - Q1_CTX) : eta_LTD * (-1.0 - Q1_CTX);
        
        double err1 = (out == 1) ? (1.0 - Q1_CB) : (-1.0 - Q1_CB);
        double lr = (out == 1) ? eta_LTP : eta_LTD;
        for (int i=0; i<N_GC; ++i) W_PF1[i] += lr * err1 * gc[i];
        for (int i=0; i<N_MLI; ++i) W_MLI1[i] -= lr * err1 * mli[i];
    } else {
        Q2_CTX += (out == 1) ? eta_LTP * (1.0 - Q2_CTX) : eta_LTD * (-1.0 - Q2_CTX);
        
        double err2 = (out == 1) ? (1.0 - Q2_CB) : (-1.0 - Q2_CB);
        double lr = (out == 1) ? eta_LTP : eta_LTD;
        for (int i=0; i<N_GC; ++i) W_PF2[i] += lr * err2 * gc[i];
        for (int i=0; i<N_MLI; ++i) W_MLI2[i] -= lr * err2 * mli[i];
    }
    
    for(int k=8; k>=0; --k) { mf[k+1] = mf[k]; mf[10+k+1] = mf[10+k]; }
    mf[0] = (ch == 1) ? 1.0 : -1.0;
    mf[10] = (out == 1) ? 1.0 : -1.0;
  }
  return List::create(Named("nll") = nll, Named("prob1") = out_prob1, Named("lambda") = out_lambda);
}
'
sourceCpp(code = cpp_code)

N <- nrow(dat_all)
compute_bic <- function(nll, k) { k * log(N) + 2 * nll }
compute_aic <- function(nll, k) { 2 * k + 2 * nll }

cat("Optimizing Pure CFMR...\n")
obj_cfmr <- function(phi) {
    if (any(phi < c(0.1, 0.01, 0.0, 0.0, 0.0)) || any(phi > c(5.0, 1.0, 10.0, 1.0, 1.0))) return(1e9)
    res <- evaluate_cfmr_full(as.numeric(phi), as.integer(dat_all$Resp), as.integer(dat_all$F), as.integer(dat_all$participant_factor), as.numeric(dat_all$RT), as.integer(dat_all$trial_idx))
    return(res$nll)
}
opt_cfmr <- cma_es(c(1.0, 0.3, 2.0, 0.5, 0.5), obj_cfmr, lower=c(0.1, 0.01, 0.0, 0.0, 0.0), upper=c(5.0, 1.0, 10.0, 1.0, 1.0), control=list(maxit=50, trace=FALSE))

cat("Optimizing Pure Exp-Comp Cerebellum (20:400:80)...\n")
obj_ngrc <- function(phi) {
    if (any(phi < c(0.1, 0.01, 0.0, 0.0, 0.0)) || any(phi > c(5.0, 1.0, 10.0, 1.0, 1.0))) return(1e9)
    res <- evaluate_expcomp_full(as.numeric(phi), as.integer(dat_all$Resp), as.integer(dat_all$F), as.integer(dat_all$participant_factor), as.numeric(dat_all$RT), as.integer(dat_all$trial_idx))
    return(res$nll)
}
opt_ngrc <- cma_es(c(1.0, 0.3, 2.0, 0.5, 0.5), obj_ngrc, lower=c(0.1, 0.01, 0.0, 0.0, 0.0), upper=c(5.0, 1.0, 10.0, 1.0, 1.0), control=list(maxit=50, trace=FALSE))

cat("Optimizing Exponential Hand-Off...\n")
obj_exp <- function(phi) {
    if (any(phi < c(0.1, 0.01, 0.0, 0.0, 0.0, 0.0)) || any(phi > c(5.0, 1.0, 10.0, 1.0, 1.0, 0.5))) return(1e9)
    res <- evaluate_handoff_exp_full(as.numeric(phi), as.integer(dat_all$Resp), as.integer(dat_all$F), as.integer(dat_all$participant_factor), as.numeric(dat_all$RT), as.integer(dat_all$trial_idx))
    return(res$nll)
}
opt_exp <- cma_es(c(1.0, 0.3, 2.0, 0.5, 0.5, 0.05), obj_exp, lower=c(0.1, 0.01, 0.0, 0.0, 0.0, 0.0), upper=c(5.0, 1.0, 10.0, 1.0, 1.0, 0.5), control=list(maxit=50, trace=FALSE))

cat("Optimizing Uncertainty Hand-Off...\n")
obj_unc <- function(phi) {
    if (any(phi < c(0.1, 0.01, 0.0, 0.0, 0.0, 0.0)) || any(phi > c(5.0, 1.0, 10.0, 1.0, 1.0, 1.0))) return(1e9)
    res <- evaluate_handoff_unc_full(as.numeric(phi), as.integer(dat_all$Resp), as.integer(dat_all$F), as.integer(dat_all$participant_factor), as.numeric(dat_all$RT), as.integer(dat_all$trial_idx))
    return(res$nll)
}
opt_unc <- cma_es(c(1.0, 0.3, 2.0, 0.5, 0.5, 0.1), obj_unc, lower=c(0.1, 0.01, 0.0, 0.0, 0.0, 0.0), upper=c(5.0, 1.0, 10.0, 1.0, 1.0, 1.0), control=list(maxit=50, trace=FALSE))


res_cfmr <- evaluate_cfmr_full(opt_cfmr$par, as.integer(dat_all$Resp), as.integer(dat_all$F), as.integer(dat_all$participant_factor), as.numeric(dat_all$RT), as.integer(dat_all$trial_idx))
res_ngrc <- evaluate_expcomp_full(opt_ngrc$par, as.integer(dat_all$Resp), as.integer(dat_all$F), as.integer(dat_all$participant_factor), as.numeric(dat_all$RT), as.integer(dat_all$trial_idx))
res_exp <- evaluate_handoff_exp_full(opt_exp$par, as.integer(dat_all$Resp), as.integer(dat_all$F), as.integer(dat_all$participant_factor), as.numeric(dat_all$RT), as.integer(dat_all$trial_idx))
res_unc <- evaluate_handoff_unc_full(opt_unc$par, as.integer(dat_all$Resp), as.integer(dat_all$F), as.integer(dat_all$participant_factor), as.numeric(dat_all$RT), as.integer(dat_all$trial_idx))

brier <- function(prob, true_ch) { mean((ifelse(true_ch==1, 1, 0) - prob)^2) }
b_cfmr <- brier(res_cfmr$prob1, dat_all$Resp)
b_ngrc <- brier(res_ngrc$prob1, dat_all$Resp)
b_exp <- brier(res_exp$prob1, dat_all$Resp)
b_unc <- brier(res_unc$prob1, dat_all$Resp)


df_out <- data.frame(
    Model = c("Pure CFMR (Cortex)", "Pure Exp-Comp (20:400:80)", "Exponential Hand-Off", "Uncertainty Hand-Off"),
    k = c(5, 5, 6, 6),
    Total_NLL = c(res_cfmr$nll, res_ngrc$nll, res_exp$nll, res_unc$nll),
    AIC = c(compute_aic(res_cfmr$nll, 5), compute_aic(res_ngrc$nll, 5), compute_aic(res_exp$nll, 6), compute_aic(res_unc$nll, 6)),
    BIC = c(compute_bic(res_cfmr$nll, 5), compute_bic(res_ngrc$nll, 5), compute_bic(res_exp$nll, 6), compute_bic(res_unc$nll, 6)),
    Brier_Score = c(b_cfmr, b_ngrc, b_exp, b_unc)
)

cat("\n========================================================================\n")
cat("          MINIMUM DESCRIPTION LENGTH / MODEL SELECTION COMPARISON       \n")
cat("========================================================================\n")
print(df_out, row.names = FALSE)
cat("========================================================================\n")

cat(sprintf("\nOptimal Tau (Exp Decay): %.5f\n", opt_exp$par[6]))
cat(sprintf("Optimal Alpha_U (Uncertainty LR): %.5f\n", opt_unc$par[6]))

# Plot the lambda_t curves
dat_all$lambda_exp <- res_exp$lambda
dat_all$lambda_unc <- res_unc$lambda

agg_lambda <- dat_all %>%
  group_by(trial_idx) %>%
  summarize(
      Mean_Exp_Lambda = mean(lambda_exp),
      Mean_Unc_Lambda = mean(lambda_unc)
  )

p <- ggplot(agg_lambda, aes(x = trial_idx)) +
  geom_line(aes(y = Mean_Exp_Lambda, color = "Exponential Fade-out"), linewidth=1.2) +
  geom_line(aes(y = Mean_Unc_Lambda, color = "Uncertainty-Driven Fade-out"), linewidth=1.2) +
  theme_minimal() +
  labs(
      title = "Exp-Comp Cerebellar Weight (Lambda) over Time",
      subtitle = "If > 0, Exp-Comp 20:400:80 dominates. If approaching 0, Cortex (CFMR) takes over.",
      x = "Trial Number", y = "Lambda_t (Cerebellar Weight)", color = "Hand-Off Model"
  )

plot_path <- "C:/Users/DCCS5/.gemini/antigravity/brain/1d8f9958-fd49-4502-b57b-97a7887eb7ad/handoff_lambda_curves.png"
ggsave(plot_path, p, width = 8, height = 5)
cat(sprintf("Saved plot to %s\n", plot_path))
