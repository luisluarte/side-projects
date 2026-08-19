library(cmaes)
library(Rcpp)
library(dplyr)
library(PRROC)

cat("Loading dataset for Grid Search...\n")
dataset_path <- "data/raw/behavioral_compilate.csv"
if (!file.exists(dataset_path)) {
  dataset_path <- "c:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv"
}

dat_all <- read.csv(dataset_path)
dat_all[['RT']] <- (as.numeric(dat_all[['ttr']]) - as.numeric(dat_all[['ttp']])) / 1000.0
dat_all <- dat_all[!is.na(dat_all[['RT']]) & dat_all[['RT']] >= 0.1 & dat_all[['RT']] <= 3.0 & dat_all[['Resp']] %in% c(1, 2), ]

# Subsample N=30 participants to speed up the grid search
set.seed(123)
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

// Parameterized Modulatory Cerebellum with dynamic network sizes
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

grids <- list(
  c(100, 20),
  c(400, 20),
  c(400, 80),
  c(400, 200),
  c(1000, 200)
)

results <- list()

for (i in seq_along(grids)) {
  n_gc <- grids[[i]][1]
  n_mli <- grids[[i]][2]
  
  cat(sprintf("Optimizing Modulatory Cerebellum with %d GC and %d MLI...\n", n_gc, n_mli))
  
  opt <- cma_es(c(1.0, 0.3, 2.0, 0.5, 0.5, 0.5), 
                function(p) { if(any(p<c(0.1,0.01,0,0,0,-5.0))||any(p>c(5,1,10,1,1,5.0))) 1e9 else eval_arch_modulatory(p, dat_all$Resp, dat_all$F, dat_all$participant_factor, dat_all$RT, n_gc, n_mli)$nll }, 
                lower=c(0.1, 0.01, 0, 0, 0, -5.0), upper=c(5, 1, 10, 1, 1, 5.0), control=list(maxit=50, trace=FALSE))
  
  res <- eval_arch_modulatory(opt$par, dat_all$Resp, dat_all$F, dat_all$participant_factor, dat_all$RT, n_gc, n_mli)
  
  col_name <- paste0("p1_gc", n_gc, "_mli", n_mli)
  dat_all[[col_name]] <- res$prob1
  
  results[[i]] <- list(
    Grid = sprintf("20:%d:%d", n_gc, n_mli),
    Total_NLL = res$nll
  )
}

# Evaluate all participants and compute PR-AUC
subj_res <- dat_all %>% group_by(participant_id) %>% summarize(
  prauc_100_20 = pr.curve(scores.class0 = p1_gc100_mli20, weights.class0 = ifelse(Resp==1, 1, 0))$auc.integral,
  prauc_400_20 = pr.curve(scores.class0 = p1_gc400_mli20, weights.class0 = ifelse(Resp==1, 1, 0))$auc.integral,
  prauc_400_80 = pr.curve(scores.class0 = p1_gc400_mli80, weights.class0 = ifelse(Resp==1, 1, 0))$auc.integral,
  prauc_400_200 = pr.curve(scores.class0 = p1_gc400_mli200, weights.class0 = ifelse(Resp==1, 1, 0))$auc.integral,
  prauc_1000_200 = pr.curve(scores.class0 = p1_gc1000_mli200, weights.class0 = ifelse(Resp==1, 1, 0))$auc.integral
)

cat("\n=======================================================\n")
cat("          MODULATORY CEREBELLUM GRID SEARCH            \n")
cat("=======================================================\n")

res_df <- data.frame(
    Network = c("20:100:20", "20:400:20", "20:400:80", "20:400:200", "20:1000:200"),
    Total_NLL = sapply(results, function(x) x$Total_NLL),
    Mean_PRAUC = c(mean(subj_res$prauc_100_20, na.rm=TRUE), 
                   mean(subj_res$prauc_400_20, na.rm=TRUE), 
                   mean(subj_res$prauc_400_80, na.rm=TRUE), 
                   mean(subj_res$prauc_400_200, na.rm=TRUE), 
                   mean(subj_res$prauc_1000_200, na.rm=TRUE))
)

print(res_df, row.names=FALSE)
cat("=======================================================\n")
