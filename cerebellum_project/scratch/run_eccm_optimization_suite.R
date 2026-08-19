library(cmaes)
library(Rcpp)

dataset_path <- "data/raw/behavioral_compilate.csv"
if (!file.exists(dataset_path)) {
  dataset_path <- "c:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv"
}

dat_all <- read.csv(dataset_path)
dat_all[['RT']] <- (as.numeric(dat_all[['ttr']]) - as.numeric(dat_all[['ttp']])) / 1000.0
dat_all <- dat_all[!is.na(dat_all[['RT']]) & dat_all[['RT']] >= 0.1 & dat_all[['RT']] <= 3.0 & dat_all[['Resp']] %in% c(1, 2), ]

set.seed(42)
participants <- unique(dat_all[['participant_id']])
sample_participants <- sample(participants, 10)
dat_all <- dat_all[dat_all[['participant_id']] %in% sample_participants, ]

dat_all$is_test <- 0
for (p in sample_participants) {
  p_idx <- which(dat_all$participant_id == p)
  n_trials <- length(p_idx)
  n_train <- floor(0.70 * n_trials)
  dat_all$is_test[p_idx[(n_train + 1):n_trials]] <- 1
}

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

// [[Rcpp::export]]
List evaluate_eccm_generalized_cpp(
    const NumericVector& phi_13d, const IntegerVector& resp_R, const IntegerVector& out_R,
    const NumericVector& m1_R, const NumericVector& m2_R, const NumericVector& rt_R,
    const NumericVector& ttp_R, const NumericVector& ttf_R, const IntegerVector& subj_idx_R,
    const IntegerVector& is_test_R, int N_GC, int N_MLI, IntegerVector active_features, 
    bool return_test_metrics = false
) {
  int N_t = resp_R.size();
  double beta_v=phi_13d[0], a_0=phi_13d[1], t_nd=phi_13d[2], kappa_a=phi_13d[3], mu_beta=phi_13d[4], sigma_beta=phi_13d[5];
  double lambda_d=phi_13d[6], mu_tau=phi_13d[7], sigma_tau=phi_13d[8], rho_base=phi_13d[9], eta=phi_13d[10], lambda=phi_13d[11], theta_th=phi_13d[12];
  
  int N_MF = 20;
  SimpleRNG rng(42);
  
  std::vector<int> mf_c(N_MF), mf_d(N_MF); std::vector<double> mf_beta(N_MF);
  for(int j=0; j<N_MF; ++j) {
      mf_c[j] = active_features[rng.next() % active_features.size()]; 
      mf_beta[j] = std::exp(mu_beta + sigma_beta*rng.rnorm());
      mf_d[j] = std::max(0, std::min(10, (int)std::round(lambda_d + std::sqrt(lambda_d)*rng.rnorm())));
  }

  std::vector<std::vector<int>> gc_mossy_map(N_GC, std::vector<int>(4));
  std::vector<std::vector<double>> gc_mossy_weights(N_GC, std::vector<double>(4, 0.0));
  for (int i=0; i<N_GC; ++i) {
    for (int k=0; k<4; ++k) {
      gc_mossy_map[i][k] = rng.next()%N_MF;
      gc_mossy_weights[i][k] = (rng.rnorm()>0) ? 1.0 : -1.0; 
    }
  }
  std::vector<double> tau_vec(N_GC, 1.0);
  for (int i=0; i<N_GC; ++i) tau_vec[i] = std::exp(mu_tau + sigma_tau*rng.rnorm());

  double theta_max = 2.0 / (double)N_GC;
  std::vector<std::vector<double>> W_GC_MLI(N_MLI, std::vector<double>(N_GC, 0.0));
  for (int k=0; k<N_MLI; ++k) for (int i=0; i<N_GC; ++i) W_GC_MLI[k][i] = rng.runif()*theta_max;

  std::vector<double> z_GC_curr(N_GC, 0.0), z_GC_prev(N_GC, 0.0), W_PF1(N_MLI, 0.0), W_PF2(N_MLI, 0.0);
  std::vector<std::vector<double>> state_hist(15, std::vector<double>(6, 0.0));
  
  double train_nll = 0.0, test_nll = 0.0;
  int test_count = 0;
  
  for (int t=0; t<N_t; ++t) {
    if (t>0 && subj_idx_R[t]!=subj_idx_R[t-1]) {
      std::fill(z_GC_prev.begin(), z_GC_prev.end(), 0.0); std::fill(z_GC_curr.begin(), z_GC_curr.end(), 0.0);
      std::fill(W_PF1.begin(), W_PF1.end(), 0.0); std::fill(W_PF2.begin(), W_PF2.end(), 0.0);
      for (int d=0; d<15; ++d) std::fill(state_hist[d].begin(), state_hist[d].end(), 0.0);
    }
    int ch = resp_R[t], out = out_R[t];
    bool is_test = (is_test_R[t] == 1);
    int prev_ch = (t>0 && subj_idx_R[t]==subj_idx_R[t-1]) ? resp_R[t-1] : 1;
    int prev_out = (t>0 && subj_idx_R[t]==subj_idx_R[t-1]) ? out_R[t-1] : 1;
    double prev_rt = (t>0 && subj_idx_R[t]==subj_idx_R[t-1]) ? rt_R[t-1] : 0.75;
    double delta_t_val = (t==0 || (t>0 && subj_idx_R[t]!=subj_idx_R[t-1])) ? 1.5 : std::max(0.1, (double)(ttp_R[t]-ttp_R[t-1]));
    double prev_iti = (t>0 && subj_idx_R[t]==subj_idx_R[t-1]) ? (ttp_R[t]-ttf_R[t-1]) : 7.0;

    double m_curr = (prev_ch==1) ? m1_R[t] : m2_R[t];
    double m_alt  = (prev_ch==1) ? m2_R[t] : m1_R[t];
    for (int d=14; d>0; --d) state_hist[d] = state_hist[d-1];
    state_hist[0][0] = (prev_ch==1) ? 1.0 : -1.0; state_hist[0][1] = (prev_out==1) ? 1.0 : -1.0;
    state_hist[0][2] = (m_curr - 5.5)/4.5; state_hist[0][3] = (m_alt - m_curr)/4.0;
    state_hist[0][4] = std::max(-2.0, std::min(2.0, (prev_rt - 0.75)/0.50)); state_hist[0][5] = std::max(-2.0, std::min(2.0, (prev_iti - 7.0)/3.0));

    std::vector<double> u_MF(N_MF, 0.0);
    for(int j=0; j<N_MF; ++j) {
        if (mf_d[j]==0) u_MF[j] = 1.0/(1.0+std::exp(-mf_beta[j]*state_hist[0][mf_c[j]])); 
        else { int d_idx=std::min(mf_d[j], 14); u_MF[j] = 1.0/(1.0+std::exp(-mf_beta[j]*(state_hist[0][mf_c[j]]-state_hist[d_idx][mf_c[j]]))); }
    }

    for (int i=0; i<N_GC; ++i) {
        double in_sum = 0.0;
        for (int k=0; k<4; ++k) in_sum += gc_mossy_weights[i][k]*u_MF[gc_mossy_map[i][k]];
        double gamma_decay = rho_base + (1.0-rho_base)*std::exp(-delta_t_val/tau_vec[i]);
        z_GC_curr[i] = std::max(0.0, in_sum + gamma_decay*z_GC_prev[i]);
    }

    std::vector<double> h_MLI(N_MLI, 0.0);
    double l1_mli_sum = 1e-12;
    for (int k=0; k<N_MLI; ++k) {
        double pool_sum=0.0; for (int i=0; i<N_GC; ++i) pool_sum += W_GC_MLI[k][i]*z_GC_curr[i];
        h_MLI[k] = std::max(0.0, pool_sum - theta_th); l1_mli_sum += h_MLI[k];
    }
    
    double S_MLI = 0.0;
    for (int k=0; k<N_MLI; ++k) { double pk = h_MLI[k]/l1_mli_sum; if (pk>1e-12) S_MLI -= pk*std::log(pk); }
    double norm_S = S_MLI / std::log((double)N_MLI);

    double y_PC1=0.0, y_PC2=0.0;
    for (int k=0; k<N_MLI; ++k) { y_PC1 += W_PF1[k]*h_MLI[k]; y_PC2 += W_PF2[k]*h_MLI[k]; }

    double v_t_ddm = beta_v*(y_PC1 - y_PC2);
    double a_t = std::max(0.30, a_0 + kappa_a*norm_S);
    
    double dens = wiener_pdf(rt_R[t], ch, v_t_ddm, a_t, t_nd);
    if (is_test) { test_nll -= std::log(dens); test_count++; } else { train_nll -= std::log(dens); }

    double target = ((double)out - 0.5)*2.0;
    double delta_IO = target - ((ch==1) ? y_PC1 : y_PC2);

    double c_eta = is_test ? 0.0 : eta, c_lam = is_test ? 0.0 : lambda;
    for (int k=0; k<N_MLI; ++k) {
        if (ch==1) { W_PF1[k] += c_eta*delta_IO*h_MLI[k] - c_lam*W_PF1[k]; W_PF2[k] += -c_lam*W_PF2[k]; }
        else       { W_PF2[k] += c_eta*delta_IO*h_MLI[k] - c_lam*W_PF2[k]; W_PF1[k] += -c_lam*W_PF1[k]; }
    }
    z_GC_prev = z_GC_curr;
  }
  if (return_test_metrics) return List::create(Named("Train_NLL")=train_nll, Named("Test_NLL")=test_nll, Named("Test_Count")=test_count);
  return List::create(Named("Obj")=train_nll);
}
'
sourceCpp(code = cpp_code)

lower_bounds <- c(b_v = 0.0, a_0 = 0.30, t_nd = 0.10, kappa_a = 0.0, mu_beta = -2.0, sigma_beta = 0.01, lambda_d = 0.0, mu_tau = -2.0, sigma_tau = 0.01, rho_base = 0.0, eta = 0.0, lambda = 0.0, theta_th = 0.0)
upper_bounds <- c(b_v = 3.0, a_0 = 2.50, t_nd = 0.90, kappa_a = 2.0, mu_beta =  2.0, sigma_beta = 2.00, lambda_d = 5.0, mu_tau =  2.0, sigma_tau = 2.00, rho_base = 0.95, eta = 1.0, lambda = 0.5, theta_th = 0.5)
initial_phi <- lower_bounds + (upper_bounds - lower_bounds) / 2

run_optimization <- function(n_gc, n_mli, act_feat, max_iter=25) {
    obj_fun <- function(phi) {
        if (any(phi < lower_bounds) || any(phi > upper_bounds)) return(1e9)
        res <- evaluate_eccm_generalized_cpp(
            phi, as.integer(dat_all$Resp), as.integer(dat_all$F), dat_all$Bd1, dat_all$Bd2, 
            dat_all$RT, as.numeric(dat_all$ttp), as.numeric(dat_all$ttF), 
            as.integer(as.factor(dat_all$participant_id)), dat_all$is_test, 
            n_gc, n_mli, act_feat, FALSE
        )
        return(res$Obj)
    }
    cma_res <- cma_es(initial_phi, obj_fun, lower = lower_bounds, upper = upper_bounds, control = list(maxit = max_iter, trace = FALSE, sigma = 0.2))
    phi_opt <- cma_res$par
    final_res <- evaluate_eccm_generalized_cpp(
        phi_opt, as.integer(dat_all$Resp), as.integer(dat_all$F), dat_all$Bd1, dat_all$Bd2, 
        dat_all$RT, as.numeric(dat_all$ttp), as.numeric(dat_all$ttF), 
        as.integer(as.factor(dat_all$participant_id)), dat_all$is_test, 
        n_gc, n_mli, act_feat, TRUE
    )
    return(list(train_nll=final_res$Train_NLL, test_nll=final_res$Test_NLL, test_count=final_res$Test_Count))
}

# --- STEP 1: GC GRID SEARCH ---
cat("\\n===========================================\\n")
cat("STEP 1: Granular Cell (GC) Dimensionality Search\\n")
cat("===========================================\\n")
gc_list <- c(50, 100, 200, 400, 800)
all_feat <- as.integer(0:5)
best_gc <- 200
best_test_nll <- 1e9

for (gc in gc_list) {
    cat(sprintf("Testing N_GC = %d... ", gc))
    res <- run_optimization(gc, 40, all_feat, max_iter=25) # Slightly shorter max_iter to blast through grid fast
    cat(sprintf("Train NLL: %.2f, Test NLL: %.2f\\n", res$train_nll, res$test_nll))
    if (res$test_nll < best_test_nll) {
        best_test_nll <- res$test_nll
        best_gc <- gc
    }
}
cat(sprintf("\\nOptimal N_GC = %d\\n", best_gc))

# --- STEP 2: FEATURE ABLATION ---
cat("\\n===========================================\\n")
cat("STEP 2: Feature Ablation Study\\n")
cat(sprintf("Using optimal N_GC = %d, N_MLI = 40\\n", best_gc))
cat("===========================================\\n")

features <- c("0: Prev_Choice", "1: Prev_Outcome", "2: D_Curr", "3: D_Diff", "4: Prev_RT", "5: Prev_ITI")
ablation_results <- list()

for (f_idx in 0:5) {
    cat(sprintf("Ablating %s... ", features[f_idx+1]))
    ablated_feat <- all_feat[all_feat != f_idx]
    res <- run_optimization(best_gc, 40, ablated_feat, max_iter=25)
    cat(sprintf("Train NLL: %.2f, Test NLL: %.2f\\n", res$train_nll, res$test_nll))
    ablation_results[[features[f_idx+1]]] <- res
}

cat("\\n===========================================\\n")
cat("Summary of Results:\\n")
cat(sprintf("Baseline (All Features, Best GC): Test NLL = %.2f\\n", best_test_nll))
for (f_idx in 0:5) {
    fname <- features[f_idx+1]
    nll <- ablation_results[[fname]]$test_nll
    delta <- nll - best_test_nll
    cat(sprintf("Ablated %s -> Test NLL: %.2f (Delta: %+.2f)\\n", fname, nll, delta))
}
