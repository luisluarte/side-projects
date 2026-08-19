library(cmaes)
library(Rcpp)
library(dplyr)

# 1. Load 20 participants
dataset_path <- "data/raw/behavioral_compilate.csv"
if (!file.exists(dataset_path)) {
  dataset_path <- "c:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv"
}

dat_all <- read.csv(dataset_path)
dat_all[['RT']] <- (as.numeric(dat_all[['ttr']]) - as.numeric(dat_all[['ttp']])) / 1000.0
dat_all <- dat_all[!is.na(dat_all[['RT']]) & dat_all[['RT']] >= 0.1 & dat_all[['RT']] <= 3.0 & dat_all[['Resp']] %in% c(1, 2), ]

set.seed(123) # Different seed to avoid over-tuning to previous 50 participants
sample_participants <- sample(unique(dat_all[['participant_id']]), 20)
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
NumericVector evaluate_eccm_cv_dynamic(
    const NumericVector& phi_15d, const IntegerVector& resp_R, const IntegerVector& out_R,
    const NumericVector& m1_R, const NumericVector& m2_R, const NumericVector& rt_R,
    const NumericVector& ttp_R, const NumericVector& ttf_R, const IntegerVector& subj_idx_R,
    const IntegerVector& is_test_R, int use_goc, int use_dat, int N_MF, int N_GC, int N_MLI
) {
  int N_t = resp_R.size();
  double beta_v=phi_15d[0], a_0=phi_15d[1], t_nd=phi_15d[2], kappa_a=phi_15d[3], mu_beta=phi_15d[4], sigma_beta=phi_15d[5];
  double lambda_d=phi_15d[6], mu_tau=phi_15d[7], sigma_tau=phi_15d[8], rho_base=phi_15d[9], eta=phi_15d[10], lambda=phi_15d[11], theta_th=phi_15d[12];
  double alpha_goc = phi_15d[13];
  double kappa_th = phi_15d[14];
  
  SimpleRNG rng(42);
  
  std::vector<int> mf_c(N_MF), mf_d(N_MF); std::vector<double> mf_beta(N_MF);
  for(int j=0; j<N_MF; ++j) {
      mf_c[j] = rng.next() % 6; 
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
  
  double train_nll = 0.0;
  std::vector<double> test_nlls(21, 0.0); 
  
  for (int t=0; t<N_t; ++t) {
    if (t>0 && subj_idx_R[t]!=subj_idx_R[t-1]) {
      std::fill(z_GC_prev.begin(), z_GC_prev.end(), 0.0); std::fill(z_GC_curr.begin(), z_GC_curr.end(), 0.0);
      std::fill(W_PF1.begin(), W_PF1.end(), 0.0); std::fill(W_PF2.begin(), W_PF2.end(), 0.0);
      for (int d=0; d<15; ++d) std::fill(state_hist[d].begin(), state_hist[d].end(), 0.0);
    }
    int ch = resp_R[t], out = out_R[t], subj = subj_idx_R[t];
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

    double G_t_input = 0.0;
    std::vector<double> in_sum_vec(N_GC, 0.0);
    for (int i=0; i<N_GC; ++i) {
        for (int k=0; k<4; ++k) in_sum_vec[i] += gc_mossy_weights[i][k]*u_MF[gc_mossy_map[i][k]];
        G_t_input += in_sum_vec[i];
    }
    G_t_input /= (double)N_GC;

    for (int i=0; i<N_GC; ++i) {
        double gamma_decay = rho_base + (1.0-rho_base)*std::exp(-delta_t_val/tau_vec[i]);
        double base_update = in_sum_vec[i] + gamma_decay*z_GC_prev[i];
        if (use_goc == 1) base_update -= alpha_goc * G_t_input;
        z_GC_curr[i] = std::max(0.0, base_update);
    }

    double H_t = 0.0;
    std::vector<double> pool_sum_vec(N_MLI, 0.0);
    for (int k=0; k<N_MLI; ++k) {
        for (int i=0; i<N_GC; ++i) pool_sum_vec[k] += W_GC_MLI[k][i]*z_GC_curr[i];
        H_t += pool_sum_vec[k];
    }
    H_t /= (double)N_MLI;

    double current_theta = theta_th;
    if (use_dat == 1) current_theta += kappa_th * H_t;

    std::vector<double> h_MLI(N_MLI, 0.0);
    double l1_mli_sum = 1e-12;
    for (int k=0; k<N_MLI; ++k) {
        h_MLI[k] = std::max(0.0, pool_sum_vec[k] - current_theta);
        l1_mli_sum += h_MLI[k];
    }
    
    double S_MLI = 0.0;
    for (int k=0; k<N_MLI; ++k) { double pk = h_MLI[k]/l1_mli_sum; if (pk>1e-12) S_MLI -= pk*std::log(pk); }
    double norm_S = S_MLI / std::log((double)N_MLI);

    double y_PC1=0.0, y_PC2=0.0;
    for (int k=0; k<N_MLI; ++k) { y_PC1 += W_PF1[k]*h_MLI[k]; y_PC2 += W_PF2[k]*h_MLI[k]; }

    double pc_diff = y_PC1 - y_PC2;
    double v_t_ddm = beta_v*pc_diff;
    double a_t = std::max(0.30, a_0 + kappa_a*norm_S);
    
    double dens = wiener_pdf(rt_R[t], ch, v_t_ddm, a_t, t_nd);
    double nll_t = -std::log(dens);

    if (is_test_R[t] == 0) {
        train_nll += nll_t;
        double target = ((double)out - 0.5)*2.0;
        double delta_IO = target - ((ch==1) ? y_PC1 : y_PC2);
        for (int k=0; k<N_MLI; ++k) {
            if (ch==1) { W_PF1[k] += eta*delta_IO*h_MLI[k] - lambda*W_PF1[k]; W_PF2[k] += -lambda*W_PF2[k]; }
            else       { W_PF2[k] += eta*delta_IO*h_MLI[k] - lambda*W_PF2[k]; W_PF1[k] += -lambda*W_PF1[k]; }
        }
    } else {
        if(subj >= 1 && subj <= 20) test_nlls[subj] += nll_t;
    }
    
    z_GC_prev = z_GC_curr;
  }
  
  if (std::isnan(train_nll) || std::isinf(train_nll)) train_nll = 1e9;
  test_nlls[0] = train_nll; 
  return wrap(test_nlls);
}
'
sourceCpp(code = cpp_code)

grid_mf <- c(20, 40, 60)
grid_gc <- c(100, 200, 400)
grid_mli <- c(40, 80, 120)

results_list <- list()

start_time <- Sys.time()

for (mf in grid_mf) {
  for (gc in grid_gc) {
    for (mli in grid_mli) {
      if (gc > mf && gc > mli) {
        cat(sprintf("\\nEvaluating MF=%d | GC=%d | MLI=%d\\n", mf, gc, mli))
        
        lower_bounds <- c(b_v = 0.0, a_0 = 0.30, t_nd = 0.10, kappa_a = 0.0, mu_beta = -2.0, sigma_beta = 0.01, lambda_d = 0.0, mu_tau = -2.0, sigma_tau = 0.01, rho_base = 0.0, eta = 0.0, lambda = 0.0, theta_th = 0.0, alpha_goc = 0.0, kappa_th = 0.0)
        upper_bounds <- c(b_v = 3.0, a_0 = 2.50, t_nd = 0.90, kappa_a = 2.0, mu_beta =  2.0, sigma_beta = 2.00, lambda_d = 5.0, mu_tau =  2.0, sigma_tau = 2.00, rho_base = 0.95, eta = 1.0, lambda = 0.5, theta_th = 0.5, alpha_goc = 2.0, kappa_th = 2.0)
        
        lb <- lower_bounds
        ub <- upper_bounds
        initial_phi <- lb + (ub - lb) / 2
        
        obj_fun <- function(phi) {
            if (any(phi < lb) || any(phi > ub)) return(1e9)
            res <- evaluate_eccm_cv_dynamic(phi, as.integer(dat_all$Resp), as.integer(dat_all$F), dat_all$Bd1, dat_all$Bd2, dat_all$RT, as.numeric(dat_all$ttp), as.numeric(dat_all$ttF), dat_all$participant_factor, as.integer(dat_all$is_test), 1, 1, mf, gc, mli)
            return(res[1]) 
        }
        
        # Fast CMA-ES for grid search (15 iterations is enough to gauge convergence slope on N=20)
        cma_res <- cma_es(initial_phi, obj_fun, lower = lb, upper = ub, control = list(maxit = 15, trace = FALSE, sigma = 0.2))
        
        final_res <- evaluate_eccm_cv_dynamic(cma_res$par, as.integer(dat_all$Resp), as.integer(dat_all$F), dat_all$Bd1, dat_all$Bd2, dat_all$RT, as.numeric(dat_all$ttp), as.numeric(dat_all$ttF), dat_all$participant_factor, as.integer(dat_all$is_test), 1, 1, mf, gc, mli)
        
        test_nll_sum <- sum(final_res[2:21])
        cat(sprintf("Test NLL: %.2f\\n", test_nll_sum))
        
        results_list[[length(results_list) + 1]] <- data.frame(
          MF = mf, GC = gc, MLI = mli, Test_NLL = test_nll_sum
        )
      }
    }
  }
}

end_time <- Sys.time()
execution_time <- as.numeric(difftime(end_time, start_time, units="secs"))
cat(sprintf("\\nGrid Search Execution Time: %.2f seconds\\n", execution_time))

df_res <- bind_rows(results_list)
df_res <- df_res[order(df_res$Test_NLL), ]
write.csv(df_res, "results/tables/combined_grid_search.csv", row.names=FALSE)

cat("\\nTop 5 Architectures:\\n")
print(head(df_res, 5))
