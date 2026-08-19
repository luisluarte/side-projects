library(cmaes)
library(Rcpp)
library(dplyr)

dataset_path <- "data/raw/behavioral_compilate.csv"
if (!file.exists(dataset_path)) {
  dataset_path <- "c:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv"
}

dat_all <- read.csv(dataset_path)
dat_all[['RT']] <- (as.numeric(dat_all[['ttr']]) - as.numeric(dat_all[['ttp']])) / 1000.0
dat_all <- dat_all[!is.na(dat_all[['RT']]) & dat_all[['RT']] >= 0.1 & dat_all[['RT']] <= 3.0 & dat_all[['Resp']] %in% c(1, 2), ]

set.seed(123) # New seed for 30 participants
participants <- unique(dat_all[['participant_id']])
sample_participants <- sample(participants, 30)
dat_all <- dat_all[dat_all[['participant_id']] %in% sample_participants, ]
dat_all$participant_factor <- as.integer(as.factor(dat_all$participant_id))

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
List extract_eccm_topology_cpp(
    const NumericVector& phi_13d, const IntegerVector& resp_R, const IntegerVector& out_R,
    const NumericVector& m1_R, const NumericVector& m2_R, const NumericVector& rt_R,
    const NumericVector& ttp_R, const NumericVector& ttf_R, const IntegerVector& subj_idx_R,
    bool extract_mode = false
) {
  int N_t = resp_R.size();
  double beta_v=phi_13d[0], a_0=phi_13d[1], t_nd=phi_13d[2], kappa_a=phi_13d[3], mu_beta=phi_13d[4], sigma_beta=phi_13d[5];
  double lambda_d=phi_13d[6], mu_tau=phi_13d[7], sigma_tau=phi_13d[8], rho_base=phi_13d[9], eta=phi_13d[10], lambda=phi_13d[11], theta_th=phi_13d[12];
  
  int N_MF = 40, N_GC = 200, N_MLI = 80;
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
  
  double total_nll = 0.0;
  
  // Vectors for extraction
  NumericVector ext_trial(N_t), ext_subj(N_t), ext_choice(N_t), ext_rt(N_t), ext_out(N_t), ext_iti(N_t);
  NumericVector ext_gc_l1(N_t), ext_gc_sparse(N_t), ext_mli_entropy(N_t), ext_pc_diff(N_t);
  NumericVector ext_vt(N_t), ext_at(N_t), ext_lik(N_t), ext_pred_err(N_t);
  
  for (int t=0; t<N_t; ++t) {
    if (t>0 && subj_idx_R[t]!=subj_idx_R[t-1]) {
      std::fill(z_GC_prev.begin(), z_GC_prev.end(), 0.0); std::fill(z_GC_curr.begin(), z_GC_curr.end(), 0.0);
      std::fill(W_PF1.begin(), W_PF1.end(), 0.0); std::fill(W_PF2.begin(), W_PF2.end(), 0.0);
      for (int d=0; d<15; ++d) std::fill(state_hist[d].begin(), state_hist[d].end(), 0.0);
    }
    int ch = resp_R[t], out = out_R[t];
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

    double gc_l1 = 0.0;
    double gc_active = 0.0;
    for (int i=0; i<N_GC; ++i) {
        double in_sum = 0.0;
        for (int k=0; k<4; ++k) in_sum += gc_mossy_weights[i][k]*u_MF[gc_mossy_map[i][k]];
        double gamma_decay = rho_base + (1.0-rho_base)*std::exp(-delta_t_val/tau_vec[i]);
        z_GC_curr[i] = std::max(0.0, in_sum + gamma_decay*z_GC_prev[i]);
        gc_l1 += z_GC_curr[i];
        if (z_GC_curr[i] > 1e-6) gc_active += 1.0;
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

    double pc_diff = y_PC1 - y_PC2;
    double v_t_ddm = beta_v*pc_diff;
    double a_t = std::max(0.30, a_0 + kappa_a*norm_S);
    
    double dens = wiener_pdf(rt_R[t], ch, v_t_ddm, a_t, t_nd);
    total_nll -= std::log(dens);
    
    // Predictive accuracy (if v_t > 0 -> 1, if v_t < 0 -> 2)
    int pred_ch = (v_t_ddm > 0) ? 1 : 2;
    double pred_err = (pred_ch != ch) ? 1.0 : 0.0;
    
    if (extract_mode) {
        ext_trial[t] = t; ext_subj[t] = subj_idx_R[t]; ext_choice[t] = ch; ext_rt[t] = rt_R[t]; ext_out[t] = out; ext_iti[t] = prev_iti;
        ext_gc_l1[t] = gc_l1; ext_gc_sparse[t] = gc_active / (double)N_GC; ext_mli_entropy[t] = norm_S; ext_pc_diff[t] = pc_diff;
        ext_vt[t] = v_t_ddm; ext_at[t] = a_t; ext_lik[t] = dens; ext_pred_err[t] = pred_err;
    }

    double target = ((double)out - 0.5)*2.0;
    double delta_IO = target - ((ch==1) ? y_PC1 : y_PC2);

    for (int k=0; k<N_MLI; ++k) {
        if (ch==1) { W_PF1[k] += eta*delta_IO*h_MLI[k] - lambda*W_PF1[k]; W_PF2[k] += -lambda*W_PF2[k]; }
        else       { W_PF2[k] += eta*delta_IO*h_MLI[k] - lambda*W_PF2[k]; W_PF1[k] += -lambda*W_PF1[k]; }
    }
    z_GC_prev = z_GC_curr;
  }
  
  if (extract_mode) {
      return List::create(
          Named("Subject") = ext_subj, Named("Trial") = ext_trial, Named("Choice") = ext_choice, Named("RT") = ext_rt, Named("Outcome") = ext_out, Named("Prev_ITI") = ext_iti,
          Named("GC_L1") = ext_gc_l1, Named("GC_Sparsity") = ext_gc_sparse, Named("MLI_Entropy") = ext_mli_entropy, Named("PC_Diff") = ext_pc_diff,
          Named("v_t") = ext_vt, Named("a_t") = ext_at, Named("Likelihood") = ext_lik, Named("Pred_Error") = ext_pred_err
      );
  }
  
  if (std::isnan(total_nll) || std::isinf(total_nll)) return List::create(Named("Obj")=1e9);
  return List::create(Named("Obj")=total_nll);
}
'
sourceCpp(code = cpp_code)

# --- CMA-ES GLOBAL OPTIMIZATION (30 Participants) ---
lower_bounds <- c(b_v = 0.0, a_0 = 0.30, t_nd = 0.10, kappa_a = 0.0, mu_beta = -2.0, sigma_beta = 0.01, lambda_d = 0.0, mu_tau = -2.0, sigma_tau = 0.01, rho_base = 0.0, eta = 0.0, lambda = 0.0, theta_th = 0.0)
upper_bounds <- c(b_v = 3.0, a_0 = 2.50, t_nd = 0.90, kappa_a = 2.0, mu_beta =  2.0, sigma_beta = 2.00, lambda_d = 5.0, mu_tau =  2.0, sigma_tau = 2.00, rho_base = 0.95, eta = 1.0, lambda = 0.5, theta_th = 0.5)
initial_phi <- lower_bounds + (upper_bounds - lower_bounds) / 2

cat("Running Global CMA-ES on 30 Participants...\\n")
obj_fun <- function(phi) {
    if (any(phi < lower_bounds) || any(phi > upper_bounds)) return(1e9)
    res <- extract_eccm_topology_cpp(phi, as.integer(dat_all$Resp), as.integer(dat_all$F), dat_all$Bd1, dat_all$Bd2, dat_all$RT, as.numeric(dat_all$ttp), as.numeric(dat_all$ttF), dat_all$participant_factor, FALSE)
    return(res$Obj)
}
cma_res <- cma_es(initial_phi, obj_fun, lower = lower_bounds, upper = upper_bounds, control = list(maxit = 35, trace = TRUE, sigma = 0.2))
phi_opt <- cma_res$par
cat(sprintf("Optimal Global NLL: %.2f\\n", cma_res$value))

# --- TOPOLOGICAL METRIC EXTRACTION ---
cat("Extracting topological states...\\n")
states_list <- extract_eccm_topology_cpp(phi_opt, as.integer(dat_all$Resp), as.integer(dat_all$F), dat_all$Bd1, dat_all$Bd2, dat_all$RT, as.numeric(dat_all$ttp), as.numeric(dat_all$ttF), dat_all$participant_factor, TRUE)

df_states <- as.data.frame(states_list)

# --- ANALYZE FAILURES VS TOPOLOGY ---
cat("\\n=== Topology vs Behavioral Prediction Error ===\\n")
df_states$Prediction_State <- ifelse(df_states$Pred_Error == 1, "Failed (Wrong Choice)", "Success (Correct Choice)")

agg_res <- df_states %>%
    group_by(Prediction_State) %>%
    summarise(
        Count = n(),
        Mean_Likelihood = mean(Likelihood),
        Mean_GC_Sparsity = mean(GC_Sparsity),
        Mean_GC_L1 = mean(GC_L1),
        Mean_MLI_Entropy = mean(MLI_Entropy),
        Mean_PC_Diff_Abs = mean(abs(PC_Diff))
    )
print(agg_res)

# Let's perform some t-tests to see if topological states systematically differ when the model fails
cat("\\n--- T-Tests on Topological Mechanics ---\\n")
tt_entropy <- t.test(MLI_Entropy ~ Prediction_State, data=df_states)
cat(sprintf("MLI Spatial Entropy (Failed vs Success): t = %.2f, p = %.2e\\n", tt_entropy$statistic, tt_entropy$p.value))

tt_sparsity <- t.test(GC_Sparsity ~ Prediction_State, data=df_states)
cat(sprintf("GC Sparsity (Failed vs Success): t = %.2f, p = %.2e\\n", tt_sparsity$statistic, tt_sparsity$p.value))

write.csv(df_states, "results/tables/eccm_topological_states_30_subj.csv", row.names=FALSE)
cat("\\nSaved states to results/tables/eccm_topological_states_30_subj.csv\\n")
