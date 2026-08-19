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

set.seed(456) 
sample_participants <- sample(unique(dat_all[['participant_id']]), 50)
dat_all <- dat_all[dat_all[['participant_id']] %in% sample_participants, ]
dat_all$participant_factor <- as.integer(as.factor(dat_all$participant_id))

# Split 70/30 chronologically
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
NumericVector evaluate_eccm_cv_mod(
    const NumericVector& phi_15d, const IntegerVector& resp_R, const IntegerVector& out_R,
    const NumericVector& m1_R, const NumericVector& m2_R, const NumericVector& rt_R,
    const NumericVector& ttp_R, const NumericVector& ttf_R, const IntegerVector& subj_idx_R,
    const IntegerVector& is_test_R, int use_goc, int use_dat
) {
  int N_t = resp_R.size();
  double beta_v=phi_15d[0], a_0=phi_15d[1], t_nd=phi_15d[2], kappa_a=phi_15d[3], mu_beta=phi_15d[4], sigma_beta=phi_15d[5];
  double lambda_d=phi_15d[6], mu_tau=phi_15d[7], sigma_tau=phi_15d[8], rho_base=phi_15d[9], eta=phi_15d[10], lambda=phi_15d[11], theta_th=phi_15d[12];
  double alpha_goc = phi_15d[13];
  double kappa_th = phi_15d[14];
  
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
  
  double train_nll = 0.0;
  std::vector<double> test_nlls(51, 0.0); 
  
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
        test_nlls[subj] += nll_t;
    }
    
    z_GC_prev = z_GC_curr;
  }
  
  if (std::isnan(train_nll) || std::isinf(train_nll)) train_nll = 1e9;
  test_nlls[0] = train_nll; 
  return wrap(test_nlls);
}
'
sourceCpp(code = cpp_code)

run_cv_model <- function(model_name, use_goc, use_dat) {
    cat(sprintf("\\n======================================\\n"))
    cat(sprintf("Training Model: %s (GoC=%d, DAT=%d)\\n", model_name, use_goc, use_dat))
    
    lower_bounds <- c(b_v = 0.0, a_0 = 0.30, t_nd = 0.10, kappa_a = 0.0, mu_beta = -2.0, sigma_beta = 0.01, lambda_d = 0.0, mu_tau = -2.0, sigma_tau = 0.01, rho_base = 0.0, eta = 0.0, lambda = 0.0, theta_th = 0.0, alpha_goc = 0.0, kappa_th = 0.0)
    upper_bounds <- c(b_v = 3.0, a_0 = 2.50, t_nd = 0.90, kappa_a = 2.0, mu_beta =  2.0, sigma_beta = 2.00, lambda_d = 5.0, mu_tau =  2.0, sigma_tau = 2.00, rho_base = 0.95, eta = 1.0, lambda = 0.5, theta_th = 0.5, alpha_goc = 2.0, kappa_th = 2.0)
    
    active_params <- c("b_v", "a_0", "t_nd", "kappa_a", "mu_beta", "sigma_beta", "lambda_d", "mu_tau", "sigma_tau", "rho_base", "eta", "lambda", "theta_th")
    if (use_goc) active_params <- c(active_params, "alpha_goc")
    if (use_dat) active_params <- c(active_params, "kappa_th")
    
    lb <- lower_bounds[active_params]
    ub <- upper_bounds[active_params]
    initial_phi <- lb + (ub - lb) / 2
    
    obj_fun <- function(phi) {
        if (any(phi < lb) || any(phi > ub)) return(1e9)
        
        phi_full <- numeric(15)
        phi_full[1:13] <- phi[1:13]
        phi_full[14] <- if(use_goc) phi["alpha_goc"] else 0.0
        phi_full[15] <- if(use_dat) phi["kappa_th"] else 0.0
        
        res <- evaluate_eccm_cv_mod(phi_full, as.integer(dat_all$Resp), as.integer(dat_all$F), dat_all$Bd1, dat_all$Bd2, dat_all$RT, as.numeric(dat_all$ttp), as.numeric(dat_all$ttF), dat_all$participant_factor, as.integer(dat_all$is_test), use_goc, use_dat)
        return(res[1]) # Train NLL
    }
    
    cma_res <- cma_es(initial_phi, obj_fun, lower = lb, upper = ub, control = list(maxit = 25, trace = TRUE, sigma = 0.2))
    
    # Final eval on Test
    phi_final <- numeric(15)
    phi_final[1:13] <- cma_res$par[1:13]
    phi_final[14] <- if(use_goc) cma_res$par["alpha_goc"] else 0.0
    phi_final[15] <- if(use_dat) cma_res$par["kappa_th"] else 0.0
    
    final_res <- evaluate_eccm_cv_mod(phi_final, as.integer(dat_all$Resp), as.integer(dat_all$F), dat_all$Bd1, dat_all$Bd2, dat_all$RT, as.numeric(dat_all$ttp), as.numeric(dat_all$ttF), dat_all$participant_factor, as.integer(dat_all$is_test), use_goc, use_dat)
    
    test_nlls <- final_res[2:51] # Extract subjects 1 to 50
    return(test_nlls)
}

baseline_test <- run_cv_model("Baseline", 0, 0)
goc_test <- run_cv_model("GoC Only", 1, 0)
dat_test <- run_cv_model("DAT Only", 0, 1)
combo_test <- run_cv_model("Combined (GoC + DAT)", 1, 1)

df_results <- data.frame(
    Participant = 1:50,
    Baseline_NLL = baseline_test,
    GoC_NLL = goc_test,
    DAT_NLL = dat_test,
    Combined_NLL = combo_test
)

cat("\\n======================================\\n")
cat("AGGREGATE TEST NLL ACROSS 50 PARTICIPANTS\\n")
cat(sprintf("Baseline : %.2f\\n", sum(df_results$Baseline_NLL)))
cat(sprintf("GoC Only : %.2f\\n", sum(df_results$GoC_NLL)))
cat(sprintf("DAT Only : %.2f\\n", sum(df_results$DAT_NLL)))
cat(sprintf("Combined : %.2f\\n", sum(df_results$Combined_NLL)))

cat("\\n--- PAIRED T-TESTS (Participant Level) ---\\n")
t1 <- t.test(df_results$Baseline_NLL, df_results$GoC_NLL, paired=TRUE)
cat(sprintf("Baseline vs GoC: t = %.2f, p = %.3e (Mean Diff: %.2f)\\n", t1$statistic, t1$p.value, t1$estimate))

t2 <- t.test(df_results$Baseline_NLL, df_results$DAT_NLL, paired=TRUE)
cat(sprintf("Baseline vs DAT: t = %.2f, p = %.3e (Mean Diff: %.2f)\\n", t2$statistic, t2$p.value, t2$estimate))

t3 <- t.test(df_results$Baseline_NLL, df_results$Combined_NLL, paired=TRUE)
cat(sprintf("Baseline vs Combined: t = %.2f, p = %.3e (Mean Diff: %.2f)\\n", t3$statistic, t3$p.value, t3$estimate))

write.csv(df_results, "results/tables/eccm_topological_ablation_50_subj.csv", row.names=FALSE)
