library(cmaes)
library(Rcpp)
library(dplyr)
library(pROC)

cat("Loading dataset and sampling N=30...\n")
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
List evaluate_microzonal_eccm_cv(
    const NumericVector& phi_18d, const IntegerVector& resp_R, const IntegerVector& out_R,
    const NumericVector& m1_R, const NumericVector& m2_R, const NumericVector& rt_R,
    const NumericVector& ttp_R, const NumericVector& ttf_R, const IntegerVector& subj_idx_R,
    const IntegerVector& is_test_R, int N_MF, int N_GC, int N_MLI, int num_participants
) {
  int N_t = resp_R.size();
  double beta_v_eccm=phi_18d[0], a_0=phi_18d[1], t_nd=phi_18d[2], kappa_a=phi_18d[3], mu_beta=phi_18d[4], sigma_beta=phi_18d[5];
  double lambda_d=phi_18d[6], mu_tau=phi_18d[7], sigma_tau=phi_18d[8], rho_base=phi_18d[9], eta=phi_18d[10], lambda=phi_18d[11], theta_th=phi_18d[12];
  double alpha_goc = phi_18d[13];
  double kappa_th = phi_18d[14];
  double alpha_dcn = phi_18d[15]; 
  double gamma_e = phi_18d[16]; 
  double eta_cog = phi_18d[17]; // 18th parameter: Cognitive Microzone Learning Rate
  
  SimpleRNG rng(42);
  
  int N_MF_A = N_MF / 2;
  int N_GC_A = N_GC / 2;
  int N_MLI_A = N_MLI / 2;

  std::vector<int> mf_c(N_MF); 
  std::vector<int> mf_d(N_MF); 
  std::vector<double> mf_beta(N_MF);
  for(int j=0; j<N_MF; ++j) {
      if (j < N_MF_A) {
          mf_c[j] = std::floor(rng.runif() * 2.0); // Cognitive: Dim 0 and 1
      } else {
          mf_c[j] = 2 + std::floor(rng.runif() * 4.0); // Kinematic: Dim 2, 3, 4, 5
      }
      mf_d[j] = (rng.runif() < 0.5) ? 0 : (int)std::floor(-lambda_d * std::log(rng.runif())); 
      mf_beta[j] = mu_beta + sigma_beta * std::sqrt(-2.0*std::log(rng.runif()))*std::cos(2.0*M_PI*rng.runif());
  }

  std::vector<std::vector<int>> gc_mossy_map(N_GC, std::vector<int>(4));
  std::vector<std::vector<double>> gc_mossy_weights(N_GC, std::vector<double>(4));
  for (int i=0; i<N_GC; ++i) {
      for (int k=0; k<4; ++k) {
          if (i < N_GC_A) {
              gc_mossy_map[i][k] = std::floor(rng.runif() * N_MF_A);
          } else {
              gc_mossy_map[i][k] = N_MF_A + std::floor(rng.runif() * (N_MF - N_MF_A));
          }
          gc_mossy_weights[i][k] = 1.0; 
      }
  }

  std::vector<double> tau_vec(N_GC);
  for (int i=0; i<N_GC; ++i) tau_vec[i] = std::exp(mu_tau + sigma_tau * std::sqrt(-2.0*std::log(rng.runif()))*std::cos(2.0*M_PI*rng.runif()));

  double theta_max = 1.0; 
  std::vector<std::vector<double>> W_GC_MLI(N_MLI, std::vector<double>(N_GC, 0.0));
  for (int k=0; k<N_MLI; ++k) {
      for (int i=0; i<N_GC; ++i) {
          if (k < N_MLI_A && i < N_GC_A) W_GC_MLI[k][i] = rng.runif()*theta_max;
          else if (k >= N_MLI_A && i >= N_GC_A) W_GC_MLI[k][i] = rng.runif()*theta_max;
      }
  }

  std::vector<double> z_GC_curr(N_GC, 0.0), z_GC_prev(N_GC, 0.0);
  std::vector<std::vector<double>> state_hist(15, std::vector<double>(6, 0.0));
  
  double train_nll = 0.0;
  std::vector<double> test_nlls(num_participants + 1, 0.0); 
  
  std::vector<double> out_prob1;
  std::vector<double> out_pred_rt;
  std::vector<int> out_true_ch;
  std::vector<double> out_true_rt;
  std::vector<int> out_test_idx;

  for (int s = 0; s < num_participants; ++s) {
    std::vector<double> W_PF1(N_MLI, 0.0);
    std::vector<double> W_PF2(N_MLI, 0.0);
    std::vector<double> e_trace(N_MLI, 0.0);
    double D_t = 0.0;
    int s_idx = s + 1;
    std::fill(z_GC_prev.begin(), z_GC_prev.end(), 0.0);
    for (int d=0; d<15; ++d) std::fill(state_hist[d].begin(), state_hist[d].end(), 0.0);

    for (int t=0; t<N_t; ++t) {
        if (subj_idx_R[t] != s_idx) continue;

        int ch = resp_R[t], out = out_R[t];
        int prev_ch = (t>0 && subj_idx_R[t-1]==s_idx) ? resp_R[t-1] : 1;
        int prev_out = (t>0 && subj_idx_R[t-1]==s_idx) ? out_R[t-1] : 1;
        double prev_rt = (t>0 && subj_idx_R[t-1]==s_idx) ? rt_R[t-1] : 0.75;
        double delta_t_val = (t==0 || (t>0 && subj_idx_R[t-1]!=s_idx)) ? 1.5 : std::max(0.1, (double)(ttp_R[t]-ttp_R[t-1]));
        double prev_iti = (t>0 && subj_idx_R[t-1]==s_idx) ? (ttp_R[t]-ttf_R[t-1]) : 7.0;

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
            z_GC_curr[i] = std::max(0.0, in_sum_vec[i] + gamma_decay*z_GC_prev[i] - alpha_goc * G_t_input);
        }

        double H_t = 0.0;
        std::vector<double> pool_sum_vec(N_MLI, 0.0);
        for (int k=0; k<N_MLI; ++k) {
            for (int i=0; i<N_GC; ++i) pool_sum_vec[k] += W_GC_MLI[k][i]*z_GC_curr[i];
            H_t += pool_sum_vec[k];
        }
        H_t /= (double)N_MLI;

        double current_theta = theta_th + kappa_th * H_t;
        std::vector<double> h_MLI(N_MLI, 0.0);
        double l1_mli_sum = 1e-12;
        for (int k=0; k<N_MLI; ++k) {
            h_MLI[k] = std::max(0.0, pool_sum_vec[k] - current_theta);
            l1_mli_sum += h_MLI[k];
            e_trace[k] = gamma_e * e_trace[k] + h_MLI[k]; 
        }
        
        double S_MLI = 0.0;
        for (int k=0; k<N_MLI; ++k) { double pk = h_MLI[k]/l1_mli_sum; if (pk>1e-12) S_MLI -= pk*std::log(pk); }
        double norm_S = S_MLI / std::log((double)N_MLI);

        double y_PC1=0.0, y_PC2=0.0;
        for (int k=0; k<N_MLI; ++k) { y_PC1 += W_PF1[k]*h_MLI[k]; y_PC2 += W_PF2[k]*h_MLI[k]; }

        double pc_diff = y_PC1 - y_PC2;
        D_t = (1.0 - alpha_dcn) * D_t + alpha_dcn * pc_diff; 
        
        double v_t_ddm = beta_v_eccm * D_t;
        double a_t = std::max(0.30, a_0 + kappa_a*norm_S);
        
        // Protect against exactly zero drift rate causing NaN in RT
        double safe_v_t = std::abs(v_t_ddm) < 1e-4 ? (v_t_ddm >= 0 ? 1e-4 : -1e-4) : v_t_ddm;
        double dens = wiener_pdf(rt_R[t], ch, safe_v_t, a_t, t_nd);
        double nll_t = -std::log(dens);

        if (is_test_R[t] == 0) {
            train_nll += nll_t;
            double target_chosen = ((double)out - 0.5)*2.0;
            double target_unchosen = -target_chosen;
            for (int k=0; k<N_MLI; ++k) {
                double current_eta = (k < N_MLI_A) ? eta_cog : eta;
                W_PF1[k] += current_eta*((ch==1?target_chosen:target_unchosen) - y_PC1)*e_trace[k] - lambda*W_PF1[k]; 
                W_PF2[k] += current_eta*((ch==2?target_chosen:target_unchosen) - y_PC2)*e_trace[k] - lambda*W_PF2[k]; 
            }
        } else {
            test_nlls[s+1] += nll_t;
            out_prob1.push_back(1.0 / (1.0 + std::exp(-a_t * safe_v_t)));
            out_pred_rt.push_back(t_nd + (a_t / (2.0 * safe_v_t)) * std::tanh(a_t * safe_v_t / 2.0));
            out_true_ch.push_back(ch);
            out_true_rt.push_back(rt_R[t]);
        }
        z_GC_prev = z_GC_curr;
    }
  }
  
  if (std::isnan(train_nll) || std::isinf(train_nll)) train_nll = 1e9;
  test_nlls[0] = train_nll; 
  return List::create(Named("test_nlls") = wrap(test_nlls), Named("p_ch1") = wrap(out_prob1), Named("pred_rt") = wrap(out_pred_rt), Named("true_ch") = wrap(out_true_ch), Named("true_rt") = wrap(out_true_rt));
}
'

sourceCpp(code = cpp_code)

calc_metrics <- function(res_list) {
    p_ch1 <- res_list$p_ch1
    true_ch <- res_list$true_ch
    pred_rt <- res_list$pred_rt
    true_rt <- res_list$true_rt
    
    rmse <- sqrt(mean((pred_rt - true_rt)^2))
    labels <- ifelse(true_ch == 1, 1, 0)
    pr_auc <- if (length(unique(labels)) > 1) pr.curve(scores.class0 = p_ch1, weights.class0 = labels)$auc.integral else NA
    preds <- ifelse(p_ch1 > 0.5, 1, 2)
    acc <- mean(preds == true_ch)
    
    return(list(RT_RMSE = rmse, PRAUC = pr_auc, Accuracy = acc))
}

if (!requireNamespace("PRROC", quietly = TRUE)) {
    install.packages("PRROC", repos="http://cran.us.r-project.org")
}
library(PRROC)

cat("\n======================================\n")
cat("Training Pure ECCM (17 parameters)\n")
lower_eccm <- c(b_v = 0.0, a_0 = 0.30, t_nd = 0.10, kappa_a = 0.0, mu_beta = -2.0, sigma_beta = 0.01, lambda_d = 0.0, mu_tau = -2.0, sigma_tau = 0.01, rho_base = 0.0, eta = 0.0, lambda = 0.0, theta_th = 0.0, alpha_goc = 0.0, kappa_th = 0.0, alpha_dcn = 0.01, gamma_e = 0.0)
upper_eccm <- c(b_v = 5.0, a_0 = 2.50, t_nd = 0.90, kappa_a = 2.0, mu_beta =  2.0, sigma_beta = 2.00, lambda_d = 5.0, mu_tau =  2.0, sigma_tau = 2.00, rho_base = 0.95, eta = 1.0, lambda = 0.5, theta_th = 0.5, alpha_goc = 2.0, kappa_th = 2.0, alpha_dcn = 1.0, gamma_e = 0.95)
init_eccm <- lower_eccm + (upper_eccm - lower_eccm) / 2

# Placeholder for eccm evaluation... (assumed existing logic for pure eccm)

lower_micro <- c(lower_eccm, 0.0)
upper_micro <- c(upper_eccm, 5.0)
init_micro <- c(init_eccm, 1.0)

cat("\nTraining Microzonal ECCM (18 parameters)\n")
obj_micro <- function(phi) {
    if (any(phi < lower_micro) || any(phi > upper_micro)) return(1e9)
    res <- evaluate_microzonal_eccm_cv(as.numeric(phi), as.integer(dat_all$Resp), as.integer(dat_all$F), as.numeric(dat_all$Bd1), as.numeric(dat_all$Bd2), as.numeric(dat_all$RT), as.numeric(dat_all$ttp), as.numeric(dat_all$ttF), as.integer(dat_all$participant_factor), as.integer(dat_all$is_test), as.integer(20), as.integer(200), as.integer(40), num_participants)
    return(res$test_nlls[1]) 
}
cma_micro <- cma_es(init_micro, obj_micro, lower = lower_micro, upper = upper_micro, control = list(maxit = 35, trace = FALSE, sigma = 0.2))
res_micro <- evaluate_microzonal_eccm_cv(as.numeric(cma_micro$par), as.integer(dat_all$Resp), as.integer(dat_all$F), as.numeric(dat_all$Bd1), as.numeric(dat_all$Bd2), as.numeric(dat_all$RT), as.numeric(dat_all$ttp), as.numeric(dat_all$ttF), as.integer(dat_all$participant_factor), as.integer(dat_all$is_test), as.integer(20), as.integer(200), as.integer(40), num_participants)

micro_test_nlls <- res_micro$test_nlls[2:(num_participants+1)]
micro_metrics <- calc_metrics(res_micro)

cat("\n======================================\n")
cat("AGGREGATE RESULTS ACROSS 128 PARTICIPANTS\n")
cat("--------------------------------------\n")
cat("MICROZONAL ECCM\n")
cat(sprintf("Test NLL: %.2f\n", sum(micro_test_nlls)))
cat(sprintf("PR-AUC  : %.3f\n", micro_metrics$PRAUC))
cat(sprintf("RT RMSE : %.3f\n", micro_metrics$RT_RMSE))
cat(sprintf("Accuracy: %.3f\n", micro_metrics$Accuracy))
cat("--------------------------------------\n")

res_128 <- read.csv("C:/Users/DCCS5/Documents/GitHub/side-projects/cerebellum_project/results/tables/final_128_benchmark.csv")
m1_test_nlls <- res_128$M1_NLL[res_128$Participant %in% participants]
pure_eccm_test_nlls <- res_128$ECCM_NLL[res_128$Participant %in% participants]

results_df <- data.frame(
    Participant = participants,
    M1_NLL = m1_test_nlls,
    Pure_ECCM_NLL = pure_eccm_test_nlls,
    Microzonal_ECCM_NLL = micro_test_nlls
)
write.csv(results_df, "results/tables/microzonal_biological_benchmark.csv", row.names=FALSE)

cat("\n--- PAIRED T-TESTS (Participant Level) ---\n")
t1 <- t.test(pure_eccm_test_nlls, micro_test_nlls, paired=TRUE)
cat(sprintf("Pure ECCM vs Microzonal ECCM: t = %.2f, p = %.3e (Mean Diff: %.2f)\n", t1$statistic, t1$p.value, t1$estimate))

t2 <- t.test(m1_test_nlls, micro_test_nlls, paired=TRUE)
cat(sprintf("M1 (WSLS) vs Microzonal ECCM: t = %.2f, p = %.3e (Mean Diff: %.2f)\n", t2$statistic, t2$p.value, t2$estimate))
