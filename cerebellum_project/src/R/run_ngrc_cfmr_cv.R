library(cmaes)
library(Rcpp)
library(dplyr)
library(pROC)

cat("Loading dataset and sampling N=128...\n")
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
List evaluate_ngrc_cfmr_cv(
    const NumericVector& phi, const IntegerVector& resp_R, const IntegerVector& out_R,
    const IntegerVector& subj_idx_R, const NumericVector& rt_R, const IntegerVector& is_test_R, int num_participants
) {
  double a = phi[0];
  double t_nd = phi[1];
  double beta_v = phi[2];
  double eta_LTP = phi[3];
  double eta_LTD = phi[4];
  
  double train_nll = 0.0;
  std::vector<double> test_nlls(num_participants + 1, 0.0); 
  
  std::vector<double> out_prob1;
  std::vector<double> out_pred_rt;
  std::vector<double> out_true_ch;
  std::vector<double> out_true_rt;
  
  int N_t = resp_R.size();
  int N_GC = 50;
  
  std::vector<double> W_PF1(N_GC, 0.0);
  std::vector<double> W_PF2(N_GC, 0.0);
  std::vector<std::vector<double>> W_MF_GC(N_GC, std::vector<double>(3, 0.0));
  
  int hist_c1 = 0;
  int hist_c2 = 0;
  int hist_c3 = 0;

  for (int t=0; t<N_t; ++t) {
        int s_idx = subj_idx_R[t];
        
        if (t == 0 || subj_idx_R[t] != subj_idx_R[t-1]) {
            std::fill(W_PF1.begin(), W_PF1.end(), 0.0);
            std::fill(W_PF2.begin(), W_PF2.end(), 0.0);
            hist_c1 = 0; hist_c2 = 0; hist_c3 = 0;
            
            SimpleRNG rng(s_idx + 42);
            for (int i=0; i<N_GC; ++i) {
                for (int j=0; j<3; ++j) {
                    W_MF_GC[i][j] = rng.rnorm() / std::sqrt(3.0);
                }
            }
        }

        int ch = resp_R[t];
        int out = out_R[t];

        std::vector<double> h_t(N_GC, 0.0);
        double Q1 = 0.0;
        double Q2 = 0.0;
        
        for (int i=0; i<N_GC; ++i) {
            double net = W_MF_GC[i][0] * hist_c1 + W_MF_GC[i][1] * hist_c2 + W_MF_GC[i][2] * hist_c3;
            h_t[i] = std::tanh(net);
            Q1 += W_PF1[i] * h_t[i];
            Q2 += W_PF2[i] * h_t[i];
        }

        double v_t_ddm = beta_v * (Q1 - Q2);
        
        double safe_v_t = std::abs(v_t_ddm) < 1e-4 ? (v_t_ddm >= 0 ? 1e-4 : -1e-4) : v_t_ddm;
        double dens = wiener_pdf(rt_R[t], ch, safe_v_t, a, t_nd);
        double nll_t = -std::log(dens);

        if (is_test_R[t] == 0) {
            train_nll += nll_t;
            if (ch == 1) {
                for (int i=0; i<N_GC; ++i) {
                    if (out == 1) W_PF1[i] += eta_LTP * (1.0 - Q1) * h_t[i];
                    else          W_PF1[i] += eta_LTD * (-1.0 - Q1) * h_t[i];
                }
            } else {
                for (int i=0; i<N_GC; ++i) {
                    if (out == 1) W_PF2[i] += eta_LTP * (1.0 - Q2) * h_t[i];
                    else          W_PF2[i] += eta_LTD * (-1.0 - Q2) * h_t[i];
                }
            }
        } else {
            test_nlls[s_idx] += nll_t;
            out_prob1.push_back(1.0 / (1.0 + std::exp(-a * safe_v_t)));
            out_pred_rt.push_back(t_nd + (a / (2.0 * safe_v_t)) * std::tanh(a * safe_v_t / 2.0));
            out_true_ch.push_back(ch);
            out_true_rt.push_back(rt_R[t]);
            
            if (ch == 1) {
                for (int i=0; i<N_GC; ++i) {
                    if (out == 1) W_PF1[i] += eta_LTP * (1.0 - Q1) * h_t[i];
                    else          W_PF1[i] += eta_LTD * (-1.0 - Q1) * h_t[i];
                }
            } else {
                for (int i=0; i<N_GC; ++i) {
                    if (out == 1) W_PF2[i] += eta_LTP * (1.0 - Q2) * h_t[i];
                    else          W_PF2[i] += eta_LTD * (-1.0 - Q2) * h_t[i];
                }
            }
        }
        
        hist_c3 = hist_c2;
        hist_c2 = hist_c1;
        hist_c1 = (ch == 1) ? 1 : -1;
  }
  
  test_nlls[0] = train_nll;
  return List::create(
      Named("test_nlls") = test_nlls,
      Named("prob1") = out_prob1,
      Named("pred_rt") = out_pred_rt,
      Named("true_ch") = out_true_ch,
      Named("true_rt") = out_true_rt
  );
}
'
sourceCpp(code = cpp_code)

calc_metrics <- function(res) {
    p1 <- res$prob1
    preds <- ifelse(p1 >= 0.5, 1, 2)
    acc <- mean(preds == res$true_ch)
    roc_obj <- roc(ifelse(res$true_ch==1, 1, 0), p1, quiet=TRUE)
    prauc <- pr.curve(scores.class0 = p1, weights.class0 = ifelse(res$true_ch==1, 1, 0))$auc.integral
    rt_rmse <- sqrt(mean((res$pred_rt - res$true_rt)^2, na.rm=TRUE))
    return(list(Accuracy=acc, PRAUC=prauc, RT_RMSE=rt_rmse))
}

lower_cfmr <- c(0.1, 0.01, 0.0, 0.0, 0.0)
upper_cfmr <- c(5.0, 1.00, 10.0, 1.0, 1.0)
init_cfmr <-  c(1.0, 0.30, 2.0, 1.0, 1.0)

cat("\nTraining NGRC CFMR (5 parameters)\n")
obj_cfmr <- function(phi) {
    if (any(phi < lower_cfmr) || any(phi > upper_cfmr)) return(1e9)
    res <- evaluate_ngrc_cfmr_cv(as.numeric(phi), as.integer(dat_all$Resp), as.integer(dat_all$F), as.integer(dat_all$participant_factor), as.numeric(dat_all$RT), as.integer(dat_all$is_test), num_participants)
    return(res$test_nlls[1]) 
}

library(PRROC)
cma_cfmr <- cma_es(init_cfmr, obj_cfmr, lower = lower_cfmr, upper = upper_cfmr, control = list(maxit = 150, trace = FALSE, sigma = 0.5))
res_cfmr <- evaluate_ngrc_cfmr_cv(as.numeric(cma_cfmr$par), as.integer(dat_all$Resp), as.integer(dat_all$F), as.integer(dat_all$participant_factor), as.numeric(dat_all$RT), as.integer(dat_all$is_test), num_participants)

cfmr_test_nlls <- res_cfmr$test_nlls[2:(num_participants+1)]
cfmr_metrics <- calc_metrics(res_cfmr)

cat("\n======================================\n")
cat("AGGREGATE RESULTS ACROSS 128 PARTICIPANTS\n")
cat("--------------------------------------\n")
cat("NGRC CFMR\n")
cat(sprintf("Test NLL: %.2f\n", sum(cfmr_test_nlls)))
cat(sprintf("PR-AUC  : %.3f\n", cfmr_metrics$PRAUC))
cat(sprintf("RT RMSE : %.3f\n", cfmr_metrics$RT_RMSE))
cat(sprintf("Accuracy: %.3f\n", cfmr_metrics$Accuracy))
cat("--------------------------------------\n")

res_128 <- read.csv("C:/Users/DCCS5/Documents/GitHub/side-projects/cerebellum_project/results/tables/cfmr_biological_benchmark.csv")
pure_cfmr_test_nlls <- res_128$CFMR_ECCM_NLL[res_128$Participant %in% participants]
m1_test_nlls <- res_128$M1_NLL[res_128$Participant %in% participants]

results_df <- data.frame(
    Participant = participants,
    M1_NLL = m1_test_nlls,
    Pure_CFMR_NLL = pure_cfmr_test_nlls,
    NGRC_CFMR_NLL = cfmr_test_nlls
)
write.csv(results_df, "results/tables/ngrc_cfmr_biological_benchmark.csv", row.names=FALSE)

cat("\n--- PAIRED T-TESTS (Participant Level) ---\n")
t1 <- t.test(pure_cfmr_test_nlls, cfmr_test_nlls, paired=TRUE)
cat(sprintf("Pure CFMR vs NGRC CFMR: t = %.2f, p = %.3e (Mean Diff: %.2f)\n", t1$statistic, t1$p.value, t1$estimate))

t2 <- t.test(m1_test_nlls, cfmr_test_nlls, paired=TRUE)
cat(sprintf("M1 (WSLS) vs NGRC CFMR: t = %.2f, p = %.3e (Mean Diff: %.2f)\n", t2$statistic, t2$p.value, t2$estimate))
