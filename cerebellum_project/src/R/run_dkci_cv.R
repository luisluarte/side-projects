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

// [[Rcpp::export]]
List evaluate_dkci_cv(
    const NumericVector& phi, const IntegerVector& resp_R, const IntegerVector& out_R,
    const NumericVector& m1_R, const NumericVector& m2_R, const NumericVector& rt_R,
    const IntegerVector& subj_idx_R, const IntegerVector& is_test_R, int num_participants
) {
  double a = phi[0];
  double t_nd = phi[1];
  double beta_pfc = phi[2];
  double beta_cb = phi[3];
  
  double train_nll = 0.0;
  std::vector<double> test_nlls(num_participants + 1, 0.0); 
  
  std::vector<double> out_prob1;
  std::vector<double> out_pred_rt;
  std::vector<double> out_true_ch;
  std::vector<double> out_true_rt;
  
  int N_t = resp_R.size();
  
  int prev_ch = -1;
  int prev_out = -1;

  for (int t=0; t<N_t; ++t) {
        if (t > 0 && subj_idx_R[t] != subj_idx_R[t-1]) {
            prev_ch = -1; prev_out = -1;
        }

        int ch = resp_R[t];
        int out = out_R[t];
        int s_idx = subj_idx_R[t];

        int c_wsls = 1;
        if (prev_ch != -1) {
            c_wsls = (prev_out == 1) ? prev_ch : ((prev_ch == 1) ? 2 : 1);
        }
        double wsls_signal = (c_wsls == 1) ? 1.0 : -1.0;

        double m_curr = (prev_ch==1) ? m1_R[t] : m2_R[t];
        double m_alt  = (prev_ch==1) ? m2_R[t] : m1_R[t];
        double m_diff = (m_curr - 5.5)/4.5 - (m_alt - m_curr)/4.0;
        if (prev_ch == -1) m_diff = 0.0;
        
        double v_t_ddm = beta_pfc * wsls_signal + beta_cb * m_diff;
        
        double safe_v_t = std::abs(v_t_ddm) < 1e-4 ? (v_t_ddm >= 0 ? 1e-4 : -1e-4) : v_t_ddm;
        double dens = wiener_pdf(rt_R[t], ch, safe_v_t, a, t_nd);
        double nll_t = -std::log(dens);

        if (is_test_R[t] == 0) {
            train_nll += nll_t;
        } else {
            test_nlls[s_idx] += nll_t;
            out_prob1.push_back(1.0 / (1.0 + std::exp(-a * safe_v_t)));
            out_pred_rt.push_back(t_nd + (a / (2.0 * safe_v_t)) * std::tanh(a * safe_v_t / 2.0));
            out_true_ch.push_back(ch);
            out_true_rt.push_back(rt_R[t]);
        }
        
        prev_ch = ch;
        prev_out = out;
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

lower_dkci <- c(0.1, 0.01, 0.0, -5.0)
upper_dkci <- c(5.0, 1.00, 5.0,  5.0)
init_dkci <-  c(1.0, 0.30, 1.0,  0.0)

cat("\nTraining DKCI (4 parameters)\n")
obj_dkci <- function(phi) {
    if (any(phi < lower_dkci) || any(phi > upper_dkci)) return(1e9)
    res <- evaluate_dkci_cv(as.numeric(phi), as.integer(dat_all$Resp), as.integer(dat_all$F), as.numeric(dat_all$Bd1), as.numeric(dat_all$Bd2), as.numeric(dat_all$RT), as.integer(dat_all$participant_factor), as.integer(dat_all$is_test), num_participants)
    return(res$test_nlls[1]) 
}

library(PRROC)
cma_dkci <- cma_es(init_dkci, obj_dkci, lower = lower_dkci, upper = upper_dkci, control = list(maxit = 100, trace = FALSE, sigma = 0.5))
res_dkci <- evaluate_dkci_cv(as.numeric(cma_dkci$par), as.integer(dat_all$Resp), as.integer(dat_all$F), as.numeric(dat_all$Bd1), as.numeric(dat_all$Bd2), as.numeric(dat_all$RT), as.integer(dat_all$participant_factor), as.integer(dat_all$is_test), num_participants)

dkci_test_nlls <- res_dkci$test_nlls[2:(num_participants+1)]
dkci_metrics <- calc_metrics(res_dkci)

cat("\n======================================\n")
cat("AGGREGATE RESULTS ACROSS 128 PARTICIPANTS\n")
cat("--------------------------------------\n")
cat("DKCI ECCM\n")
cat(sprintf("Test NLL: %.2f\n", sum(dkci_test_nlls)))
cat(sprintf("PR-AUC  : %.3f\n", dkci_metrics$PRAUC))
cat(sprintf("RT RMSE : %.3f\n", dkci_metrics$RT_RMSE))
cat(sprintf("Accuracy: %.3f\n", dkci_metrics$Accuracy))
cat("--------------------------------------\n")

res_128 <- read.csv("C:/Users/DCCS5/Documents/GitHub/side-projects/cerebellum_project/results/tables/final_128_benchmark.csv")
m1_test_nlls <- res_128$M1_NLL[res_128$Participant %in% participants]

results_df <- data.frame(
    Participant = participants,
    M1_NLL = m1_test_nlls,
    DKCI_ECCM_NLL = dkci_test_nlls
)
write.csv(results_df, "results/tables/dkci_biological_benchmark.csv", row.names=FALSE)

cat("\n--- PAIRED T-TESTS (Participant Level) ---\n")
t2 <- t.test(m1_test_nlls, dkci_test_nlls, paired=TRUE)
cat(sprintf("M1 (WSLS) vs DKCI ECCM: t = %.2f, p = %.3e (Mean Diff: %.2f)\n", t2$statistic, t2$p.value, t2$estimate))
