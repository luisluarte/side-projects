library(Rcpp)

dataset_path <- "data/raw/behavioral_compilate.csv"
if (!file.exists(dataset_path)) {
  dataset_path <- "c:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv"
}

dat_all <- read.csv(dataset_path)
dat_all[['RT']] <- (as.numeric(dat_all[['ttr']]) - as.numeric(dat_all[['ttp']])) / 1000.0
dat_all <- dat_all[!is.na(dat_all[['RT']]) & dat_all[['RT']] >= 0.1 & dat_all[['RT']] <= 3.0 & dat_all[['Resp']] %in% c(1, 2), ]

# Use the EXACT same 10 participants and 70/30 split as ECCM
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

// [[Rcpp::export]]
List evaluate_m1_70_30_cpp(
    const NumericVector& p,
    const IntegerVector& resp_R,
    const IntegerVector& out_R,
    const NumericVector& rt_R,
    const IntegerVector& subj_idx_R,
    const IntegerVector& is_test_R,
    bool return_test_metrics = false
) {
  double b_v = p[0], a_0 = p[1], t_nd = p[2];
  double train_nll = 0.0, test_nll = 0.0;
  int test_count = 0;
  int N_t = resp_R.size();
  
  for (int t = 0; t < N_t; ++t) {
    int ch = resp_R[t];
    int prev_ch = (t > 0 && subj_idx_R[t] == subj_idx_R[t-1]) ? resp_R[t-1] : 1;
    int prev_out = (t > 0 && subj_idx_R[t] == subj_idx_R[t-1]) ? out_R[t-1] : 1;
    int c_wsls = 1;
    if (t > 0 && subj_idx_R[t] == subj_idx_R[t-1]) c_wsls = (prev_out == 1) ? prev_ch : ((prev_ch == 1) ? 2 : 1);
    
    double wsls_signal = (c_wsls == 1) ? 1.0 : -1.0;
    double v_t = b_v * wsls_signal;
    double dens = wiener_pdf(rt_R[t], ch, v_t, a_0, t_nd);
    
    if (is_test_R[t] == 1) {
        test_nll -= std::log(dens);
        test_count++;
    } else {
        train_nll -= std::log(dens);
    }
  }
  
  if (return_test_metrics) {
      return List::create(Named("Train_NLL") = train_nll, Named("Test_NLL") = test_nll, Named("Test_Count") = test_count);
  } else {
      if (std::isnan(train_nll) || std::isinf(train_nll)) return List::create(Named("Obj") = 1e9);
      return List::create(Named("Obj") = train_nll);
  }
}

// [[Rcpp::export]]
List evaluate_m2_70_30_cpp(
    const NumericVector& p,
    const IntegerVector& resp_R,
    const IntegerVector& out_R,
    const NumericVector& rt_R,
    const IntegerVector& subj_idx_R,
    const IntegerVector& is_test_R,
    bool return_test_metrics = false
) {
  double alpha_q = p[0], b_v = p[1], a_0 = p[2], k_mod = p[3], t_nd = p[4];
  double Q_rw_cf[2] = {0.50, 0.50};
  double train_nll = 0.0, test_nll = 0.0;
  int test_count = 0;
  int N_t = resp_R.size();
  
  for (int t = 0; t < N_t; ++t) {
    if (t > 0 && subj_idx_R[t] != subj_idx_R[t-1]) { Q_rw_cf[0] = 0.50; Q_rw_cf[1] = 0.50; }
    
    int ch = resp_R[t], out = out_R[t];
    double q_diff = Q_rw_cf[0] - Q_rw_cf[1];
    
    double chosen_q = Q_rw_cf[ch - 1];
    double delta_rpe = (double)out - chosen_q;
    double rpe_abs = std::abs(delta_rpe);
    
    double v_t = b_v * q_diff;
    double a_t = std::max(0.30, a_0 + k_mod * rpe_abs);
    double dens = wiener_pdf(rt_R[t], ch, v_t, a_t, t_nd);
    
    if (is_test_R[t] == 1) {
        test_nll -= std::log(dens);
        test_count++;
    } else {
        train_nll -= std::log(dens);
    }
    
    double current_alpha = (is_test_R[t] == 1) ? 0.0 : alpha_q;
    
    Q_rw_cf[ch - 1] += current_alpha * delta_rpe;
    int unch_idx = (ch == 1) ? 1 : 0;
    Q_rw_cf[unch_idx] += current_alpha * ((1.0 - (double)out) - Q_rw_cf[unch_idx]);
  }
  
  if (return_test_metrics) {
      return List::create(Named("Train_NLL") = train_nll, Named("Test_NLL") = test_nll, Named("Test_Count") = test_count);
  } else {
      if (std::isnan(train_nll) || std::isinf(train_nll)) return List::create(Named("Obj") = 1e9);
      return List::create(Named("Obj") = train_nll);
  }
}
'
sourceCpp(code = cpp_code)

# --- M1 Optimization ---
m1_lower <- c(0.01, 0.40, 0.08)
m1_upper <- c(3.50, 2.50, 0.30)
m1_init <- c(0.80, 1.20, 0.18)

obj_m1 <- function(p) {
    if(any(p<m1_lower)||any(p>m1_upper)) return(1e9)
    res <- evaluate_m1_70_30_cpp(p, as.integer(dat_all$Resp), as.integer(dat_all$F), 
                                 dat_all$RT, as.integer(as.factor(dat_all$participant_id)), dat_all$is_test, FALSE)
    return(res$Obj)
}
m1_res <- optim(m1_init, obj_m1, method="L-BFGS-B", lower=m1_lower, upper=m1_upper, control=list(maxit=30))
m1_opt <- m1_res$par

final_m1 <- evaluate_m1_70_30_cpp(m1_opt, as.integer(dat_all$Resp), as.integer(dat_all$F), 
                                  dat_all$RT, as.integer(as.factor(dat_all$participant_id)), dat_all$is_test, TRUE)

cat(sprintf("M1 WSLS Baseline -> Train NLL: %.2f | Test NLL: %.2f\n", final_m1$Train_NLL, final_m1$Test_NLL))

# --- M2 Optimization ---
m2_lower <- c(0.01, 0.01, 0.40, 0.00, 0.08)
m2_upper <- c(0.90, 3.50, 2.50, 0.50, 0.30)
m2_init <- c(0.15, 0.80, 1.20, 0.05, 0.18)

obj_m2 <- function(p) {
    if(any(p<m2_lower)||any(p>m2_upper)) return(1e9)
    res <- evaluate_m2_70_30_cpp(p, as.integer(dat_all$Resp), as.integer(dat_all$F), 
                                 dat_all$RT, as.integer(as.factor(dat_all$participant_id)), dat_all$is_test, FALSE)
    return(res$Obj)
}
m2_res <- optim(m2_init, obj_m2, method="L-BFGS-B", lower=m2_lower, upper=m2_upper, control=list(maxit=30))
m2_opt <- m2_res$par

final_m2 <- evaluate_m2_70_30_cpp(m2_opt, as.integer(dat_all$Resp), as.integer(dat_all$F), 
                                  dat_all$RT, as.integer(as.factor(dat_all$participant_id)), dat_all$is_test, TRUE)

cat(sprintf("M2 RWCF Baseline -> Train NLL: %.2f | Test NLL: %.2f\n", final_m2$Train_NLL, final_m2$Test_NLL))
