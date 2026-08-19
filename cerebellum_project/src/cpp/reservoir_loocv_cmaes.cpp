// [[Rcpp::depends(RcppEigen)]]
#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace Rcpp;

inline double clamp_val(double v, double lo, double hi) {
  return (v < lo) ? lo : ((v > hi) ? hi : v);
}

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
    uint32_t next() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    }
    double runif() { return (next() % 1000000) / 1000000.0; }
    double rnorm() {
        double u1 = std::max(1e-6, runif());
        double u2 = runif();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
    }
};

// [[Rcpp::export]]
double evaluate_loocv_cmaes_objective_cpp(
    const NumericVector& phi_15d,
    const IntegerVector& resp_R,
    const IntegerVector& out_R,
    const NumericVector& m1_R,
    const NumericVector& m2_R,
    const NumericVector& rt_R,
    const NumericVector& ttp_R,
    const NumericVector& ttf_R,
    const IntegerVector& subj_idx_R
) {
  int N_t = resp_R.size();
  
  double b_v         = phi_15d[0];
  double a_0         = phi_15d[1];
  double t_nd        = phi_15d[2];
  double k_mod       = phi_15d[3]; 
  double mu_beta     = phi_15d[4];
  double sigma_beta  = phi_15d[5];
  double lambda_d    = phi_15d[6];
  double mu_tau      = phi_15d[7];
  double sigma_tau   = phi_15d[8];
  double rho_base    = phi_15d[9];
  double eta         = phi_15d[10];
  double lambda      = phi_15d[11];
  
  int N_GC = 100;
  int N_MF = 20;
  
  std::vector<int> mf_c(N_MF);
  std::vector<double> mf_beta(N_MF);
  std::vector<int> mf_d(N_MF);
  
  SimpleRNG rng(42);
  for(int j = 0; j < N_MF; ++j) {
      mf_c[j] = rng.next() % 6; 
      double beta_raw = mu_beta + sigma_beta * rng.rnorm();
      mf_beta[j] = std::exp(beta_raw);
      double d_raw = lambda_d + std::sqrt(lambda_d) * rng.rnorm();
      mf_d[j] = std::max(0, std::min(10, (int)std::round(d_raw)));
  }

  std::vector<std::vector<int>> gc_mossy_map(N_GC, std::vector<int>(1));
  std::vector<std::vector<double>> gc_mossy_weights(N_GC, std::vector<double>(1, 0.0));
  for (int i = 0; i < N_GC; ++i) {
    for (int k = 0; k < 1; ++k) {
      gc_mossy_map[i][k] = rng.next() % N_MF;
      gc_mossy_weights[i][k] = (rng.rnorm() > 0) ? 1.0 : -1.0; 
    }
  }

  std::vector<double> tau_vec(N_GC, 1.0);
  for (int i = 0; i < N_GC; ++i) {
      double tau_raw = mu_tau + sigma_tau * rng.rnorm();
      tau_vec[i] = std::exp(tau_raw);
  }

  std::vector<double> z_GC_curr(N_GC, 0.0);
  std::vector<double> z_GC_prev(N_GC, 0.0);
  
  std::vector<double> w_PF1(N_GC, 0.0);
  std::vector<double> w_PF2(N_GC, 0.0);

  std::vector<std::vector<double>> state_hist(15, std::vector<double>(6, 0.0));
  
  double total_nll = 0.0;
  double rpe_abs_prev = 0.0;
  
  for (int t = 0; t < N_t; ++t) {
    if (t > 0 && subj_idx_R[t] != subj_idx_R[t - 1]) {
      std::fill(z_GC_prev.begin(), z_GC_prev.end(), 0.0);
      std::fill(z_GC_curr.begin(), z_GC_curr.end(), 0.0);
      std::fill(w_PF1.begin(), w_PF1.end(), 0.0);
      std::fill(w_PF2.begin(), w_PF2.end(), 0.0);
      for (int d = 0; d < 15; ++d) std::fill(state_hist[d].begin(), state_hist[d].end(), 0.0);
      rpe_abs_prev = 0.0;
    }

    int ch = resp_R[t];
    int out = out_R[t];
    
    int prev_ch  = (t > 0 && subj_idx_R[t] == subj_idx_R[t - 1]) ? resp_R[t - 1] : 1;
    int prev_out = (t > 0 && subj_idx_R[t] == subj_idx_R[t - 1]) ? out_R[t - 1] : 1;
    double prev_rt = (t > 0 && subj_idx_R[t] == subj_idx_R[t - 1]) ? rt_R[t - 1] : 0.75;
    double delta_t_val = (t == 0 || (t > 0 && subj_idx_R[t] != subj_idx_R[t - 1])) ? 1.5 : std::max(0.1, (double)(ttp_R[t] - ttp_R[t - 1]));

    double r_in_1_prev = (prev_ch == 1) ? prev_out : ((prev_out == 1) ? 0.0 : 1.0);
    double r_in_2_prev = (prev_ch == 2) ? prev_out : ((prev_out == 1) ? 0.0 : 1.0);
    int prev_not_choice = (prev_ch == 1) ? 2 : 1;

    for (int d = 14; d > 0; --d) state_hist[d] = state_hist[d-1];
    state_hist[0][0] = (double)prev_ch;
    state_hist[0][1] = (double)prev_out;
    state_hist[0][2] = (prev_rt - 0.75) / 0.50;
    state_hist[0][3] = r_in_1_prev;
    state_hist[0][4] = r_in_2_prev;
    state_hist[0][5] = (double)prev_not_choice;

    std::vector<double> u_MF(N_MF, 0.0);
    for(int j = 0; j < N_MF; ++j) {
        if (mf_d[j] == 0) {
            u_MF[j] = 1.0 / (1.0 + std::exp(-mf_beta[j] * state_hist[0][mf_c[j]])); 
        } else {
            int d_idx = std::min(mf_d[j], 14);
            u_MF[j] = 1.0 / (1.0 + std::exp(-mf_beta[j] * (state_hist[0][mf_c[j]] - state_hist[d_idx][mf_c[j]]))); 
        }
    }

    for (int i = 0; i < N_GC; ++i) {
        double in_sum = 0.0;
        for (int k = 0; k < 1; ++k) in_sum += gc_mossy_weights[i][k] * u_MF[gc_mossy_map[i][k]];
        
        double gamma_decay = rho_base + (1.0 - rho_base) * std::exp(-delta_t_val / tau_vec[i]);
        z_GC_curr[i] = in_sum + gamma_decay * z_GC_prev[i];
    }

    double y_PC1 = 0.0, y_PC2 = 0.0;
    for (int i = 0; i < N_GC; ++i) {
        y_PC1 += w_PF1[i] * z_GC_curr[i];
        y_PC2 += w_PF2[i] * z_GC_curr[i];
    }

    double v_t_ddm = b_v * (y_PC1 - y_PC2);
    double a_t = std::max(0.30, a_0 + k_mod * rpe_abs_prev);
    
    double rt_emp = rt_R[t];
    double dens = wiener_pdf(rt_emp, ch, v_t_ddm, a_t, t_nd);
    total_nll -= std::log(dens);

    // Fix: IO_error is the physiological Climbing Fiber spike rate (1 = Punishment, 0 = Reward)
    // Fix: Standard Signed Error for Ridge Regression Readout
    double IO_error = ((double)out - 0.5) * 2.0 - ((ch == 1) ? y_PC1 : y_PC2); // mapped out to [-1, 1] for symmetry
    rpe_abs_prev = std::abs(IO_error);

    for (int i = 0; i < N_GC; ++i) {
        if (ch == 1) {
            double delta_w = (eta / N_GC) * IO_error * z_GC_prev[i] - lambda * w_PF1[i];
            w_PF1[i] += delta_w;
        } else {
            double delta_w = (eta / N_GC) * IO_error * z_GC_prev[i] - lambda * w_PF2[i];
            w_PF2[i] += delta_w;
        }
        z_GC_prev[i] = z_GC_curr[i];
    }
  }

  if (std::isnan(total_nll) || std::isinf(total_nll)) return 1e9;
  return total_nll;
}
// [[Rcpp::export]]
double evaluate_loocv_m1_objective_cpp(
    const NumericVector& p,
    const IntegerVector& resp_R,
    const IntegerVector& out_R,
    const NumericVector& rt_R,
    const IntegerVector& subj_idx_R
) {
  double b_v = p[0], a_0 = p[1], t_nd = p[2], total_nll = 0.0;
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
    total_nll -= std::log(dens);
  }
  return total_nll;
}

// [[Rcpp::export]]
double evaluate_loocv_m2_objective_cpp(
    const NumericVector& p,
    const IntegerVector& resp_R,
    const IntegerVector& out_R,
    const NumericVector& rt_R,
    const IntegerVector& subj_idx_R
) {
  double alpha_q = p[0], b_v = p[1], a_0 = p[2], k_mod = p[3], t_nd = p[4];
  double Q_rw_cf[2] = {0.50, 0.50}, total_nll = 0.0;
  int N_t = resp_R.size();
  for (int t = 0; t < N_t; ++t) {
    if (t > 0 && subj_idx_R[t] != subj_idx_R[t-1]) { Q_rw_cf[0] = 0.50; Q_rw_cf[1] = 0.50; }
    int ch = resp_R[t], out = out_R[t];
    double q_diff = Q_rw_cf[0] - Q_rw_cf[1];
    double chosen_q = Q_rw_cf[ch - 1];
    double delta_rpe = (double)out - chosen_q;
    double rpe_abs = std::abs(delta_rpe);
    Q_rw_cf[ch - 1] += alpha_q * delta_rpe;
    int unch_idx = (ch == 1) ? 1 : 0;
    Q_rw_cf[unch_idx] += alpha_q * ((1.0 - (double)out) - Q_rw_cf[unch_idx]);
    double v_t = b_v * q_diff;
    double a_t = std::max(0.30, a_0 + k_mod * rpe_abs);
    double dens = wiener_pdf(rt_R[t], ch, v_t, a_t, t_nd);
    total_nll -= std::log(dens);
  }
  return total_nll;
}

// [[Rcpp::export]]
NumericVector get_loocv_m1_switch_prob_cpp(
    const NumericVector& p,
    const IntegerVector& resp_R,
    const IntegerVector& out_R,
    const IntegerVector& subj_idx_R
) {
  double b_v = p[0], a_0 = p[1];
  int N_t = resp_R.size();
  NumericVector p_switch(N_t);
  for (int t = 0; t < N_t; ++t) {
    int prev_ch = (t > 0 && subj_idx_R[t] == subj_idx_R[t-1]) ? resp_R[t-1] : 1;
    int prev_out = (t > 0 && subj_idx_R[t] == subj_idx_R[t-1]) ? out_R[t-1] : 1;
    int c_wsls = 1;
    if (t > 0 && subj_idx_R[t] == subj_idx_R[t-1]) c_wsls = (prev_out == 1) ? prev_ch : ((prev_ch == 1) ? 2 : 1);
    double v_t = b_v * ((c_wsls == 1) ? 1.0 : -1.0);
    double p_ch1 = 1.0 / (1.0 + std::exp(-v_t * a_0));
    p_switch[t] = (prev_ch == 1) ? (1.0 - p_ch1) : p_ch1;
  }
  return p_switch;
}

// [[Rcpp::export]]
NumericVector get_loocv_m2_switch_prob_cpp(
    const NumericVector& p,
    const IntegerVector& resp_R,
    const IntegerVector& out_R,
    const IntegerVector& subj_idx_R
) {
  double alpha_q = p[0], b_v = p[1], a_0 = p[2], k_mod = p[3];
  double Q_rw_cf[2] = {0.50, 0.50};
  int N_t = resp_R.size();
  NumericVector p_switch(N_t);
  for (int t = 0; t < N_t; ++t) {
    if (t > 0 && subj_idx_R[t] != subj_idx_R[t-1]) { Q_rw_cf[0] = 0.50; Q_rw_cf[1] = 0.50; }
    int ch = resp_R[t];
    int prev_ch = (t > 0 && subj_idx_R[t] == subj_idx_R[t-1]) ? resp_R[t-1] : 1;
    double v_t = b_v * (Q_rw_cf[0] - Q_rw_cf[1]);
    double p_ch1 = 1.0 / (1.0 + std::exp(-v_t * a_0));
    p_switch[t] = (prev_ch == 1) ? (1.0 - p_ch1) : p_ch1;
    int out = out_R[t];
    Q_rw_cf[ch - 1] += alpha_q * ((double)out - Q_rw_cf[ch - 1]);
    int unch_idx = (ch == 1) ? 1 : 0;
    Q_rw_cf[unch_idx] += alpha_q * ((1.0 - (double)out) - Q_rw_cf[unch_idx]);
  }
  return p_switch;
}

// [[Rcpp::export]]
NumericVector get_loocv_cmaes_switch_prob_cpp(
    const NumericVector& phi_15d,
    const IntegerVector& resp_R,
    const IntegerVector& out_R,
    const NumericVector& m1_R,
    const NumericVector& m2_R,
    const NumericVector& rt_R,
    const NumericVector& ttp_R,
    const NumericVector& ttf_R,
    const IntegerVector& subj_idx_R
) {
  int N_t = resp_R.size();
  NumericVector p_switch(N_t);
  
  double b_v         = phi_15d[0];
  double a_0         = phi_15d[1];
  double t_nd        = phi_15d[2];
  double k_mod       = phi_15d[3];
  double mu_beta     = phi_15d[4];
  double sigma_beta  = phi_15d[5];
  double lambda_d    = phi_15d[6];
  double mu_tau      = phi_15d[7];
  double sigma_tau   = phi_15d[8];
  double rho_base    = phi_15d[9];
  double eta         = phi_15d[10];
  double lambda      = phi_15d[11];
  
  int N_GC = 100;
  int N_MF = 20;
  
  std::vector<int> mf_c(N_MF);
  std::vector<double> mf_beta(N_MF);
  std::vector<int> mf_d(N_MF);
  
  SimpleRNG rng(42);
  for(int j = 0; j < N_MF; ++j) {
      mf_c[j] = rng.next() % 6;
      double beta_raw = mu_beta + sigma_beta * rng.rnorm();
      mf_beta[j] = std::exp(beta_raw);
      double d_raw = lambda_d + std::sqrt(lambda_d) * rng.rnorm();
      mf_d[j] = std::max(0, std::min(10, (int)std::round(d_raw)));
  }

  std::vector<std::vector<int>> gc_mossy_map(N_GC, std::vector<int>(1));
  std::vector<std::vector<double>> gc_mossy_weights(N_GC, std::vector<double>(1, 0.0));
  for (int i = 0; i < N_GC; ++i) {
    for (int k = 0; k < 1; ++k) {
      gc_mossy_map[i][k] = rng.next() % N_MF;
      gc_mossy_weights[i][k] = (rng.rnorm() > 0) ? 1.0 : -1.0; 
    }
  }

  std::vector<double> tau_vec(N_GC, 1.0);
  for (int i = 0; i < N_GC; ++i) {
      double tau_raw = mu_tau + sigma_tau * rng.rnorm();
      tau_vec[i] = std::exp(tau_raw);
  }

  std::vector<double> z_GC_curr(N_GC, 0.0);
  std::vector<double> z_GC_prev(N_GC, 0.0);
  
  std::vector<double> w_PF1(N_GC, 0.0);
  std::vector<double> w_PF2(N_GC, 0.0);

  std::vector<std::vector<double>> state_hist(15, std::vector<double>(6, 0.0));
  double rpe_abs_prev = 0.0;
  
  for (int t = 0; t < N_t; ++t) {
    if (t > 0 && subj_idx_R[t] != subj_idx_R[t - 1]) {
      std::fill(z_GC_prev.begin(), z_GC_prev.end(), 0.0);
      std::fill(z_GC_curr.begin(), z_GC_curr.end(), 0.0);
      std::fill(w_PF1.begin(), w_PF1.end(), 0.0);
      std::fill(w_PF2.begin(), w_PF2.end(), 0.0);
      for (int d = 0; d < 15; ++d) std::fill(state_hist[d].begin(), state_hist[d].end(), 0.0);
      rpe_abs_prev = 0.0;
    }

    int ch = resp_R[t];
    int out = out_R[t];
    
    int prev_ch  = (t > 0 && subj_idx_R[t] == subj_idx_R[t - 1]) ? resp_R[t - 1] : 1;
    int prev_out = (t > 0 && subj_idx_R[t] == subj_idx_R[t - 1]) ? out_R[t - 1] : 1;
    double prev_rt = (t > 0 && subj_idx_R[t] == subj_idx_R[t - 1]) ? rt_R[t - 1] : 0.75;
    double delta_t_val = (t == 0 || (t > 0 && subj_idx_R[t] != subj_idx_R[t - 1])) ? 1.5 : std::max(0.1, (double)(ttp_R[t] - ttp_R[t - 1]));

    double r_in_1_prev = (prev_ch == 1) ? prev_out : ((prev_out == 1) ? 0.0 : 1.0);
    double r_in_2_prev = (prev_ch == 2) ? prev_out : ((prev_out == 1) ? 0.0 : 1.0);
    int prev_not_choice = (prev_ch == 1) ? 2 : 1;

    for (int d = 14; d > 0; --d) state_hist[d] = state_hist[d-1];
    state_hist[0][0] = (double)prev_ch;
    state_hist[0][1] = (double)prev_out;
    state_hist[0][2] = (prev_rt - 0.75) / 0.50;
    state_hist[0][3] = r_in_1_prev;
    state_hist[0][4] = r_in_2_prev;
    state_hist[0][5] = (double)prev_not_choice;

    std::vector<double> u_MF(N_MF, 0.0);
    for(int j = 0; j < N_MF; ++j) {
        if (mf_d[j] == 0) {
            u_MF[j] = 1.0 / (1.0 + std::exp(-mf_beta[j] * state_hist[0][mf_c[j]]));
        } else {
            int d_idx = std::min(mf_d[j], 14);
            u_MF[j] = 1.0 / (1.0 + std::exp(-mf_beta[j] * (state_hist[0][mf_c[j]] - state_hist[d_idx][mf_c[j]])));
        }
    }

    for (int i = 0; i < N_GC; ++i) {
        double in_sum = 0.0;
        for (int k = 0; k < 1; ++k) in_sum += gc_mossy_weights[i][k] * u_MF[gc_mossy_map[i][k]];
        
        double gamma_decay = rho_base + (1.0 - rho_base) * std::exp(-delta_t_val / tau_vec[i]);
        z_GC_curr[i] = in_sum + gamma_decay * z_GC_prev[i];
    }

    double y_PC1 = 0.0, y_PC2 = 0.0;
    for (int i = 0; i < N_GC; ++i) {
        y_PC1 += w_PF1[i] * z_GC_curr[i];
        y_PC2 += w_PF2[i] * z_GC_curr[i];
    }

    double v_t_ddm = b_v * (y_PC1 - y_PC2);
    double a_t = std::max(0.30, a_0 + k_mod * rpe_abs_prev);
    
    double p_ch1 = 1.0 / (1.0 + std::exp(-v_t_ddm * a_t));
    p_switch[t] = (prev_ch == 1) ? (1.0 - p_ch1) : p_ch1;

    // Fix: IO_error is the physiological Climbing Fiber spike rate (1 = Punishment, 0 = Reward)
    // Fix: Standard Signed Error for Ridge Regression Readout
    double IO_error = ((double)out - 0.5) * 2.0 - ((ch == 1) ? y_PC1 : y_PC2); // mapped out to [-1, 1] for symmetry
    rpe_abs_prev = std::abs(IO_error);

    for (int i = 0; i < N_GC; ++i) {
        if (ch == 1) {
            double delta_w = (eta / N_GC) * IO_error * z_GC_prev[i] - lambda * w_PF1[i];
            w_PF1[i] += delta_w;
        } else {
            double delta_w = (eta / N_GC) * IO_error * z_GC_prev[i] - lambda * w_PF2[i];
            w_PF2[i] += delta_w;
        }
        z_GC_prev[i] = z_GC_curr[i];
    }
  }

  return p_switch;
}
