// [[Rcpp::depends(RcppEigen)]]
#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace Rcpp;

// ==============================================================================
// HIGH-SPEED SYMPLECTIC HDDM ENGINE
// 1. One-time 1,000-D Symplectic Reservoir Forward Pass
// 2. Microsecond Wiener First-Passage Density Evaluator for Fast Optimization
// ==============================================================================

inline double clamp_val(double v, double lo, double hi) {
  return (v < lo) ? lo : ((v > hi) ? hi : v);
}

// Navarro & Fuss (2009) fast series approximation for Wiener first-passage density
inline double wiener_pdf(double t_raw, int choice, double v, double a, double t_nd, double eps = 1e-9) {
  double t = t_raw - t_nd;
  if (t <= 0.001) return 1e-12;
  
  double sign = (choice == 1) ? 1.0 : -1.0;
  double w = 0.50; // Unbiased starting point
  double x0 = (choice == 1) ? (1.0 - w) : w; // Distance to boundary
  
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
List simulate_symplectic_core_cpp(
    const IntegerVector& resp_R,
    const NumericVector& out_R,
    const NumericVector& m1_R,
    const NumericVector& m2_R,
    const NumericVector& rt_R,
    const NumericVector& ttp_R,
    const NumericVector& params_R,
    int N_GC = 1000
) {
  int N_t = resp_R.size();
  
  double p_ws_base       = params_R[0];
  double p_ls_base       = params_R[1];
  double w_mag_curr      = params_R[2];
  double w_mag_alt       = params_R[3];
  double alpha_q         = params_R[4];
  double w_streak        = params_R[5];
  double w_purkinje_inh  = params_R[6];
  double tau_kinematic   = params_R[7];
  double beta_post_err   = params_R[8];
  double kappa_entropy   = params_R[9];

  int N_GoC = std::max(5, N_GC / 4);
  int N_actions = 2;

  // 4D MOSSY FIBER INPUT: [prev_ch, prev_out, d_curr, d_diff]
  int mossy_fan_in = 4;
  std::vector<int> gc_mossy_idx(N_GC);
  std::vector<double> gc_mossy_w(N_GC, 0.35);
  for (int i = 0; i < N_GC; ++i) {
    gc_mossy_idx[i] = i % mossy_fan_in;
  }

  std::vector<std::vector<int>> goc_gc_indices(N_GoC, std::vector<int>(4));
  for (int j = 0; j < N_GoC; ++j) {
    for (int k = 0; k < 4; ++k) {
      goc_gc_indices[j][k] = (j * 4 + k) % N_GC;
    }
  }

  std::vector<double> z_GC_prev(N_GC, 0.0);
  std::vector<double> z_GC_curr(N_GC, 0.0);
  std::vector<double> z_GoC(N_GoC, 0.0);

  std::vector<double> rho_vec(N_GC, 0.70);
  std::vector<double> tau_vec(N_GC, std::max(0.001, tau_kinematic));

  std::vector<double> w_v(N_GC, 0.10);
  std::vector<std::vector<double>> W_pi(N_actions, std::vector<double>(N_GC, 0.10));
  double b_v = 0.50;

  double V_prev = 0.0;
  double Q_val[2] = {0.50, 0.50};
  int loss_streak = 0;

  NumericVector v_raw_vec(N_t);
  NumericVector spatial_entropy_vec(N_t);
  NumericVector is_pause_vec(N_t);

  for (int t = 0; t < N_t; ++t) {
    int ch = resp_R[t];
    int out = out_R[t];
    double m1 = m1_R[t];
    double m2 = m2_R[t];
    double delta_t = (t == 0) ? 1.5 : std::max(0.1, (double)(ttp_R[t] - ttp_R[t - 1]));
    is_pause_vec[t] = (delta_t >= 10.0) ? 1.0 : 0.0;

    int prev_ch  = (t > 0) ? resp_R[t - 1] : 1;
    int prev_out = (t > 0) ? out_R[t - 1] : 1;

    double m_curr = (prev_ch == 1) ? m1 : m2;
    double m_alt  = (prev_ch == 1) ? m2 : m1;
    double d_curr = (m_curr - 5.5) / 4.5;
    double d_diff = (m_alt - m_curr) / 4.0;
    double q_diff = Q_val[prev_ch - 1] - Q_val[2 - prev_ch];

    double u_arr[4] = {
      (prev_ch == 1) ? 1.0 : -1.0,
      (double)prev_out,
      d_curr,
      d_diff
    };

    // Granule cell forward integration
    std::vector<double> h_pre(N_GC, 0.0);
    for (int i = 0; i < N_GC; ++i) {
      double input_i = gc_mossy_w[i] * u_arr[gc_mossy_idx[i]];
      double gamma_i = rho_vec[i] + (1.0 - rho_vec[i]) * std::exp(-delta_t / tau_vec[i]);
      h_pre[i] = std::tanh(input_i + gamma_i * z_GC_prev[i]);
    }

    // Golgi recurrent integration
    for (int j = 0; j < N_GoC; ++j) {
      double exc = 0.15 * V_prev;
      for (int k = 0; k < 4; ++k) {
        exc += 0.25 * h_pre[goc_gc_indices[j][k]];
      }
      z_GoC[j] = std::max(0.0, exc);
    }

    // Granule cell state with Golgi inhibition
    for (int i = 0; i < N_GC; ++i) {
      double inh = w_purkinje_inh * z_GoC[i % N_GoC];
      z_GC_curr[i] = std::max(0.0, h_pre[i] - inh);
    }

    // Granular Spatial Entropy S_t
    double l1_sum = 1e-12;
    for (int i = 0; i < N_GC; ++i) l1_sum += std::abs(z_GC_curr[i]);
    double S_t = 0.0;
    for (int i = 0; i < N_GC; ++i) {
      double p_i = std::abs(z_GC_curr[i]) / l1_sum;
      if (p_i > 1e-12) S_t -= p_i * std::log(p_i);
    }
    spatial_entropy_vec[t] = S_t;

    // Value estimation
    double V_curr = b_v;
    for (int i = 0; i < N_GC; ++i) V_curr += w_v[i] * z_GC_curr[i];

    // Choice policy differential (Purkinje readout)
    double gc_diff = 0.0;
    for (int i = 0; i < N_GC; ++i) {
      gc_diff += (W_pi[0][i] - W_pi[1][i]) * z_GC_curr[i];
    }

    double logit_bias = (prev_out == 1) ? (w_mag_curr * d_curr + 0.35 * q_diff) : (w_mag_alt * d_diff - 0.35 * q_diff);
    v_raw_vec[t] = logit_bias + 0.25 * gc_diff;

    // Plasticity update
    double lr_scale = 40.0 / (double)N_GC;
    double reward = (double)out;
    double delta_rpe = reward - V_curr;
    double Omega_t = std::exp(-kappa_entropy * S_t);

    for (int i = 0; i < N_GC; ++i) {
      double kick_v = clamp_val(0.05 * lr_scale * Omega_t * delta_rpe * z_GC_curr[i], -1.0, 1.0);
      w_v[i] = w_v[i] * std::exp(kick_v);

      int a_idx = (ch == 1) ? 0 : 1;
      double kick_pi = clamp_val(0.05 * lr_scale * Omega_t * delta_rpe * 0.50 * z_GC_curr[i], -1.0, 1.0);
      W_pi[a_idx][i] = W_pi[a_idx][i] * std::exp(kick_pi);
    }
    b_v = b_v * std::exp(clamp_val(0.02 * Omega_t * delta_rpe, -0.2, 0.2));

    int chosen_idx = ch - 1;
    Q_val[chosen_idx] = Q_val[chosen_idx] + alpha_q * (reward - Q_val[chosen_idx]);

    z_GC_prev = z_GC_curr;
    V_prev = V_curr;
  }

  return List::create(
    Named("V_Raw_Vec") = v_raw_vec,
    Named("Spatial_Entropy_Vec") = spatial_entropy_vec,
    Named("Is_Pause_Vec") = is_pause_vec
  );
}

// [[Rcpp::export]]
double compute_wiener_loglik_fast_cpp(
    const IntegerVector& resp_R,
    const NumericVector& rt_R,
    const NumericVector& v_raw_R,
    const NumericVector& S_t_R,
    double beta_v,
    double a_0,
    double kappa_a,
    double t_nd
) {
  int N_t = resp_R.size();
  double total_ll = 0.0;

  for (int t = 0; t < N_t; ++t) {
    int ch = resp_R[t];
    double rt_emp = rt_R[t];
    double v_t = beta_v * v_raw_R[t];
    double a_t = std::max(0.30, a_0 + kappa_a * S_t_R[t]);

    double dens = wiener_pdf(rt_emp, ch, v_t, a_t, t_nd);
    total_ll += std::log(dens);
  }
  return total_ll;
}

// [[Rcpp::export]]
List evaluate_hddm_predictions_cpp(
    const IntegerVector& resp_R,
    const NumericVector& rt_R,
    const NumericVector& v_raw_R,
    const NumericVector& S_t_R,
    double beta_v,
    double a_0,
    double kappa_a,
    double t_nd
) {
  int N_t = resp_R.size();
  NumericVector pred_mean_rt_vec(N_t);
  NumericVector pred_choice_prob_vec(N_t);
  NumericVector loglik_vec(N_t);
  double total_ll = 0.0;

  for (int t = 0; t < N_t; ++t) {
    int ch = resp_R[t];
    double rt_emp = rt_R[t];
    double v_t = beta_v * v_raw_R[t];
    double a_t = std::max(0.30, a_0 + kappa_a * S_t_R[t]);

    double dens = wiener_pdf(rt_emp, ch, v_t, a_t, t_nd);
    double ll = std::log(dens);
    loglik_vec[t] = ll;
    total_ll += ll;

    double p_ch1 = 1.0 / (1.0 + std::exp(-v_t * a_t));
    pred_choice_prob_vec[t] = p_ch1;

    double expected_dt = (std::abs(v_t) > 1e-4) ? (a_t / (2.0 * v_t)) * std::tanh(v_t * a_t / 2.0) : (a_t * a_t / 4.0);
    pred_mean_rt_vec[t] = t_nd + std::max(0.05, expected_dt);
  }

  return List::create(
    Named("Total_LogLik") = total_ll,
    Named("LogLik_Vec") = loglik_vec,
    Named("Pred_Mean_RT") = pred_mean_rt_vec,
    Named("Pred_Choice_Prob") = pred_choice_prob_vec
  );
}
