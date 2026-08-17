// [[Rcpp::depends(RcppEigen)]]
#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace Rcpp;

// ==============================================================================
// TEMPORAL TOPOLOGICALLY-GATED SYMPLECTIC-HDDM TOURNAMENT ENGINE
// Model 0: Intercept-Only DDM (Null Baseline)
// Model 1: WSLS-HDDM (Markovian Heuristic)
// Model 2: RW-CF-HDDM (Counterfactual Value Tracker + |RPE| Boundary)
// Model 3: Kernelized Symplectic-HDDM (1,000-D Reservoir + 4D MF + Static Rebound)
// Model 4: Temporal Topologically-Gated Symplectic-HDDM (10D Temporal MF + Persistent Entropy + LC Smoothing + Dynamic Gating)
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
List precompute_temporal_topological_tournament_cpp(
    const IntegerVector& resp_R,
    const NumericVector& out_R,
    const NumericVector& m1_R,
    const NumericVector& m2_R,
    const NumericVector& rt_R,
    const NumericVector& ttp_R,
    const NumericVector& ttf_R,
    const NumericVector& params_R,
    int N_GC = 1000,
    int N_MLI = 200
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

  // 10-D TEMPORAL MOSSY FIBER FAN-IN MAPPINGS
  // Inputs: [prev_ch, prev_out, d_curr, d_diff, prev_rt, prev_ttf, iti, streak, reward_rate, urgency]
  int mossy_dim = 10;
  std::vector<std::vector<int>> gc_mossy_map(N_GC, std::vector<int>(4));
  std::vector<std::vector<double>> gc_mossy_weights(N_GC, std::vector<double>(4, 0.35));
  for (int i = 0; i < N_GC; ++i) {
    for (int k = 0; k < 4; ++k) {
      gc_mossy_map[i][k] = (i * 3 + k * 7) % mossy_dim;
      gc_mossy_weights[i][k] = 0.25 + 0.10 * std::sin((double)(i + k * 13));
    }
  }

  // 4D MOSSY FIBER FAN-IN (for Model 3 baseline)
  int mossy_fan_in_4d = 4;
  std::vector<int> gc_mossy_idx_4d(N_GC);
  for (int i = 0; i < N_GC; ++i) gc_mossy_idx_4d[i] = i % mossy_fan_in_4d;

  std::vector<std::vector<int>> goc_gc_indices(N_GoC, std::vector<int>(4));
  for (int j = 0; j < N_GoC; ++j) {
    for (int k = 0; k < 4; ++k) goc_gc_indices[j][k] = (j * 4 + k) % N_GC;
  }

  // STATIC MLI PROJECTION MATRIX (15% sparse fan-in from Granule Cells)
  int mli_fan_in = 150;
  std::vector<std::vector<int>> mli_gc_indices(N_MLI, std::vector<int>(mli_fan_in));
  for (int m = 0; m < N_MLI; ++m) {
    for (int k = 0; k < mli_fan_in; ++k) {
      mli_gc_indices[m][k] = (m * 17 + k * 7) % N_GC;
    }
  }

  // Model 3 State Variables
  std::vector<double> z_GC_prev_m3(N_GC, 0.0);
  std::vector<double> z_GC_curr_m3(N_GC, 0.0);
  std::vector<double> z_GoC_m3(N_GoC, 0.0);
  std::vector<double> h_MLI_m3(N_MLI, 0.0);
  std::vector<double> w_v_m3(N_GC, 0.10);
  std::vector<std::vector<double>> W_pi_m3(N_actions, std::vector<double>(N_GC, 0.10));
  std::vector<std::vector<double>> W_inh_m3(N_actions, std::vector<double>(N_MLI, 0.05));
  double b_v_m3 = 0.50;
  double V_prev_m3 = 0.0;
  double Q_val_m3[2] = {0.50, 0.50};

  // Model 4 State Variables (10D Temporal Reservoir)
  std::vector<double> z_GC_prev_m4(N_GC, 0.0);
  std::vector<double> z_GC_curr_m4(N_GC, 0.0);
  std::vector<double> z_GoC_m4(N_GoC, 0.0);
  std::vector<double> h_MLI_m4(N_MLI, 0.0);
  std::vector<double> w_v_m4(N_GC, 0.10);
  std::vector<std::vector<double>> W_pi_m4(N_actions, std::vector<double>(N_GC, 0.10));
  std::vector<std::vector<double>> W_inh_m4(N_actions, std::vector<double>(N_MLI, 0.05));
  double b_v_m4 = 0.50;
  double V_prev_m4 = 0.0;
  double Q_val_m4[2] = {0.50, 0.50};

  std::vector<double> rho_vec(N_GC, 0.70);
  std::vector<double> tau_vec(N_GC, std::max(0.001, tau_kinematic));

  double Q_rw_cf[2] = {0.50, 0.50};

  NumericVector v_m0_vec(N_t, 1.0); // Model 0: Intercept only
  NumericVector v_m1_vec(N_t);      // Model 1: WSLS
  NumericVector v_m2_vec(N_t);      // Model 2: RW-CF
  NumericVector rpe_m2_vec(N_t);    // Model 2: |RPE|
  NumericVector v_m3_vec(N_t);      // Model 3: Kernelized Symplectic Log-odds
  NumericVector rebound_m3_vec(N_t);// Model 3: DCN Rebound Brake
  
  // Model 4: Temporal Topologically-Gated Symplectic-HDDM
  NumericVector omega_m4_vec(N_t);   // Filtered Persistent Entropy Omega_t
  NumericVector n_eff_m4_vec(N_t);   // Effective Noradrenaline Surge N_{t, eff}
  NumericVector v_delib_m4_vec(N_t); // Deliberative drift component
  NumericVector v_heur_m4_vec(N_t);  // Heuristic drift component
  NumericVector lc_weight_vec(N_t);  // LC Salience Weight

  double max_entropy = std::log((double)N_GC);
  double N_eff_state = 0.0;
  double tau_NA = 0.50; // Noradrenergic reuptake time constant

  // Sliding window of state history for capacity tracking
  int window_size = 4;
  std::vector<std::vector<double>> history_GC_m4;

  double running_streak = 0.0;
  double running_reward_rate = 0.50;

  for (int t = 0; t < N_t; ++t) {
    int ch = resp_R[t];
    int out = out_R[t];
    double m1 = m1_R[t];
    double m2 = m2_R[t];
    double rt_t = rt_R[t];
    double ttp_t = ttp_R[t];
    double ttf_t = ttf_R[t];

    double delta_t = (t == 0) ? 1.5 : std::max(0.1, (double)(ttp_R[t] - ttp_R[t - 1]));

    int prev_ch  = (t > 0) ? resp_R[t - 1] : 1;
    int prev_out = (t > 0) ? out_R[t - 1] : 1;
    double prev_rt = (t > 0) ? rt_R[t - 1] : 0.75;
    double prev_ttf = (t > 0) ? (ttf_R[t - 1] - rt_R[t - 1]) : 2.0;
    double prev_iti = (t > 0) ? (ttp_R[t] - ttf_R[t - 1]) : 7.0;

    // Update streak and local reward rate
    if (t > 0) {
      if (prev_out == 1) {
        running_streak = (running_streak >= 0.0) ? (running_streak + 1.0) : 1.0;
      } else {
        running_streak = (running_streak <= 0.0) ? (running_streak - 1.0) : -1.0;
      }
      running_reward_rate = 0.80 * running_reward_rate + 0.20 * (double)prev_out;
    }

    double m_curr = (prev_ch == 1) ? m1 : m2;
    double m_alt  = (prev_ch == 1) ? m2 : m1;
    double d_curr = (m_curr - 5.5) / 4.5;
    double d_diff = (m_alt - m_curr) / 4.0;

    // --- MODEL 1: WSLS DETERMINISTIC HEURISTIC ---
    int c_wsls = 1;
    if (t > 0) {
      if (prev_out == 1) {
        c_wsls = prev_ch; // Win-Stay
      } else {
        c_wsls = (prev_ch == 1) ? 2 : 1; // Lose-Shift
      }
    }
    double wsls_signal = (c_wsls == 1) ? 1.0 : -1.0;
    v_m1_vec[t] = wsls_signal;

    // --- MODEL 2: RW-CF COUNTERFACTUAL LEARNING ---
    double q_diff_rw = Q_rw_cf[0] - Q_rw_cf[1];
    v_m2_vec[t] = q_diff_rw;
    double chosen_q = Q_rw_cf[ch - 1];
    double delta_rpe_rw = (double)out - chosen_q;
    rpe_m2_vec[t] = std::abs(delta_rpe_rw);

    // Counterfactual update (alpha = 0.15)
    double alpha_rw = 0.15;
    Q_rw_cf[ch - 1] += alpha_rw * delta_rpe_rw;
    int unch_idx = (ch == 1) ? 1 : 0;
    double unch_reward = 1.0 - (double)out;
    Q_rw_cf[unch_idx] += alpha_rw * (unch_reward - Q_rw_cf[unch_idx]);

    // --- MODEL 3: 4D MOSSY FIBER RESERVOIR ---
    double u_arr_4d[4] = {
      (prev_ch == 1) ? 1.0 : -1.0,
      (double)prev_out,
      d_curr,
      d_diff
    };
    std::vector<double> h_pre_m3(N_GC, 0.0);
    for (int i = 0; i < N_GC; ++i) {
      double input_i = 0.35 * u_arr_4d[gc_mossy_idx_4d[i]];
      double gamma_i = rho_vec[i] + (1.0 - rho_vec[i]) * std::exp(-delta_t / tau_vec[i]);
      h_pre_m3[i] = std::tanh(input_i + gamma_i * z_GC_prev_m3[i]);
    }
    for (int j = 0; j < N_GoC; ++j) {
      double exc = 0.15 * V_prev_m3;
      for (int k = 0; k < 4; ++k) exc += 0.25 * h_pre_m3[goc_gc_indices[j][k]];
      z_GoC_m3[j] = std::max(0.0, exc);
    }
    for (int i = 0; i < N_GC; ++i) {
      double inh = w_purkinje_inh * z_GoC_m3[i % N_GoC];
      z_GC_curr_m3[i] = std::max(0.0, h_pre_m3[i] - inh);
    }
    for (int m = 0; m < N_MLI; ++m) {
      double mli_drive = 0.0;
      for (int k = 0; k < mli_fan_in; ++k) mli_drive += z_GC_curr_m3[mli_gc_indices[m][k]];
      mli_drive /= (double)mli_fan_in;
      h_MLI_m3[m] = std::max(0.0, mli_drive - 0.05);
    }
    double l1_sum_m3 = 1e-12;
    for (int i = 0; i < N_GC; ++i) l1_sum_m3 += std::abs(z_GC_curr_m3[i]);
    double S_t_m3 = 0.0;
    for (int i = 0; i < N_GC; ++i) {
      double p_i = std::abs(z_GC_curr_m3[i]) / l1_sum_m3;
      if (p_i > 1e-12) S_t_m3 -= p_i * std::log(p_i);
    }
    double norm_entropy_m3 = std::min(1.0, S_t_m3 / max_entropy);
    rebound_m3_vec[t] = 2.85 * std::pow(norm_entropy_m3, 1.75);

    double gc_policy_diff_m3 = 0.0;
    for (int i = 0; i < N_GC; ++i) gc_policy_diff_m3 += (W_pi_m3[0][i] - W_pi_m3[1][i]) * z_GC_curr_m3[i];
    double mli_policy_diff_m3 = 0.0;
    for (int m = 0; m < N_MLI; ++m) mli_policy_diff_m3 += (W_inh_m3[0][m] - W_inh_m3[1][m]) * h_MLI_m3[m];

    double s_ch = (prev_ch == 1) ? 1.0 : -1.0;
    double logit_bias = 0.0;
    if (prev_out == 1) {
      logit_bias = s_ch * (1.0 + w_mag_curr * d_curr);
    } else {
      logit_bias = -s_ch * (1.0 + w_mag_alt * d_diff);
    }
    double q_diff_m3 = Q_val_m3[0] - Q_val_m3[1];
    v_m3_vec[t] = logit_bias + 0.45 * q_diff_m3 + 0.20 * (gc_policy_diff_m3 - mli_policy_diff_m3);

    // Update Model 3 Plasticity
    double V_curr_m3 = b_v_m3;
    for (int i = 0; i < N_GC; ++i) V_curr_m3 += w_v_m3[i] * z_GC_curr_m3[i];
    double delta_rpe_m3 = (double)out - V_curr_m3;
    double Omega_t_m3 = std::exp(-kappa_entropy * S_t_m3);
    double p_ch1_m3 = 1.0 / (1.0 + std::exp(-v_m3_vec[t]));
    double p_chosen_m3 = (ch == 1) ? p_ch1_m3 : (1.0 - p_ch1_m3);
    p_chosen_m3 = clamp_val(p_chosen_m3, 1e-6, 1.0 - 1e-6);

    double lr_scale = 40.0 / (double)N_GC;
    for (int i = 0; i < N_GC; ++i) {
      w_v_m3[i] *= std::exp(clamp_val(0.05 * lr_scale * Omega_t_m3 * delta_rpe_m3 * z_GC_curr_m3[i], -1.0, 1.0));
      int a_idx = (ch == 1) ? 0 : 1;
      W_pi_m3[a_idx][i] *= std::exp(clamp_val(0.05 * lr_scale * Omega_t_m3 * delta_rpe_m3 * (1.0 - p_chosen_m3) * z_GC_curr_m3[i], -1.0, 1.0));
    }
    for (int m = 0; m < N_MLI; ++m) {
      int a_idx = (ch == 1) ? 0 : 1;
      W_inh_m3[a_idx][m] *= std::exp(clamp_val(0.02 * lr_scale * Omega_t_m3 * delta_rpe_m3 * (1.0 - p_chosen_m3) * h_MLI_m3[m], -1.0, 1.0));
    }
    b_v_m3 *= std::exp(clamp_val(0.02 * Omega_t_m3 * delta_rpe_m3, -0.2, 0.2));
    Q_val_m3[ch - 1] += alpha_q * ((double)out - Q_val_m3[ch - 1]);
    z_GC_prev_m3 = z_GC_curr_m3;
    V_prev_m3 = V_curr_m3;

    // --- MODEL 4: 10-D TEMPORAL MOSSY FIBER RESERVOIR ---
    double u_arr_10d[10] = {
      (prev_ch == 1) ? 1.0 : -1.0,
      (double)prev_out,
      d_curr,
      d_diff,
      clamp_val((prev_rt - 0.75) / 0.50, -2.0, 2.0),
      clamp_val((prev_ttf - 2.00) / 1.00, -2.0, 2.0),
      clamp_val((prev_iti - 7.00) / 3.00, -2.0, 2.0),
      clamp_val(running_streak / 4.0, -2.0, 2.0),
      clamp_val((running_reward_rate - 0.50) / 0.30, -2.0, 2.0),
      clamp_val(std::log(1.0 + prev_iti) - 1.5, -2.0, 2.0)
    };

    std::vector<double> h_pre_m4(N_GC, 0.0);
    for (int i = 0; i < N_GC; ++i) {
      double input_i = 0.0;
      for (int k = 0; k < 4; ++k) {
        input_i += gc_mossy_weights[i][k] * u_arr_10d[gc_mossy_map[i][k]];
      }
      double gamma_i = rho_vec[i] + (1.0 - rho_vec[i]) * std::exp(-delta_t / tau_vec[i]);
      h_pre_m4[i] = std::tanh(input_i + gamma_i * z_GC_prev_m4[i]);
    }

    for (int j = 0; j < N_GoC; ++j) {
      double exc = 0.15 * V_prev_m4;
      for (int k = 0; k < 4; ++k) exc += 0.25 * h_pre_m4[goc_gc_indices[j][k]];
      z_GoC_m4[j] = std::max(0.0, exc);
    }

    for (int i = 0; i < N_GC; ++i) {
      double inh = w_purkinje_inh * z_GoC_m4[i % N_GoC];
      z_GC_curr_m4[i] = std::max(0.0, h_pre_m4[i] - inh);
    }

    for (int m = 0; m < N_MLI; ++m) {
      double mli_drive = 0.0;
      for (int k = 0; k < mli_fan_in; ++k) mli_drive += z_GC_curr_m4[mli_gc_indices[m][k]];
      mli_drive /= (double)mli_fan_in;
      h_MLI_m4[m] = std::max(0.0, mli_drive - 0.05);
    }

    // Spatial Entropy and Symplectic Persistence on 10D Manifold
    double l1_sum_m4 = 1e-12;
    for (int i = 0; i < N_GC; ++i) l1_sum_m4 += std::abs(z_GC_curr_m4[i]);
    double S_t_m4 = 0.0;
    for (int i = 0; i < N_GC; ++i) {
      double p_i = std::abs(z_GC_curr_m4[i]) / l1_sum_m4;
      if (p_i > 1e-12) S_t_m4 -= p_i * std::log(p_i);
    }
    double norm_entropy_m4 = std::min(1.0, S_t_m4 / max_entropy);

    history_GC_m4.push_back(z_GC_curr_m4);
    if (history_GC_m4.size() > (size_t)window_size) history_GC_m4.erase(history_GC_m4.begin());

    double capacity_proxy_m4 = 0.0;
    if (history_GC_m4.size() > 1) {
      double diff_sq = 0.0;
      for (int i = 0; i < N_GC; ++i) {
        double diff = history_GC_m4.back()[i] - history_GC_m4.front()[i];
        diff_sq += diff * diff;
      }
      capacity_proxy_m4 = std::sqrt(diff_sq / (double)N_GC);
    } else {
      capacity_proxy_m4 = 0.10;
    }

    // Wasserstein filtered Persistent Entropy
    double theta_W = 0.18;
    double raw_omega_m4 = capacity_proxy_m4 * norm_entropy_m4;
    double omega_tilde_m4 = (raw_omega_m4 > theta_W) ? (raw_omega_m4 - theta_W) : 0.0;
    omega_m4_vec[t] = omega_tilde_m4;

    // DCN Biophysical Rebound Dynamics
    double g_L = 1.0;
    double E_L = -60.0;
    double g_GABA = 4.0;
    double E_Cl = -75.0;
    double V_half = -68.0;
    double k_h = 2.5;
    double kappa_sync = 3.5;

    double S_sync_m4 = std::exp(-kappa_sync * omega_tilde_m4);
    double V_star_DCN = (g_L * E_L + g_GABA * S_sync_m4 * E_Cl) / (g_L + g_GABA * S_sync_m4);
    double h_inf = 1.0 / (1.0 + std::exp((V_star_DCN - V_half) / k_h));
    double N_raw_m4 = 1.0 * h_inf;

    // Extracellular reuptake convolution
    double decay_factor = std::exp(-delta_t / tau_NA);
    N_eff_state = decay_factor * N_eff_state + (1.0 - decay_factor) * N_raw_m4;
    n_eff_m4_vec[t] = N_eff_state;

    // Policy extraction
    double gc_policy_diff_m4 = 0.0;
    for (int i = 0; i < N_GC; ++i) gc_policy_diff_m4 += (W_pi_m4[0][i] - W_pi_m4[1][i]) * z_GC_curr_m4[i];
    double mli_policy_diff_m4 = 0.0;
    for (int m = 0; m < N_MLI; ++m) mli_policy_diff_m4 += (W_inh_m4[0][m] - W_inh_m4[1][m]) * h_MLI_m4[m];

    double q_diff_m4 = Q_val_m4[0] - Q_val_m4[1];
    
    // Deliberative vs Heuristic components
    v_delib_m4_vec[t] = 0.55 * q_diff_m4 + 0.35 * (gc_policy_diff_m4 - mli_policy_diff_m4);
    v_heur_m4_vec[t]  = logit_bias;

    // LC Salience
    double V_curr_m4 = b_v_m4;
    for (int i = 0; i < N_GC; ++i) V_curr_m4 += w_v_m4[i] * z_GC_curr_m4[i];
    double delta_rpe_m4 = (double)out - V_curr_m4;
    lc_weight_vec[t] = 1.0 + 0.50 * std::log(1.0 + delta_t) + 0.50 * std::abs(delta_rpe_m4);

    // Multiplicative Plasticity for Model 4
    double Omega_t_m4 = std::exp(-kappa_entropy * S_t_m4);
    double p_ch1_m4 = 1.0 / (1.0 + std::exp(-(logit_bias + v_delib_m4_vec[t])));
    double p_chosen_m4 = (ch == 1) ? p_ch1_m4 : (1.0 - p_ch1_m4);
    p_chosen_m4 = clamp_val(p_chosen_m4, 1e-6, 1.0 - 1e-6);

    for (int i = 0; i < N_GC; ++i) {
      w_v_m4[i] *= std::exp(clamp_val(0.05 * lr_scale * Omega_t_m4 * delta_rpe_m4 * z_GC_curr_m4[i], -1.0, 1.0));
      int a_idx = (ch == 1) ? 0 : 1;
      W_pi_m4[a_idx][i] *= std::exp(clamp_val(0.05 * lr_scale * Omega_t_m4 * delta_rpe_m4 * (1.0 - p_chosen_m4) * z_GC_curr_m4[i], -1.0, 1.0));
    }
    for (int m = 0; m < N_MLI; ++m) {
      int a_idx = (ch == 1) ? 0 : 1;
      W_inh_m4[a_idx][m] *= std::exp(clamp_val(0.02 * lr_scale * Omega_t_m4 * delta_rpe_m4 * (1.0 - p_chosen_m4) * h_MLI_m4[m], -1.0, 1.0));
    }
    b_v_m4 *= std::exp(clamp_val(0.02 * Omega_t_m4 * delta_rpe_m4, -0.2, 0.2));
    Q_val_m4[ch - 1] += alpha_q * ((double)out - Q_val_m4[ch - 1]);

    z_GC_prev_m4 = z_GC_curr_m4;
    V_prev_m4 = V_curr_m4;
  }

  return List::create(
    Named("V_M0") = v_m0_vec,
    Named("V_M1") = v_m1_vec,
    Named("V_M2") = v_m2_vec,
    Named("RPE_M2") = rpe_m2_vec,
    Named("V_M3") = v_m3_vec,
    Named("Rebound_M3") = rebound_m3_vec,
    Named("Omega_M4") = omega_m4_vec,
    Named("N_eff_M4") = n_eff_m4_vec,
    Named("V_delib_M4") = v_delib_m4_vec,
    Named("V_heur_M4") = v_heur_m4_vec,
    Named("LC_Weight") = lc_weight_vec
  );
}

// [[Rcpp::export]]
double compute_temporal_topological_hddm_deviance_cpp(
    const IntegerVector& resp_R,
    const NumericVector& rt_R,
    const NumericVector& v_delib_R,
    const NumericVector& v_heur_R,
    const NumericVector& n_eff_R,
    const NumericVector& weights_R,
    double beta_delib,
    double beta_heur,
    double a_0,
    double eta_a,
    double t_nd
) {
  int N_t = resp_R.size();
  double total_w_nll = 0.0;

  for (int t = 0; t < N_t; ++t) {
    int ch = resp_R[t];
    double rt_emp = rt_R[t];
    double wt = weights_R[t];
    double n_eff = n_eff_R[t];

    double v_t = (1.0 - n_eff) * (beta_delib * v_delib_R[t]) + n_eff * (beta_heur * v_heur_R[t]);
    double a_t = std::max(0.30, a_0 * std::exp(-eta_a * n_eff));

    double dens = wiener_pdf(rt_emp, ch, v_t, a_t, t_nd);
    total_w_nll -= wt * std::log(dens);
  }
  return 2.0 * total_w_nll;
}

// [[Rcpp::export]]
List evaluate_temporal_topological_hddm_model_cpp(
    const IntegerVector& resp_R,
    const NumericVector& rt_R,
    const NumericVector& v_delib_R,
    const NumericVector& v_heur_R,
    const NumericVector& n_eff_R,
    double beta_delib,
    double beta_heur,
    double a_0,
    double eta_a,
    double t_nd
) {
  int N_t = resp_R.size();
  NumericVector p_choice1_vec(N_t);
  NumericVector p_switch_vec(N_t);
  NumericVector pred_mean_rt_vec(N_t);
  NumericVector a_t_vec(N_t);
  NumericVector v_t_vec(N_t);
  double total_unweighted_deviance = 0.0;
  double total_brier = 0.0;
  int correct_choice_matches = 0;

  for (int t = 0; t < N_t; ++t) {
    int ch = resp_R[t];
    int prev_ch = (t > 0) ? resp_R[t - 1] : 1;
    double rt_emp = rt_R[t];
    double n_eff = n_eff_R[t];

    double v_t = (1.0 - n_eff) * (beta_delib * v_delib_R[t]) + n_eff * (beta_heur * v_heur_R[t]);
    double a_t = std::max(0.30, a_0 * std::exp(-eta_a * n_eff));

    v_t_vec[t] = v_t;
    a_t_vec[t] = a_t;

    double dens = wiener_pdf(rt_emp, ch, v_t, a_t, t_nd);
    total_unweighted_deviance -= 2.0 * std::log(dens);

    double p_ch1 = 1.0 / (1.0 + std::exp(-v_t * a_t));
    p_choice1_vec[t] = p_ch1;

    // Switch Probability
    double p_sw = (prev_ch == 1) ? (1.0 - p_ch1) : p_ch1;
    p_switch_vec[t] = p_sw;

    double actual_ch1 = (ch == 1) ? 1.0 : 0.0;
    double brier_err = (p_ch1 - actual_ch1);
    total_brier += (brier_err * brier_err);

    int pred_ch = (v_t >= 0.0) ? 1 : 2;
    if (pred_ch == ch) correct_choice_matches++;

    double expected_dt = (std::abs(v_t) > 1e-4) ? (a_t / (2.0 * v_t)) * std::tanh(v_t * a_t / 2.0) : (a_t * a_t / 4.0);
    pred_mean_rt_vec[t] = t_nd + std::max(0.05, expected_dt);
  }

  return List::create(
    Named("Deviance") = total_unweighted_deviance,
    Named("Brier_Score") = total_brier / (double)N_t,
    Named("Choice_Accuracy") = (double)correct_choice_matches / (double)N_t * 100.0,
    Named("Pred_Mean_RT") = pred_mean_rt_vec,
    Named("P_Choice1") = p_choice1_vec,
    Named("P_Switch") = p_switch_vec,
    Named("A_t") = a_t_vec,
    Named("V_t") = v_t_vec
  );
}

// [[Rcpp::export]]
double compute_standard_tournament_deviance_cpp(
    const IntegerVector& resp_R,
    const NumericVector& rt_R,
    const NumericVector& v_raw_R,
    const NumericVector& mod_raw_R,
    const NumericVector& weights_R,
    double beta_v,
    double a_0,
    double kappa_mod,
    double t_nd
) {
  int N_t = resp_R.size();
  double total_w_nll = 0.0;

  for (int t = 0; t < N_t; ++t) {
    int ch = resp_R[t];
    double rt_emp = rt_R[t];
    double wt = weights_R[t];
    double v_t = beta_v * v_raw_R[t];
    double a_t = std::max(0.30, a_0 + kappa_mod * mod_raw_R[t]);

    double dens = wiener_pdf(rt_emp, ch, v_t, a_t, t_nd);
    total_w_nll -= wt * std::log(dens);
  }
  return 2.0 * total_w_nll;
}

// [[Rcpp::export]]
List evaluate_standard_model_cpp(
    const IntegerVector& resp_R,
    const NumericVector& rt_R,
    const NumericVector& v_raw_R,
    const NumericVector& mod_raw_R,
    double beta_v,
    double a_0,
    double kappa_mod,
    double t_nd
) {
  int N_t = resp_R.size();
  NumericVector p_choice1_vec(N_t);
  NumericVector p_switch_vec(N_t);
  NumericVector pred_mean_rt_vec(N_t);
  double total_unweighted_deviance = 0.0;
  double total_brier = 0.0;
  int correct_choice_matches = 0;

  for (int t = 0; t < N_t; ++t) {
    int ch = resp_R[t];
    int prev_ch = (t > 0) ? resp_R[t - 1] : 1;
    double rt_emp = rt_R[t];
    double v_t = beta_v * v_raw_R[t];
    double a_t = std::max(0.30, a_0 + kappa_mod * mod_raw_R[t]);

    double dens = wiener_pdf(rt_emp, ch, v_t, a_t, t_nd);
    total_unweighted_deviance -= 2.0 * std::log(dens);

    double p_ch1 = 1.0 / (1.0 + std::exp(-v_t * a_t));
    p_choice1_vec[t] = p_ch1;

    double p_sw = (prev_ch == 1) ? (1.0 - p_ch1) : p_ch1;
    p_switch_vec[t] = p_sw;

    double actual_ch1 = (ch == 1) ? 1.0 : 0.0;
    double brier_err = (p_ch1 - actual_ch1);
    total_brier += (brier_err * brier_err);

    int pred_ch = (v_t >= 0.0) ? 1 : 2;
    if (pred_ch == ch) correct_choice_matches++;

    double expected_dt = (std::abs(v_t) > 1e-4) ? (a_t / (2.0 * v_t)) * std::tanh(v_t * a_t / 2.0) : (a_t * a_t / 4.0);
    pred_mean_rt_vec[t] = t_nd + std::max(0.05, expected_dt);
  }

  return List::create(
    Named("Deviance") = total_unweighted_deviance,
    Named("Brier_Score") = total_brier / (double)N_t,
    Named("Choice_Accuracy") = (double)correct_choice_matches / (double)N_t * 100.0,
    Named("Pred_Mean_RT") = pred_mean_rt_vec,
    Named("P_Choice1") = p_choice1_vec,
    Named("P_Switch") = p_switch_vec
  );
}
