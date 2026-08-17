// [[Rcpp::depends(RcppEigen)]]
#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace Rcpp;

// ==============================================================================
// 5-WAY STOCHASTIC TOURNAMENT ENGINE
// Model 0: Intercept-Only DDM (Null Baseline)
// Model 1: WSLS-HDDM (Markovian Heuristic)
// Model 2: RW-CF-HDDM (Counterfactual Value Tracker + |RPE| Boundary)
// Model 3: Kernelized Symplectic-HDDM (1,000-D Reservoir + MLI + Static DCN Rebound)
// Model 4: Topologically-Gated Symplectic-HDDM (Persistent Entropy + LC Smoothing + Dynamic a_t/v_t Gating)
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
List precompute_topological_tournament_subject_cpp(
    const IntegerVector& resp_R,
    const NumericVector& out_R,
    const NumericVector& m1_R,
    const NumericVector& m2_R,
    const NumericVector& rt_R,
    const NumericVector& ttp_R,
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

  // 4D MOSSY FIBER INPUT: [prev_ch, prev_out, d_curr, d_diff]
  int mossy_fan_in = 4;
  std::vector<int> gc_mossy_idx(N_GC);
  std::vector<double> gc_mossy_w(N_GC, 0.35);
  for (int i = 0; i < N_GC; ++i) gc_mossy_idx[i] = i % mossy_fan_in;

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

  std::vector<double> z_GC_prev(N_GC, 0.0);
  std::vector<double> z_GC_curr(N_GC, 0.0);
  std::vector<double> z_GoC(N_GoC, 0.0);
  std::vector<double> h_MLI(N_MLI, 0.0);

  std::vector<double> rho_vec(N_GC, 0.70);
  std::vector<double> tau_vec(N_GC, std::max(0.001, tau_kinematic));

  std::vector<double> w_v(N_GC, 0.10);
  std::vector<std::vector<double>> W_pi(N_actions, std::vector<double>(N_GC, 0.10));
  std::vector<std::vector<double>> W_inh(N_actions, std::vector<double>(N_MLI, 0.05));
  double b_v = 0.50;

  double V_prev = 0.0;
  double Q_val[2] = {0.50, 0.50};
  double Q_rw_cf[2] = {0.50, 0.50};

  NumericVector v_m0_vec(N_t, 1.0); // Model 0: Intercept only
  NumericVector v_m1_vec(N_t);      // Model 1: WSLS
  NumericVector v_m2_vec(N_t);      // Model 2: RW-CF
  NumericVector rpe_m2_vec(N_t);    // Model 2: |RPE|
  NumericVector v_m3_vec(N_t);      // Model 3: Kernelized Symplectic Log-odds
  NumericVector rebound_m3_vec(N_t);// Model 3: DCN Rebound Brake g(S_t)
  
  // Model 4: Topologically-Gated Symplectic-HDDM
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
  std::vector<std::vector<double>> history_GC;

  for (int t = 0; t < N_t; ++t) {
    int ch = resp_R[t];
    int out = out_R[t];
    double m1 = m1_R[t];
    double m2 = m2_R[t];
    double delta_t = (t == 0) ? 1.5 : std::max(0.1, (double)(ttp_R[t] - ttp_R[t - 1]));

    int prev_ch  = (t > 0) ? resp_R[t - 1] : 1;
    int prev_out = (t > 0) ? out_R[t - 1] : 1;

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

    // --- SYMPLECTIC RESERVOIR MICRO-CIRCUIT ---
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

    // Golgi recurrent inhibition
    for (int j = 0; j < N_GoC; ++j) {
      double exc = 0.15 * V_prev;
      for (int k = 0; k < 4; ++k) exc += 0.25 * h_pre[goc_gc_indices[j][k]];
      z_GoC[j] = std::max(0.0, exc);
    }

    for (int i = 0; i < N_GC; ++i) {
      double inh = w_purkinje_inh * z_GoC[i % N_GoC];
      z_GC_curr[i] = std::max(0.0, h_pre[i] - inh);
    }

    // Static MLI Feedforward Inhibition
    for (int m = 0; m < N_MLI; ++m) {
      double mli_drive = 0.0;
      for (int k = 0; k < mli_fan_in; ++k) {
        mli_drive += z_GC_curr[mli_gc_indices[m][k]];
      }
      mli_drive /= (double)mli_fan_in;
      h_MLI[m] = std::max(0.0, mli_drive - 0.05);
    }

    // --- SPATIAL ENTROPY & SYMPLECTIC PERSISTENCE ---
    double l1_sum = 1e-12;
    for (int i = 0; i < N_GC; ++i) l1_sum += std::abs(z_GC_curr[i]);
    double S_t = 0.0;
    for (int i = 0; i < N_GC; ++i) {
      double p_i = std::abs(z_GC_curr[i]) / l1_sum;
      if (p_i > 1e-12) S_t -= p_i * std::log(p_i);
    }
    double norm_entropy = std::min(1.0, S_t / max_entropy);
    
    // Model 3 Static Rebound Brake
    double dcn_rebound_m3 = 2.85 * std::pow(norm_entropy, 1.75);
    rebound_m3_vec[t] = dcn_rebound_m3;

    // --- MODEL 4: TOPOLOGICALLY-GATED DYNAMICS ---
    // Symplectic capacity tracking across temporal history
    history_GC.push_back(z_GC_curr);
    if (history_GC.size() > (size_t)window_size) history_GC.erase(history_GC.begin());

    double capacity_proxy = 0.0;
    if (history_GC.size() > 1) {
      double diff_sq_sum = 0.0;
      for (int i = 0; i < N_GC; ++i) {
        double d_step = history_GC.back()[i] - history_GC.front()[i];
        diff_sq_sum += d_step * d_step;
      }
      capacity_proxy = std::sqrt(diff_sq_sum / (double)N_GC);
    } else {
      capacity_proxy = 0.10;
    }

    // Wasserstein noise thresholding on persistent entropy
    double theta_W = 0.20;
    double raw_omega = capacity_proxy * norm_entropy;
    double omega_tilde = (raw_omega > theta_W) ? (raw_omega - theta_W) : 0.0;
    omega_m4_vec[t] = omega_tilde;

    // DCN Biophysical Rebound Dynamics
    double g_L = 1.0;
    double E_L = -60.0;
    double g_GABA = 4.0;
    double E_Cl = -75.0;
    double V_half = -68.0;
    double k_h = 2.5;
    double kappa_sync = 3.5;

    double S_sync = std::exp(-kappa_sync * omega_tilde);
    double V_star_DCN = (g_L * E_L + g_GABA * S_sync * E_Cl) / (g_L + g_GABA * S_sync);
    double h_inf = 1.0 / (1.0 + std::exp((V_star_DCN - V_half) / k_h));
    double N_raw = 1.0 * h_inf; // Burst magnitude quantum Gamma = 1.0

    // Extracellular reuptake temporal convolution
    double decay_factor = std::exp(-delta_t / tau_NA);
    N_eff_state = decay_factor * N_eff_state + (1.0 - decay_factor) * N_raw;
    n_eff_m4_vec[t] = N_eff_state;

    // Value estimation
    double V_curr = b_v;
    for (int i = 0; i < N_GC; ++i) V_curr += w_v[i] * z_GC_curr[i];

    // Purkinje - MLI Policy Readout
    double gc_policy_diff = 0.0;
    for (int i = 0; i < N_GC; ++i) {
      gc_policy_diff += (W_pi[0][i] - W_pi[1][i]) * z_GC_curr[i];
    }
    double mli_policy_diff = 0.0;
    for (int m = 0; m < N_MLI; ++m) {
      mli_policy_diff += (W_inh[0][m] - W_inh[1][m]) * h_MLI[m];
    }

    // Directional Logit signals
    double s_ch = (prev_ch == 1) ? 1.0 : -1.0;
    double logit_bias = 0.0;
    if (prev_out == 1) {
      logit_bias = s_ch * (1.0 + w_mag_curr * d_curr); // Win-Stay with magnitude
    } else {
      logit_bias = -s_ch * (1.0 + w_mag_alt * d_diff);  // Lose-Shift with magnitude
    }

    double q_diff_cereb = Q_val[0] - Q_val[1];
    double net_choice_logit = logit_bias + 0.45 * q_diff_cereb + 0.20 * (gc_policy_diff - mli_policy_diff);
    v_m3_vec[t] = net_choice_logit;

    // Model 4 Components: Deliberative vs. Heuristic drift
    v_delib_m4_vec[t] = 0.50 * q_diff_cereb + 0.35 * (gc_policy_diff - mli_policy_diff);
    v_heur_m4_vec[t] = logit_bias;

    // Locus Coeruleus (LC) Salience Metric: w_t = 1.0 + 0.50*log(1 + dt) + 0.50*|RPE|
    double delta_rpe_cereb = (double)out - V_curr;
    double w_salience = 1.0 + 0.50 * std::log(1.0 + delta_t) + 0.50 * std::abs(delta_rpe_cereb);
    lc_weight_vec[t] = w_salience;

    // Multiplicative Symplectic Plasticity
    double lr_scale = 40.0 / (double)N_GC;
    double Omega_t = std::exp(-kappa_entropy * S_t);
    double p_ch1_est = 1.0 / (1.0 + std::exp(-net_choice_logit));
    double p_chosen = (ch == 1) ? p_ch1_est : (1.0 - p_ch1_est);
    p_chosen = clamp_val(p_chosen, 1e-6, 1.0 - 1e-6);

    for (int i = 0; i < N_GC; ++i) {
      double kick_v = clamp_val(0.05 * lr_scale * Omega_t * delta_rpe_cereb * z_GC_curr[i], -1.0, 1.0);
      w_v[i] = w_v[i] * std::exp(kick_v);

      int a_idx = (ch == 1) ? 0 : 1;
      double kick_pi = clamp_val(0.05 * lr_scale * Omega_t * delta_rpe_cereb * (1.0 - p_chosen) * z_GC_curr[i], -1.0, 1.0);
      W_pi[a_idx][i] = W_pi[a_idx][i] * std::exp(kick_pi);
    }
    for (int m = 0; m < N_MLI; ++m) {
      int a_idx = (ch == 1) ? 0 : 1;
      double kick_inh = clamp_val(0.02 * lr_scale * Omega_t * delta_rpe_cereb * (1.0 - p_chosen) * h_MLI[m], -1.0, 1.0);
      W_inh[a_idx][m] = W_inh[a_idx][m] * std::exp(kick_inh);
    }
    b_v = b_v * std::exp(clamp_val(0.02 * Omega_t * delta_rpe_cereb, -0.2, 0.2));

    int chosen_idx = ch - 1;
    Q_val[chosen_idx] += alpha_q * ((double)out - Q_val[chosen_idx]);

    z_GC_prev = z_GC_curr;
    V_prev = V_curr;
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
double compute_topological_hddm_deviance_cpp(
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

    // Dynamically gated drift rate: smooth interpolation between deliberative and heuristic
    double v_t = (1.0 - n_eff) * (beta_delib * v_delib_R[t]) + n_eff * (beta_heur * v_heur_R[t]);
    
    // Dynamically gated boundary: collapses under high NE surge
    double a_t = std::max(0.30, a_0 * std::exp(-eta_a * n_eff));

    double dens = wiener_pdf(rt_emp, ch, v_t, a_t, t_nd);
    total_w_nll -= wt * std::log(dens);
  }
  return 2.0 * total_w_nll;
}

// [[Rcpp::export]]
List evaluate_topological_hddm_model_cpp(
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
  NumericVector pred_mean_rt_vec(N_t);
  NumericVector a_t_vec(N_t);
  NumericVector v_t_vec(N_t);
  double total_unweighted_deviance = 0.0;
  double total_brier = 0.0;
  int correct_choice_matches = 0;

  for (int t = 0; t < N_t; ++t) {
    int ch = resp_R[t];
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

    double actual_ch1 = (ch == 1) ? 1.0 : 0.0;
    double brier_err = (p_ch1 - actual_ch1);
    total_brier += (brier_err * brier_err);

    // Discrete choice accuracy
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
  NumericVector pred_mean_rt_vec(N_t);
  double total_unweighted_deviance = 0.0;
  double total_brier = 0.0;
  int correct_choice_matches = 0;

  for (int t = 0; t < N_t; ++t) {
    int ch = resp_R[t];
    double rt_emp = rt_R[t];
    double v_t = beta_v * v_raw_R[t];
    double a_t = std::max(0.30, a_0 + kappa_mod * mod_raw_R[t]);

    double dens = wiener_pdf(rt_emp, ch, v_t, a_t, t_nd);
    total_unweighted_deviance -= 2.0 * std::log(dens);

    double p_ch1 = 1.0 / (1.0 + std::exp(-v_t * a_t));
    p_choice1_vec[t] = p_ch1;

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
    Named("P_Choice1") = p_choice1_vec
  );
}
