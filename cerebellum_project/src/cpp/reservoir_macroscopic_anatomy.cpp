// [[Rcpp::depends(RcppEigen)]]
#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace Rcpp;

// ==============================================================================
// MACROSCOPIC ANATOMY & POPULATION CODING RESERVOIR ENGINE
// Integrates:
// 1. Anatomical Segregation: 500-D Cognitive Crus I/II vs. 500-D Motor Hemisphere (Option A)
// 2. Pontine Population Vector Encoding: Gaussian RBF Time Cells for Delta_t & RT_prev (Option B)
// ==============================================================================

inline double clamp_val(double v, double lo, double hi) {
  return (v < lo) ? lo : ((v > hi) ? hi : v);
}

// [[Rcpp::export]]
List run_macroscopic_reservoir_cpp(
    const IntegerVector& resp_R,
    const NumericVector& out_R,
    const NumericVector& m1_R,
    const NumericVector& m2_R,
    const NumericVector& rt_R,
    const NumericVector& ttp_R,
    const NumericVector& ttr_R,
    const NumericVector& ttf_R,
    const NumericVector& params_R,
    double mean_rt_global = 0.55,
    bool enable_segregation = true,
    bool enable_rbf_timecells = true
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

  int N_GC_total = 1000;
  int N_GC_cog   = enable_segregation ? 500 : 1000;
  int N_GC_mot   = enable_segregation ? 500 : 1000;

  int N_GoC_cog = std::max(5, N_GC_cog / 4);
  int N_GoC_mot = std::max(5, N_GC_mot / 4);
  int N_actions = 2;

  // 1. MOSS FIBER INPUT VECTOR CONFIGURATION
  // Standard inputs: [prev_ch, prev_out, d_curr, d_diff, norm_delta_t, norm_prev_rt, norm_prev_ttf] (dim=7)
  // OPTION B: Pontine RBF expansion for Delta_t (5 kernels: 0.5s, 2.0s, 10s, 30s, 60s) and RT_prev (3 kernels: 0.2s, 0.6s, 1.5s)
  // Total mossy fiber dimensions: 4 (discrete) + 5 (Delta_t RBFs) + 3 (RT RBFs) + 1 (TTF) = 13 dimensions
  int n_mossy = enable_rbf_timecells ? 13 : 7;

  // Mossy fan-in connections
  std::vector<int> cog_mossy_idx(N_GC_cog);
  std::vector<double> cog_mossy_w(N_GC_cog, 0.35);
  for (int i = 0; i < N_GC_cog; ++i) {
    cog_mossy_idx[i] = i % n_mossy;
  }

  std::vector<int> mot_mossy_idx(N_GC_mot);
  std::vector<double> mot_mossy_w(N_GC_mot, 0.35);
  for (int i = 0; i < N_GC_mot; ++i) {
    mot_mossy_idx[i] = i % n_mossy;
  }

  // Golgi connections
  std::vector<std::vector<int>> goc_cog_indices(N_GoC_cog, std::vector<int>(4));
  for (int j = 0; j < N_GoC_cog; ++j) {
    for (int k = 0; k < 4; ++k) goc_cog_indices[j][k] = (j * 4 + k) % N_GC_cog;
  }
  std::vector<std::vector<int>> goc_mot_indices(N_GoC_mot, std::vector<int>(4));
  for (int j = 0; j < N_GoC_mot; ++j) {
    for (int k = 0; k < 4; ++k) goc_mot_indices[j][k] = (j * 4 + k) % N_GC_mot;
  }

  // States
  std::vector<double> z_cog_prev(N_GC_cog, 0.0);
  std::vector<double> z_cog_curr(N_GC_cog, 0.0);
  std::vector<double> z_GoC_cog(N_GoC_cog, 0.0);

  std::vector<double> z_mot_prev(N_GC_mot, 0.0);
  std::vector<double> z_mot_curr(N_GC_mot, 0.0);
  std::vector<double> z_GoC_mot(N_GoC_mot, 0.0);

  std::vector<double> rho_cog(N_GC_cog, 0.70);
  std::vector<double> tau_cog(N_GC_cog, std::max(0.001, tau_kinematic));

  std::vector<double> rho_mot(N_GC_mot, 0.70);
  std::vector<double> tau_mot(N_GC_mot, std::max(0.001, tau_kinematic));

  std::vector<double> w_v_cog(N_GC_cog, 0.10);
  std::vector<std::vector<double>> W_pi_cog(N_actions, std::vector<double>(N_GC_cog, 0.10));
  double b_v_cog = 0.50;

  double V_prev_cog = 0.0;
  double Q_val[2] = {0.50, 0.50};
  int loss_streak = 0;
  double rt_filter = mean_rt_global;

  NumericVector log_lik_choice_vec(N_t);
  NumericVector p_chosen_vec(N_t);
  NumericVector rt_base_vec(N_t);
  NumericVector spatial_entropy_vec(N_t);
  NumericVector sample_weight_vec(N_t);
  
  // Output Motor Granular Matrix for Kinematic Decoding
  NumericMatrix z_mot_matrix(N_t, N_GC_mot);

  double total_choice_nll = 0.0;

  // Gaussian RBF centers and widths
  double rbf_centers_dt[5] = {0.5, 2.0, 10.0, 30.0, 60.0};
  double rbf_widths_dt[5]  = {0.4, 1.2,  6.0, 15.0, 30.0};

  double rbf_centers_rt[3] = {0.25, 0.60, 1.50};
  double rbf_widths_rt[3]  = {0.15, 0.25, 0.60};

  for (int t = 0; t < N_t; ++t) {
    int ch = resp_R[t];
    int out = out_R[t];
    double m1 = m1_R[t];
    double m2 = m2_R[t];
    double rt_emp = rt_R[t];
    double delta_t = (t == 0) ? 1.5 : std::max(0.1, (double)(ttp_R[t] - ttp_R[t - 1]));

    int prev_ch  = (t > 0) ? resp_R[t - 1] : 1;
    int prev_out = (t > 0) ? out_R[t - 1] : 1;
    double prev_rt = (t > 0) ? rt_R[t - 1] : mean_rt_global;
    double prev_ttf = (t > 0) ? std::max(0.1, (double)(ttf_R[t - 1] - ttr_R[t - 1])) : 1.5;

    double m_curr = (prev_ch == 1) ? m1 : m2;
    double m_alt  = (prev_ch == 1) ? m2 : m1;
    double d_curr = (m_curr - 5.5) / 4.5;
    double d_diff = (m_alt - m_curr) / 4.0;
    double q_diff = Q_val[prev_ch - 1] - Q_val[2 - prev_ch];

    std::vector<double> u_vec(n_mossy);
    if (!enable_rbf_timecells) {
      u_vec[0] = (prev_ch == 1) ? 1.0 : -1.0;
      u_vec[1] = (double)prev_out;
      u_vec[2] = d_curr;
      u_vec[3] = d_diff;
      u_vec[4] = clamp_val(delta_t / 10.0, 0.0, 5.0);
      u_vec[5] = clamp_val(prev_rt / 1.0, 0.0, 3.0);
      u_vec[6] = clamp_val(prev_ttf / 2.0, 0.0, 3.0);
    } else {
      // Pontine Population Vector Encoding (Option B: Time Cells)
      u_vec[0] = (prev_ch == 1) ? 1.0 : -1.0;
      u_vec[1] = (double)prev_out;
      u_vec[2] = d_curr;
      u_vec[3] = d_diff;
      
      // 5 Delta_t RBF kernels
      for (int k = 0; k < 5; ++k) {
        double d_log = std::log(1.0 + delta_t) - std::log(1.0 + rbf_centers_dt[k]);
        double w_log = std::log(1.0 + rbf_widths_dt[k]);
        u_vec[4 + k] = std::exp(-0.5 * (d_log * d_log) / (w_log * w_log + 1e-6));
      }
      
      // 3 RT RBF kernels
      for (int k = 0; k < 3; ++k) {
        double d_rt = prev_rt - rbf_centers_rt[k];
        u_vec[9 + k] = std::exp(-0.5 * (d_rt * d_rt) / (rbf_widths_rt[k] * rbf_widths_rt[k] + 1e-6));
      }
      
      // 1 TTF scalar
      u_vec[12] = clamp_val(prev_ttf / 2.0, 0.0, 3.0);
    }

    // --- COGNITIVE RESERVOIR FORWARD PASS (CRUS I/II) ---
    std::vector<double> h_pre_cog(N_GC_cog, 0.0);
    for (int i = 0; i < N_GC_cog; ++i) {
      double input_i = cog_mossy_w[i] * u_vec[cog_mossy_idx[i]];
      double gamma_i = rho_cog[i] + (1.0 - rho_cog[i]) * std::exp(-delta_t / tau_cog[i]);
      h_pre_cog[i] = std::tanh(input_i + gamma_i * z_cog_prev[i]);
    }
    for (int j = 0; j < N_GoC_cog; ++j) {
      double exc = 0.15 * V_prev_cog;
      for (int k = 0; k < 4; ++k) exc += 0.25 * h_pre_cog[goc_cog_indices[j][k]];
      z_GoC_cog[j] = std::max(0.0, exc);
    }
    for (int i = 0; i < N_GC_cog; ++i) {
      z_cog_curr[i] = std::max(0.0, h_pre_cog[i] - w_purkinje_inh * z_GoC_cog[i % N_GoC_cog]);
    }

    // --- KINEMATIC RESERVOIR FORWARD PASS (MOTOR HEMISPHERE) ---
    std::vector<double> h_pre_mot(N_GC_mot, 0.0);
    for (int i = 0; i < N_GC_mot; ++i) {
      double input_i = mot_mossy_w[i] * u_vec[mot_mossy_idx[i]];
      double gamma_i = rho_mot[i] + (1.0 - rho_mot[i]) * std::exp(-delta_t / tau_mot[i]);
      h_pre_mot[i] = std::tanh(input_i + gamma_i * z_mot_prev[i]);
    }
    for (int j = 0; j < N_GoC_mot; ++j) {
      double exc = 0.10;
      for (int k = 0; k < 4; ++k) exc += 0.25 * h_pre_mot[goc_mot_indices[j][k]];
      z_GoC_mot[j] = std::max(0.0, exc);
    }
    for (int i = 0; i < N_GC_mot; ++i) {
      z_mot_curr[i] = std::max(0.0, h_pre_mot[i] - w_purkinje_inh * z_GoC_mot[i % N_GoC_mot]);
      z_mot_matrix(t, i) = z_mot_curr[i];
    }

    // Spatial Entropy of Motor Reservoir
    double l1_sum = 1e-12;
    for (int i = 0; i < N_GC_mot; ++i) l1_sum += std::abs(z_mot_curr[i]);
    double S_t = 0.0;
    for (int i = 0; i < N_GC_mot; ++i) {
      double p_i = std::abs(z_mot_curr[i]) / l1_sum;
      if (p_i > 1e-12) S_t -= p_i * std::log(p_i);
    }
    spatial_entropy_vec[t] = S_t;

    // Value estimation in Cognitive Reservoir
    double V_curr_cog = b_v_cog;
    for (int i = 0; i < N_GC_cog; ++i) V_curr_cog += w_v_cog[i] * z_cog_curr[i];

    // Choice policy in Cognitive Reservoir
    double gc_diff = 0.0;
    for (int i = 0; i < N_GC_cog; ++i) gc_diff += (W_pi_cog[0][i] - W_pi_cog[1][i]) * z_cog_curr[i];

    double p_stay = 0.50;
    double p_switch = 0.50;

    if (t == 0) {
      p_stay = 0.50;
      p_switch = 0.50;
      loss_streak = 0;
    } else {
      if (prev_out == 1) {
        loss_streak = 0;
        double logit_ws = std::log(p_ws_base / (1.0 - p_ws_base + 1e-12));
        double logit_stay = logit_ws + w_mag_curr * d_curr + 0.35 * q_diff + 0.15 * gc_diff;
        p_stay = 1.0 / (1.0 + std::exp(-logit_stay));
        p_stay = clamp_val(p_stay, 0.001, 0.999);
        p_switch = 1.0 - p_stay;
      } else {
        loss_streak += 1;
        double streak_term = w_streak * std::log(1.0 + loss_streak);
        double logit_ls = std::log(p_ls_base / (1.0 - p_ls_base + 1e-12));
        double logit_shift = logit_ls + w_mag_alt * d_diff + streak_term - 0.35 * q_diff - 0.15 * gc_diff;
        p_switch = 1.0 / (1.0 + std::exp(-logit_shift));
        p_switch = clamp_val(p_switch, 0.001, 0.999);
        p_stay = 1.0 - p_switch;
      }
    }

    double prob_1 = (prev_ch == 1) ? p_stay : p_switch;
    double prob_2 = 1.0 - prob_1;

    double p_chosen = (ch == 1) ? prob_1 : prob_2;
    p_chosen = clamp_val(p_chosen, 1e-6, 1.0 - 1e-6);
    double log_lik_choice = std::log(p_chosen);

    log_lik_choice_vec[t] = log_lik_choice;
    p_chosen_vec[t] = p_chosen;
    total_choice_nll -= log_lik_choice;

    // Autoregressive baseline RT
    double rt_base_t = 0.70 * tau_kinematic * rt_filter + (1.0 - 0.70 * tau_kinematic) * mean_rt_global;
    rt_base_vec[t] = rt_base_t;

    // Update RT filter
    rt_filter = (1.0 - 0.20) * rt_filter + 0.20 * rt_emp;

    // Plasticity update strictly in Cognitive Reservoir
    double lr_scale = 40.0 / (double)N_GC_cog;
    double reward = (double)out;
    double delta_rpe = reward - V_curr_cog;
    double Omega_t = std::exp(-kappa_entropy * S_t);

    for (int i = 0; i < N_GC_cog; ++i) {
      double kick_v = clamp_val(0.05 * lr_scale * Omega_t * delta_rpe * z_cog_curr[i], -1.0, 1.0);
      w_v_cog[i] = w_v_cog[i] * std::exp(kick_v);

      int a_idx = (ch == 1) ? 0 : 1;
      double kick_pi = clamp_val(0.05 * lr_scale * Omega_t * delta_rpe * (1.0 - p_chosen) * z_cog_curr[i], -1.0, 1.0);
      W_pi_cog[a_idx][i] = W_pi_cog[a_idx][i] * std::exp(kick_pi);
    }
    b_v_cog = b_v_cog * std::exp(clamp_val(0.02 * Omega_t * delta_rpe, -0.2, 0.2));

    // Q-value update
    int chosen_idx = ch - 1;
    Q_val[chosen_idx] = Q_val[chosen_idx] + alpha_q * (reward - Q_val[chosen_idx]);

    // LC Noradrenergic Sample Weight
    double w_sample = 1.0 + 0.60 * std::log(1.0 + delta_t) + 0.60 * std::abs(delta_rpe);
    sample_weight_vec[t] = w_sample;

    z_cog_prev = z_cog_curr;
    z_mot_prev = z_mot_curr;
    V_prev_cog = V_curr_cog;
  }

  return List::create(
    Named("Choice_LogLik") = -total_choice_nll,
    Named("Log_Lik_Choice_Vec") = log_lik_choice_vec,
    Named("RT_Base_Vec") = rt_base_vec,
    Named("Spatial_Entropy_Vec") = spatial_entropy_vec,
    Named("Sample_Weight_Vec") = sample_weight_vec,
    Named("Z_Mot_Matrix") = z_mot_matrix
  );
}
