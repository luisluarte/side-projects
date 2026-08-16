// [[Rcpp::depends(RcppEigen)]]
#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

using namespace Rcpp;

// ==============================================================================
// SYMPLECTIC HIGH-DIMENSIONAL CORTICO-CEREBELLAR RESERVOIR ENGINE
// ==============================================================================
// Combines High-Dimensional Sparse Granule-Golgi Recurrent Microcircuit Dynamics
// with Symplectic Multiplicative Synaptic Updates:
// w_{v, t+1} = w_{v, t} \odot \exp(\eta_v \Omega_t \delta_t e_t)
// W_{\pi, a_t, t+1} = W_{\pi, a_t, t} \odot \exp(\eta_\pi \Omega_t \delta_t (1 - \pi_{a_t}) e_t)
// ==============================================================================

inline double clamp_val(double v, double lo, double hi) {
  return (v < lo) ? lo : ((v > hi) ? hi : v);
}

// [[Rcpp::export]]
List run_symplectic_simulation_cpp(
    const IntegerVector& resp_R,
    const NumericVector& out_R,
    const NumericVector& m1_R,
    const NumericVector& m2_R,
    const NumericVector& rt_R,
    const NumericVector& ttp_R,
    const NumericVector& params_R,
    int N_GC = 40
) {
  int N_t = resp_R.size();
  
  // Parse Parameters
  double p_ws_base       = params_R[0]; // Base Win-Stay probability
  double p_ls_base       = params_R[1]; // Base Lose-Shift probability
  double w_mag_curr      = params_R[2]; // Current magnitude weight
  double w_mag_alt       = params_R[3]; // Alternative magnitude weight
  double alpha_q         = params_R[4]; // Q-learning rate
  double w_streak        = params_R[5]; // Streak acceleration
  double w_purkinje_inh  = params_R[6]; // Purkinje inhibition
  double tau_kinematic   = params_R[7]; // Kinematic filter
  double beta_post_err   = params_R[8]; // Post-error slowing
  double kappa_entropy   = params_R[9]; // Entropy modulation

  int N_GoC = std::max(5, N_GC / 4);
  int N_actions = 2;

  // High-dimensional Granule and Golgi state vectors
  std::vector<double> z_GC_prev(N_GC, 0.0);
  std::vector<double> z_GC_curr(N_GC, 0.0);
  std::vector<double> z_GoC(N_GoC, 0.0);

  std::vector<double> rho_vec(N_GC, 0.70);
  std::vector<double> tau_vec(N_GC, std::max(0.05, tau_kinematic));

  // Mutable symplectic synaptic weights
  std::vector<double> w_v(N_GC, 0.10);
  std::vector<std::vector<double>> W_pi(N_actions, std::vector<double>(N_GC, 0.10));
  double b_v = 0.50;

  double V_prev = 0.0;
  double Q_val[2] = {0.50, 0.50};
  int loss_streak = 0;

  NumericVector log_lik_vec(N_t);
  NumericVector p_chosen_vec(N_t);
  NumericVector spatial_entropy_vec(N_t);
  NumericVector state_norm_vec(N_t);
  NumericVector value_vec(N_t);

  double total_choice_nll = 0.0;

  for (int t = 0; t < N_t; ++t) {
    int ch = resp_R[t];
    int out = out_R[t];
    double m1 = m1_R[t];
    double m2 = m2_R[t];
    double delta_t = (t == 0) ? 1.5 : std::max(0.1, (double)(ttp_R[t] - ttp_R[t - 1]));

    int prev_ch  = (t > 0) ? resp_R[t - 1] : 1;
    int prev_out = (t > 0) ? out_R[t - 1] : 1;

    // Option magnitudes
    double m_curr = (prev_ch == 1) ? m1 : m2;
    double m_alt  = (prev_ch == 1) ? m2 : m1;
    double d_curr = (m_curr - 5.5) / 4.5;
    double d_diff = (m_alt - m_curr) / 4.0;
    double q_diff = Q_val[prev_ch - 1] - Q_val[2 - prev_ch];

    // Mossy fiber input u_t (4 dimensions)
    double u0 = (prev_ch == 1) ? 1.0 : -1.0;
    double u1 = (double)prev_out;
    double u2 = d_curr;
    double u3 = d_diff;
    double u_arr[4] = {u0, u1, u2, u3};

    // Granule cell forward integration + fading memory
    std::vector<double> h_pre(N_GC, 0.0);
    for (int i = 0; i < N_GC; ++i) {
      double gamma_i = rho_vec[i] + (1.0 - rho_vec[i]) * std::exp(-delta_t / tau_vec[i]);
      double input_i = 0.40 * u_arr[i % 4];
      h_pre[i] = std::tanh(input_i + gamma_i * z_GC_prev[i]);
    }

    // Golgi recurrent inhibition
    for (int j = 0; j < N_GoC; ++j) {
      double exc = 0.15 * V_prev;
      for (int k = 0; k < 3; ++k) {
        int gc_idx = (j * 3 + k) % N_GC;
        exc += 0.25 * h_pre[gc_idx];
      }
      z_GoC[j] = std::max(0.0, exc);
    }

    // Granule cell state
    double state_norm_sq = 0.0;
    for (int i = 0; i < N_GC; ++i) {
      double inh = w_purkinje_inh * z_GoC[i % N_GoC];
      z_GC_curr[i] = std::max(0.0, h_pre[i] - inh);
      state_norm_sq += z_GC_curr[i] * z_GC_curr[i];
    }
    double state_norm = std::sqrt(state_norm_sq);

    // Value estimation
    double V_curr = b_v;
    for (int i = 0; i < N_GC; ++i) {
      V_curr += w_v[i] * z_GC_curr[i];
    }

    // Cortico-cerebellar policy integration
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
        double logit_stay = logit_ws + w_mag_curr * d_curr + 0.35 * q_diff;
        p_stay = 1.0 / (1.0 + std::exp(-logit_stay));
        p_stay = clamp_val(p_stay, 0.001, 0.999);
        p_switch = 1.0 - p_stay;
      } else {
        loss_streak += 1;
        double streak_term = w_streak * std::log(1.0 + loss_streak);
        double logit_ls = std::log(p_ls_base / (1.0 - p_ls_base + 1e-12));
        double logit_shift = logit_ls + w_mag_alt * d_diff + streak_term - 0.35 * q_diff;
        p_switch = 1.0 / (1.0 + std::exp(-logit_shift));
        p_switch = clamp_val(p_switch, 0.001, 0.999);
        p_stay = 1.0 - p_switch;
      }
    }

    double prob_1 = (prev_ch == 1) ? p_stay : p_switch;
    double prob_2 = 1.0 - prob_1;

    double p_chosen = (ch == 1) ? prob_1 : prob_2;
    p_chosen = clamp_val(p_chosen, 1e-6, 1.0 - 1e-6);

    log_lik_vec[t] = std::log(p_chosen);
    p_chosen_vec[t] = p_chosen;
    total_choice_nll -= std::log(p_chosen);
    value_vec[t] = V_curr;
    state_norm_vec[t] = state_norm;

    // Granule cell spatial entropy
    double l1_sum = 1e-12;
    for (int i = 0; i < N_GC; ++i) {
      l1_sum += std::abs(z_GC_curr[i]);
    }
    double S_t = 0.0;
    for (int i = 0; i < N_GC; ++i) {
      double p_i = std::abs(z_GC_curr[i]) / l1_sum;
      if (p_i > 1e-12) {
        S_t -= p_i * std::log(p_i);
      }
    }
    spatial_entropy_vec[t] = S_t;
    double Omega_t = std::exp(-kappa_entropy * S_t);

    // Symplectic Multiplicative Synaptic Updates
    double reward = (double)out;
    double delta_rpe = reward - V_curr;

    for (int i = 0; i < N_GC; ++i) {
      double kick_v = clamp_val(0.05 * Omega_t * delta_rpe * z_GC_curr[i], -1.0, 1.0);
      w_v[i] = w_v[i] * std::exp(kick_v);

      int a_idx = (ch == 1) ? 0 : 1;
      double kick_pi = clamp_val(0.05 * Omega_t * delta_rpe * (1.0 - p_chosen) * z_GC_curr[i], -1.0, 1.0);
      W_pi[a_idx][i] = W_pi[a_idx][i] * std::exp(kick_pi);
    }
    b_v = b_v * std::exp(clamp_val(0.02 * Omega_t * delta_rpe, -0.2, 0.2));

    // Q-value update
    int chosen_idx = ch - 1;
    Q_val[chosen_idx] = Q_val[chosen_idx] + alpha_q * (reward - Q_val[chosen_idx]);

    // Update state buffers
    z_GC_prev = z_GC_curr;
    V_prev = V_curr;
  }

  return List::create(
    Named("Total_Choice_NLL") = total_choice_nll,
    Named("Log_Likelihood") = log_lik_vec,
    Named("P_Chosen") = p_chosen_vec,
    Named("Spatial_Entropy") = spatial_entropy_vec,
    Named("State_Norm") = state_norm_vec,
    Named("Value") = value_vec
  );
}