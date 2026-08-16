// [[Rcpp::depends(RcppEigen)]]
#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace Rcpp;

// ==============================================================================
// JOINT KINEMATIC & CHOICE SPARSE RESERVOIR ENGINE
// Evaluates Bivariate Joint Objective: Decision (Choice NLL) + Continuous RT (Gamma/Log-Normal Likelihood)
// ==============================================================================

inline double clamp_val(double v, double lo, double hi) {
  return (v < lo) ? lo : ((v > hi) ? hi : v);
}

// [[Rcpp::export]]
List run_scalable_kinematic_reservoir_cpp(
    const IntegerVector& resp_R,
    const NumericVector& out_R,
    const NumericVector& m1_R,
    const NumericVector& m2_R,
    const NumericVector& rt_R,
    const NumericVector& ttp_R,
    const NumericVector& params_R,
    double mean_rt_sub = 0.55,
    int N_GC = 100,
    bool is_ablated = false
) {
  int N_t = resp_R.size();
  
  // Parse Parameters
  double p_ws_base       = params_R[0]; // Base Win-Stay probability
  double p_ls_base       = params_R[1]; // Base Lose-Shift probability
  double w_mag_curr      = params_R[2]; // Magnitude sensitivity
  double w_mag_alt       = params_R[3]; // Alternative magnitude sensitivity
  double alpha_q         = params_R[4]; // Q-learning rate
  double w_streak        = params_R[5]; // Streak acceleration
  double w_purkinje_inh  = is_ablated ? 0.0 : params_R[6]; // Severed in ablation
  double tau_kinematic   = is_ablated ? 0.001 : params_R[7]; // Fading memory eliminated in ablation
  double beta_post_err   = params_R[8]; // Post-error slowing
  double kappa_entropy   = params_R[9]; // Entropy modulation on RT

  int N_GoC = std::max(5, N_GC / 4);
  int N_actions = 2;

  // Sparse indexing for Mossy -> Granule (approx 10-25% biological sparsity)
  int mossy_fan_in = 4;
  std::vector<int> gc_mossy_idx(N_GC);
  std::vector<double> gc_mossy_w(N_GC);
  for (int i = 0; i < N_GC; ++i) {
    gc_mossy_idx[i] = i % mossy_fan_in;
    gc_mossy_w[i] = 0.40;
  }

  // Sparse Granule -> Golgi connections (fan-in = 4)
  std::vector<std::vector<int>> goc_gc_indices(N_GoC, std::vector<int>(4));
  for (int j = 0; j < N_GoC; ++j) {
    for (int k = 0; k < 4; ++k) {
      goc_gc_indices[j][k] = (j * 4 + k) % N_GC;
    }
  }

  // State buffers
  std::vector<double> z_GC_prev(N_GC, 0.0);
  std::vector<double> z_GC_curr(N_GC, 0.0);
  std::vector<double> z_GoC(N_GoC, 0.0);

  std::vector<double> rho_vec(N_GC, is_ablated ? 0.0 : 0.70);
  std::vector<double> tau_vec(N_GC, std::max(0.001, tau_kinematic));

  // Mutable synaptic weights
  std::vector<double> w_v(N_GC, 0.10);
  std::vector<std::vector<double>> W_pi(N_actions, std::vector<double>(N_GC, 0.10));
  double b_v = 0.50;

  double V_prev = 0.0;
  double Q_val[2] = {0.50, 0.50};
  int loss_streak = 0;
  double rt_filter = mean_rt_sub;

  NumericVector log_lik_choice_vec(N_t);
  NumericVector log_lik_rt_vec(N_t);
  NumericVector joint_log_lik_vec(N_t);
  NumericVector p_chosen_vec(N_t);
  NumericVector rt_pred_vec(N_t);
  NumericVector spatial_entropy_vec(N_t);

  double total_choice_nll = 0.0;
  double total_rt_sse = 0.0;
  double total_joint_nll = 0.0;
  double sigma_rt = 0.25; // Standard error of Log-Normal RT

  for (int t = 0; t < N_t; ++t) {
    int ch = resp_R[t];
    int out = out_R[t];
    double m1 = m1_R[t];
    double m2 = m2_R[t];
    double rt_emp = rt_R[t];
    double delta_t = (t == 0) ? 1.5 : std::max(0.1, (double)(ttp_R[t] - ttp_R[t - 1]));

    int prev_ch  = (t > 0) ? resp_R[t - 1] : 1;
    int prev_out = (t > 0) ? out_R[t - 1] : 1;

    // Option magnitudes
    double m_curr = (prev_ch == 1) ? m1 : m2;
    double m_alt  = (prev_ch == 1) ? m2 : m1;
    double d_curr = (m_curr - 5.5) / 4.5;
    double d_diff = (m_alt - m_curr) / 4.0;
    double q_diff = Q_val[prev_ch - 1] - Q_val[2 - prev_ch];

    // Mossy fiber input u_t
    double u_arr[4] = {
      (prev_ch == 1) ? 1.0 : -1.0,
      (double)prev_out,
      d_curr,
      d_diff
    };

    // Granule cell forward integration + fading memory
    std::vector<double> h_pre(N_GC, 0.0);
    for (int i = 0; i < N_GC; ++i) {
      double input_i = gc_mossy_w[i] * u_arr[gc_mossy_idx[i]];
      double gamma_i = is_ablated ? 0.0 : (rho_vec[i] + (1.0 - rho_vec[i]) * std::exp(-delta_t / tau_vec[i]));
      h_pre[i] = std::tanh(input_i + gamma_i * z_GC_prev[i]);
    }

    if (!is_ablated) {
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
    } else {
      // Pure instantaneous feedforward expansion
      for (int i = 0; i < N_GC; ++i) {
        z_GC_curr[i] = std::max(0.0, h_pre[i]);
      }
    }

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

    // Value estimation
    double V_curr = b_v;
    for (int i = 0; i < N_GC; ++i) {
      V_curr += w_v[i] * z_GC_curr[i];
    }

    // Policy estimation
    double gc_diff = 0.0;
    for (int i = 0; i < N_GC; ++i) {
      gc_diff += (W_pi[0][i] - W_pi[1][i]) * z_GC_curr[i];
    }

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

    // Continuous RT Readout: RT_hat = tau_kin * RT_filter + beta_err * I_err + kappa_ent * S_t
    double err_indicator = (prev_out == 0) ? 1.0 : 0.0;
    double rt_hat = 0.65 * tau_kinematic * rt_filter + beta_post_err * err_indicator + 0.05 * kappa_entropy * S_t;
    rt_hat = clamp_val(rt_hat, 0.15, 2.50);

    // Update RT kinematic filter
    rt_filter = (1.0 - 0.20) * rt_filter + 0.20 * rt_emp;

    // Continuous RT Log-Likelihood (Log-Normal Model)
    double rt_diff = std::log(rt_emp) - std::log(rt_hat);
    double log_lik_rt = -0.5 * std::log(2.0 * M_PI) - std::log(sigma_rt) - std::log(rt_emp) - 0.5 * (rt_diff * rt_diff) / (sigma_rt * sigma_rt);

    // Joint Log-Likelihood
    double joint_ll = log_lik_choice + log_lik_rt;

    log_lik_choice_vec[t] = log_lik_choice;
    log_lik_rt_vec[t] = log_lik_rt;
    joint_log_lik_vec[t] = joint_ll;
    p_chosen_vec[t] = p_chosen;
    rt_pred_vec[t] = rt_hat;

    total_choice_nll -= log_lik_choice;
    total_rt_sse += (rt_emp - rt_hat) * (rt_emp - rt_hat);
    total_joint_nll -= joint_ll;

    // Symplectic multiplicative plasticity
    double lr_scale = 40.0 / (double)N_GC;
    double reward = (double)out;
    double delta_rpe = reward - V_curr;
    double Omega_t = std::exp(-kappa_entropy * S_t);

    for (int i = 0; i < N_GC; ++i) {
      double kick_v = clamp_val(0.05 * lr_scale * Omega_t * delta_rpe * z_GC_curr[i], -1.0, 1.0);
      w_v[i] = w_v[i] * std::exp(kick_v);

      int a_idx = (ch == 1) ? 0 : 1;
      double kick_pi = clamp_val(0.05 * lr_scale * Omega_t * delta_rpe * (1.0 - p_chosen) * z_GC_curr[i], -1.0, 1.0);
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

  double rt_rmse = std::sqrt(total_rt_sse / std::max(1, N_t));
  double total_choice_ll = -total_choice_nll;
  double total_rt_ll = sum(log_lik_rt_vec);
  double total_joint_ll = -total_joint_nll;

  return List::create(
    Named("Choice_LogLik") = total_choice_ll,
    Named("RT_LogLik") = total_rt_ll,
    Named("Joint_LogLik") = total_joint_ll,
    Named("RT_RMSE") = rt_rmse,
    Named("RT_Pred") = rt_pred_vec,
    Named("P_Chosen") = p_chosen_vec,
    Named("Spatial_Entropy") = spatial_entropy_vec
  );
}
