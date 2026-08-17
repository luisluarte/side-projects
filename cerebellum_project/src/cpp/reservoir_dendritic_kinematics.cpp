// [[Rcpp::depends(RcppEigen)]]
#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace Rcpp;

// ==============================================================================
// DENDRITIC COMPARTMENTALIZATION & NON-LINEAR PURKINJE INTEGRATION ENGINE
// Generates:
// 1. Shunted Granular State Matrix: Z_shunted (N_t x N_GC) where Z_shunted(t, i) = z_GC(t, i) * (1 - h_MLI(t, i))
// 2. Compartmentalized Dendritic Tree State Matrix: C_comp (N_t x N_comp) where N_comp = 50
// ==============================================================================

inline double clamp_val(double v, double lo, double hi) {
  return (v < lo) ? lo : ((v > hi) ? hi : v);
}

// [[Rcpp::export]]
List run_dendritic_reservoir_cpp(
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
    int N_GC = 1000,
    int N_comp = 50
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

  // Mossy -> Granule fan-in (7 inputs)
  int mossy_fan_in = 7;
  std::vector<int> gc_mossy_idx(N_GC);
  std::vector<double> gc_mossy_w(N_GC);
  for (int i = 0; i < N_GC; ++i) {
    gc_mossy_idx[i] = i % mossy_fan_in;
    gc_mossy_w[i] = 0.35;
  }

  // Sparse Granule -> Golgi connections (fan-in = 4)
  std::vector<std::vector<int>> goc_gc_indices(N_GoC, std::vector<int>(4));
  for (int j = 0; j < N_GoC; ++j) {
    for (int k = 0; k < 4; ++k) {
      goc_gc_indices[j][k] = (j * 4 + k) % N_GC;
    }
  }

  // Local MLI shunting filters: 1 MLI inhibitory signal per parallel fiber
  std::vector<int> mli_neighbor_idx(N_GC);
  for (int i = 0; i < N_GC; ++i) {
    mli_neighbor_idx[i] = (i + 13) % N_GC;
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
  double rt_filter = mean_rt_global;

  NumericVector log_lik_choice_vec(N_t);
  NumericVector p_chosen_vec(N_t);
  NumericVector rt_base_vec(N_t);
  NumericVector spatial_entropy_vec(N_t);
  NumericVector sample_weight_vec(N_t);
  
  // Topology A Matrix: Multiplicative Shunted State
  NumericMatrix z_shunted_matrix(N_t, N_GC);
  // Topology B Matrix: Compartmentalized Dendritic State (50 branches)
  NumericMatrix c_comp_matrix(N_t, N_comp);
  // Standard Point-Neuron State Matrix
  NumericMatrix z_point_matrix(N_t, N_GC);

  int gc_per_comp = N_GC / N_comp; // 20 PF per branch

  double total_choice_nll = 0.0;

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

    double norm_delta_t = clamp_val(delta_t / 10.0, 0.0, 5.0);
    double norm_prev_rt = clamp_val(prev_rt / 1.0, 0.0, 3.0);
    double norm_prev_ttf = clamp_val(prev_ttf / 2.0, 0.0, 3.0);

    double u_arr[7] = {
      (prev_ch == 1) ? 1.0 : -1.0,
      (double)prev_out,
      d_curr,
      d_diff,
      norm_delta_t,
      norm_prev_rt,
      norm_prev_ttf
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
      z_point_matrix(t, i) = z_GC_curr[i];
    }

    // 1. TOPOLOGY A: Multiplicative Dendritic Shunting
    // Local MLI signal: h_MLI_i = max(0, 0.40 * z_GC_neighbor - 0.02)
    // Gated PF signal: z_shunted_i = z_GC_i * max(0, 1 - h_MLI_i)
    for (int i = 0; i < N_GC; ++i) {
      double h_mli_i = std::max(0.0, 0.45 * z_GC_curr[mli_neighbor_idx[i]] - 0.02);
      double shunted_gain = std::max(0.0, 1.0 - h_mli_i);
      z_shunted_matrix(t, i) = z_GC_curr[i] * shunted_gain;
    }

    // 2. TOPOLOGY B: Hierarchical Two-Stage Dendritic Compartmentalization (50 Branches)
    for (int j = 0; j < N_comp; ++j) {
      double branch_exc = 0.0;
      int start_idx = j * gc_per_comp;
      for (int k = 0; k < gc_per_comp; ++k) {
        branch_exc += z_GC_curr[start_idx + k];
      }
      // Local MLI branch inhibition
      double branch_inh = 0.35 * z_GoC[j % N_GoC];
      double net_branch_input = branch_exc - branch_inh;
      // Dendritic calcium spike activation (Sigmoid / Softplus activation)
      double c_branch = 1.0 / (1.0 + std::exp(-3.5 * (net_branch_input - 0.40)));
      c_comp_matrix(t, j) = c_branch;
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

    // Choice policy
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

    log_lik_choice_vec[t] = log_lik_choice;
    p_chosen_vec[t] = p_chosen;
    total_choice_nll -= log_lik_choice;

    // Autoregressive baseline RT
    double rt_base_t = 0.70 * tau_kinematic * rt_filter + (1.0 - 0.70 * tau_kinematic) * mean_rt_global;
    rt_base_vec[t] = rt_base_t;

    // Update RT filter
    rt_filter = (1.0 - 0.20) * rt_filter + 0.20 * rt_emp;

    // Plasticity update
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

    // LC Noradrenergic Sample Weight
    double w_sample = 1.0 + 0.60 * std::log(1.0 + delta_t) + 0.60 * std::abs(delta_rpe);
    sample_weight_vec[t] = w_sample;

    z_GC_prev = z_GC_curr;
    V_prev = V_curr;
  }

  return List::create(
    Named("Choice_LogLik") = -total_choice_nll,
    Named("Log_Lik_Choice_Vec") = log_lik_choice_vec,
    Named("RT_Base_Vec") = rt_base_vec,
    Named("Spatial_Entropy_Vec") = spatial_entropy_vec,
    Named("Sample_Weight_Vec") = sample_weight_vec,
    Named("Z_Point_Matrix") = z_point_matrix,
    Named("Z_Shunted_Matrix") = z_shunted_matrix,
    Named("C_Comp_Matrix") = c_comp_matrix
  );
}
