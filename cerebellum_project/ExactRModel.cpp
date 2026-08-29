// ==============================================================================
// ExactRModel.cpp - Ultra-Precision Engine with Deep-Layer State Derivatives
// ==============================================================================
#include <RcppEigen.h>
#include <vector>
#include <cmath>
#include <algorithm>

// [[Rcpp::depends(RcppEigen)]]
using namespace Rcpp;
using namespace Eigen;

inline double clamp_val(double v, double lo, double hi) {
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}

// [[Rcpp::export]]
Rcpp::List run_exact_r_simulation_cpp(
    const NumericVector& resp_r,
    const NumericVector& outcome_r,
    const NumericVector& m1_r,
    const NumericVector& m2_r,
    const NumericVector& rt_emp_r,
    const NumericVector& theta_r
) {
    int N_trials = resp_r.size();
    
    // Parse Parameters
    double p_ws_base       = theta_r[0]; // Base Win-Stay logit
    double p_ls_base       = theta_r[1]; // Base Lose-Shift logit
    double w_mag_curr      = theta_r[2]; // Magnitude sensitivity current option
    double w_mag_alt       = theta_r[3]; // Magnitude sensitivity alternative option
    double alpha_q         = theta_r[4]; // Q-value learning rate
    double w_streak        = theta_r[5]; // Loss streak multiplier
    double w_purkinje_inh  = theta_r[6]; // Purkinje cross-inhibition weight
    double tau_kinematic   = theta_r[7]; // Kinematic filter time constant
    double beta_post_err   = theta_r[8]; // Post-error slowing
    double kappa_entropy   = theta_r[9]; // Shannon entropy dilation
    
    NumericVector choice_nll_vec(N_trials);
    NumericVector switch_labels(N_trials - 1);
    NumericVector switch_probs(N_trials - 1);
    NumericVector rt_preds(N_trials);
    NumericVector value_traj(N_trials);
    NumericVector uncert_traj(N_trials);
    NumericVector state_norm_traj(N_trials);
    NumericVector manifold_vel_traj(N_trials);
    NumericVector dkl_traj(N_trials);
    NumericVector eligibility_traj(N_trials);
    
    double total_choice_nll = 0.0;
    Vector2d Q_val(0.5, 0.5);
    
    // Subject baseline RT
    double mean_rt_sub = 0.0;
    for (int i = 0; i < N_trials; ++i) mean_rt_sub += rt_emp_r[i];
    mean_rt_sub /= std::max(1, N_trials);
    
    double rt_filter = mean_rt_sub;
    int loss_streak = 0;
    
    // Granular reservoir state vector z_GC in R^6
    VectorXd z_gc = VectorXd::Zero(6);
    VectorXd z_gc_prev = VectorXd::Zero(6);
    
    double prev_p1 = 0.50;
    double prev_p2 = 0.50;
    double e_pc = 0.50; // Purkinje eligibility trace
    
    for (int t = 0; t < N_trials; ++t) {
        int ch = static_cast<int>(resp_r[t]);
        int out = static_cast<int>(outcome_r[t]);
        double m1 = m1_r[t];
        double m2 = m2_r[t];
        double rt_e = rt_emp_r[t];
        
        int prev_ch = (t > 0) ? static_cast<int>(resp_r[t - 1]) : 1;
        int prev_out = (t > 0) ? static_cast<int>(outcome_r[t - 1]) : 1;
        
        // Contextual option magnitudes
        double m_curr = (prev_ch == 1) ? m1 : m2;
        double m_alt  = (prev_ch == 1) ? m2 : m1;
        double d_curr = (m_curr - 5.5) / 4.5;
        double d_diff = (m_alt - m_curr) / 4.0;
        
        // Observable morphisms
        double val_t = (prev_ch == 1) ? Q_val[0] : Q_val[1];
        double q_diff = Q_val[prev_ch - 1] - Q_val[2 - prev_ch];
        
        double p_stay = 0.50;
        double p_switch = 0.50;
        
        if (t == 0) {
            p_stay = 0.50;
            p_switch = 0.50;
            loss_streak = 0;
        } else {
            if (prev_out == 1) {
                // Rewarded: High Win-Stay conditioned on reward magnitude
                loss_streak = 0;
                double logit_stay = std::log(p_ws_base / (1.0 - p_ws_base)) + w_mag_curr * d_curr + 0.35 * q_diff;
                p_stay = 1.0 / (1.0 + std::exp(-logit_stay));
                p_stay = clamp_val(p_stay, 0.001, 0.999);
                p_switch = 1.0 - p_stay;
            } else {
                // Unrewarded: Loss-Shift conditioned on alternative option magnitude & streak
                loss_streak += 1;
                double streak_term = w_streak * std::log(1.0 + loss_streak);
                double logit_shift = std::log(p_ls_base / (1.0 - p_ls_base)) + w_mag_alt * d_diff + streak_term - 0.35 * q_diff;
                p_switch = 1.0 / (1.0 + std::exp(-logit_shift));
                p_switch = clamp_val(p_switch, 0.001, 0.999);
                p_stay = 1.0 - p_switch;
            }
        }
        
        double prob_1 = (prev_ch == 1) ? p_stay : p_switch;
        double prob_2 = 1.0 - prob_1;
        
        // Shannon Entropy Uncertainty
        double H_t = -(prob_1 * std::log(prob_1 + 1e-12) + prob_2 * std::log(prob_2 + 1e-12)) / std::log(2.0);
        double uncert_t = clamp_val(H_t, 0.0, 1.0);
        
        // Update 6-D Granular Reservoir State Vector
        z_gc_prev = z_gc;
        z_gc[0] = d_curr;                               // Context magnitude
        z_gc[1] = d_diff;                               // Alternative contrast
        z_gc[2] = Q_val[0];                             // Option 1 value
        z_gc[3] = Q_val[1];                             // Option 2 value
        z_gc[4] = std::log(1.0 + loss_streak);          // Calcium rebound streak trace
        z_gc[5] = (prev_ch == 1) ? 1.0 : -1.0;          // Efference copy trace
        
        double state_norm_t = z_gc.norm();
        
        // 1. Granular Manifold Velocity: || \dot{z}_GC,t ||_2
        double manifold_vel_t = (t > 0) ? (z_gc - z_gc_prev).norm() : 0.0;
        
        // 2. DCN Prior Divergence: D_KL( \pi_t || \pi_{t-1} )
        double dkl_t = 0.0;
        if (t > 0) {
            double p1_c = clamp_val(prob_1, 1e-12, 1.0 - 1e-12);
            double p2_c = clamp_val(prob_2, 1e-12, 1.0 - 1e-12);
            double prev_p1_c = clamp_val(prev_p1, 1e-12, 1.0 - 1e-12);
            double prev_p2_c = clamp_val(prev_p2, 1e-12, 1.0 - 1e-12);
            dkl_t = p1_c * std::log(p1_c / prev_p1_c) + p2_c * std::log(p2_c / prev_p2_c);
            dkl_t = std::max(0.0, dkl_t);
        }
        prev_p1 = prob_1;
        prev_p2 = prob_2;
        
        // 3. Purkinje Synaptic Eligibility Trace: E_PC,t
        double ctx_norm = std::abs(d_curr) + std::abs(d_diff);
        e_pc = 0.85 * e_pc + 0.15 * ctx_norm;
        
        value_traj[t] = val_t;
        uncert_traj[t] = uncert_t;
        state_norm_traj[t] = state_norm_t;
        manifold_vel_traj[t] = manifold_vel_t;
        dkl_traj[t] = dkl_t;
        eligibility_traj[t] = e_pc;
        
        double p_chosen = (ch == 1) ? prob_1 : prob_2;
        p_chosen = clamp_val(p_chosen, 1e-12, 1.0 - 1e-12);
        double nll_t = -std::log(p_chosen);
        choice_nll_vec[t] = nll_t;
        total_choice_nll += nll_t;
        
        if (t > 0) {
            switch_labels[t - 1] = (ch != prev_ch) ? 1.0 : 0.0;
            switch_probs[t - 1] = (prev_ch == 1) ? prob_2 : prob_1;
        }
        
        // Purkinje-DCN Kinematic Reaction Time Readout
        double post_err = (t > 0 && prev_out == 0) ? beta_post_err : 0.0;
        double mag_diff_abs = std::abs(m1 - m2) / 10.0;
        
        double rt_hat = tau_kinematic * rt_filter + (1.0 - tau_kinematic) * mean_rt_sub + post_err + kappa_entropy * (uncert_t - 0.5) - 0.03 * mag_diff_abs;
        rt_hat = clamp_val(rt_hat, 0.12, 2.90);
        rt_preds[t] = rt_hat;
        
        // Filter update
        rt_filter = 0.96 * rt_filter + 0.04 * rt_e;
        
        // Value Learning
        double reward = (out == 1) ? ((ch == 1) ? m1 : m2) / 10.0 : 0.0;
        double rpe = reward - Q_val[ch - 1];
        Q_val[ch - 1] += alpha_q * rpe;
    }
    
    return Rcpp::List::create(
        Rcpp::Named("Choice_NLL") = total_choice_nll,
        Rcpp::Named("Choice_NLL_Vec") = choice_nll_vec,
        Rcpp::Named("Switch_Labels") = switch_labels,
        Rcpp::Named("Switch_Probs") = switch_probs,
        Rcpp::Named("RT_Preds") = rt_preds,
        Rcpp::Named("RT_Emp") = rt_emp_r,
        Rcpp::Named("Value_Traj") = value_traj,
        Rcpp::Named("Uncertainty_Traj") = uncert_traj,
        Rcpp::Named("State_Norm_Traj") = state_norm_traj,
        Rcpp::Named("Manifold_Vel_Traj") = manifold_vel_traj,
        Rcpp::Named("DKL_Traj") = dkl_traj,
        Rcpp::Named("Eligibility_Traj") = eligibility_traj
    );
}
