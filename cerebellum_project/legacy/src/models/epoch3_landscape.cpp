#include <Rcpp.h>
#include "shared_utils.h"
using namespace Rcpp;

inline double calc_expected_rt_dist_t(double v, double a, double w, double t_nd) {
    double z = w * a;
    if (std::abs(v) < 1e-4) return t_nd + z * (a - z);
    return t_nd + (z / v) - (a / v) * ((std::exp(2.0 * v * z) - 1.0) / (std::exp(2.0 * v * a) - 1.0));
}

// [[Rcpp::export]]
NumericVector extract_baseline_5(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt, bool return_ll) {
    double a = std::exp(phi[0]);
    double t_nd = 1.0 / (1.0 + std::exp(-phi[1]));
    double beta_v = std::exp(phi[2]);
    double alpha = 1.0 / (1.0 + std::exp(-phi[3]));
    double alpha_c = 1.0 / (1.0 + std::exp(-phi[4]));
    double theta_ctx = std::exp(phi[5]); 
    double beta_a = std::exp(phi[6]);    
    double rho = 1.0 / (1.0 + std::exp(-phi[7])) * 0.05; 
    double beta_R_v = phi[8];
    double beta_R_a = phi[9];
    
    int T = resp.size();
    NumericVector out_vec(T);
    double Q[2] = {0.5, 0.5};
    double R_bar = 0.5;
    
    for (int t=0; t<T; ++t) {
        int ch = resp[t] - 1; 
        double R = (out[t] == 1) ? 1.0 : 0.0;
        double U_base = beta_v * (Q[1] - Q[0]);
        double v = U_base * std::exp(beta_R_v * (0.5 - R_bar));
        double safe_v = std::abs(v) < 1e-4 ? (v >= 0 ? 1e-4 : -1e-4) : v;
        double prev_rt = (t == 0) ? 0.5 : rt[t-1];
        double w_bias = 0.5 + 0.45 * std::tanh(theta_ctx * prev_rt);
        double a_t = a + beta_a * std::tanh(theta_ctx * prev_rt) + beta_R_a * (0.5 - R_bar);
        if (a_t < 0.01) a_t = 0.01;
        if (a_t > 10.0) a_t = 10.0;
        int w_choice = (resp[t] == 2) ? 1 : 2;
        
        if (return_ll) {
            out_vec[t] = std::log(wiener_pdf_w(rt[t], w_choice, safe_v, a_t, t_nd, w_bias));
        } else {
            out_vec[t] = calc_expected_rt_dist_t(safe_v, a_t, w_bias, t_nd);
        }
        
        Q[ch] = Q[ch] + alpha * (R - Q[ch]);
        int unch = (ch == 1) ? 0 : 1;
        Q[unch] = Q[unch] + alpha_c * ((1.0 - R) - Q[unch]);
        R_bar = (1.0 - rho) * R_bar + rho * R;
    }
    return out_vec;
}

// [[Rcpp::export]]
inline double eval_baseline_5(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    NumericVector ll = extract_baseline_5(phi, resp, out, rt, true);
    double nll = 0.0;
    for(int i=0; i<ll.size(); ++i) nll -= 2.0 * ll[i];
    return (std::isnan(nll) || std::isinf(nll)) ? 1e9 : nll;
}

// [[Rcpp::export]]
NumericVector extract_topo_9(const std::vector<double>& phi, const std::vector<int>& genes, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt, const NumericVector& iti, const NumericVector& f_dur, bool return_ll) {
    // Topo 9: Dual-Pathway Predictive Integrator
    double alpha_ctx = 1.0 / (1.0 + std::exp(-phi[0]));
    double lambda_gc = std::exp(phi[1]);
    double kappa_v = std::exp(phi[2]);
    double a_base = std::exp(phi[3]);
    double t_nd = 1.0 / (1.0 + std::exp(-phi[4]));
    double beta_a = std::exp(phi[5]);
    double u = phi[6] * 0.001; 
    
    // Striatal (Basal Ganglia) Slow Integration
    double rho_str = 1.0 / (1.0 + std::exp(-phi[7])) * 0.05; 
    double beta_str_a = phi[8];
    double beta_str_v = phi[9];
    
    // Cerebellar bias weights
    double theta_cb_bias = phi[10]; 
    
    int N_MF = 5;
    double a_tau = 0.05, b_tau = 5.0;
    std::vector<double> tau_k(N_MF);
    for(int k=0; k<N_MF; ++k) tau_k[k] = a_tau * std::pow(b_tau / a_tau, (double)k / (N_MF - 1.0));
    
    double Q_ctx[2] = {0.5, 0.5};
    std::vector<double> mf_state(N_MF, 0.0);
    std::vector<double> w_gc1(N_MF, 0.0), w_gc2(N_MF, 0.0);
    double Str_bar = 0.5;
    
    int T = resp.size();
    NumericVector out_vec(T);
    
    for (int t = 0; t < T; ++t) {
        int ch = resp[t] - 1; 
        double R = (out[t] == 1) ? 1.0 : 0.0;
        
        if (iti[t] > 0.01) {
            for(int k=0; k<N_MF; ++k) mf_state[k] = mf_state[k] * std::exp(-iti[t] / tau_k[k]);
            double decay_gc = std::exp(-lambda_gc * iti[t]);
            for (int k=0; k<N_MF; ++k) { w_gc1[k] *= decay_gc; w_gc2[k] *= decay_gc; }
        }
        
        double Q_cb_1 = 0.0, Q_cb_2 = 0.0;
        for (int k=0; k<N_MF; ++k) {
            Q_cb_1 += w_gc1[k] * mf_state[k];
            Q_cb_2 += w_gc2[k] * mf_state[k];
        }
        
        double delta_Q_ctx = Q_ctx[1] - Q_ctx[0];
        double delta_Q_cb = Q_cb_2 - Q_cb_1;
        
        // Cortex drives drift, Striatum explicitly dilates/contracts drift for macroscopic fatigue (Gate 2 support)
        double v_effective = kappa_v * delta_Q_ctx;
        v_effective *= std::exp(beta_str_v * (0.5 - Str_bar)); 
        double safe_v = std::abs(v_effective) < 1e-4 ? (v_effective >= 0 ? 1e-4 : -1e-4) : v_effective;
        
        double conflict = std::abs(Q_ctx[1] - Q_cb_2) + std::abs(Q_ctx[0] - Q_cb_1); 
        
        // Striatum explicitly widens boundary to capture long-tail fatigue variance (Gate 2 support)
        double a_effective = a_base + beta_a * conflict;
        a_effective += beta_str_a * (0.5 - Str_bar);
        
        if (a_effective < 0.01) a_effective = 0.01;
        if (a_effective > 10.0) a_effective = 10.0;
        
        // Cerebellum acts strictly as the starting point forward model (Gate 1 winner)
        double w_bias = 0.5 + 0.45 * std::tanh(theta_cb_bias * delta_Q_cb);
        
        int w_choice = (resp[t] == 2) ? 1 : 2;
        if (return_ll) {
            out_vec[t] = std::log(wiener_pdf_w(rt[t], w_choice, safe_v, a_effective, t_nd, w_bias));
        } else {
            out_vec[t] = calc_expected_rt_dist_t(safe_v, a_effective, w_bias, t_nd);
        }
        
        double RPE_ctx = R - Q_ctx[ch];
        Q_ctx[ch] += alpha_ctx * RPE_ctx;
        
        // Striatum updates slow macroscopic state
        Str_bar = (1.0 - rho_str) * Str_bar + rho_str * R;
        
        if (f_dur[t] > 0.01) {
            double E_cb1 = (ch == 0) ? (R - Q_cb_1) : 0.0;
            double E_cb2 = (ch == 1) ? (R - Q_cb_2) : 0.0;
            for(int k=0; k<N_MF; ++k) mf_state[k] = R + (mf_state[k] - R) * std::exp(-f_dur[t] / tau_k[k]);
            double decay_gc = std::exp(-lambda_gc * f_dur[t]);
            double int_gc = (1.0 - decay_gc) / (lambda_gc + 1e-8);
            for (int k=0; k<N_MF; ++k) {
                w_gc1[k] = w_gc1[k] * decay_gc + 0.5 * E_cb1 * int_gc * mf_state[k];
                w_gc2[k] = w_gc2[k] * decay_gc + 0.5 * E_cb2 * int_gc * mf_state[k];
            }
        }
    }
    return out_vec;
}

// [[Rcpp::export]]
inline double eval_topo_9(const std::vector<double>& phi, const std::vector<int>& genes, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt, const NumericVector& iti, const NumericVector& f_dur) {
    NumericVector ll = extract_topo_9(phi, genes, resp, out, rt, iti, f_dur, true);
    double nll = 0.0;
    for(int i=0; i<ll.size(); ++i) nll -= 2.0 * ll[i];
    return (std::isnan(nll) || std::isinf(nll)) ? 1e9 : nll;
}
