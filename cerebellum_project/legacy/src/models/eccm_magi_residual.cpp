#include <Rcpp.h>
#include "shared_utils.h"
using namespace Rcpp;

inline double calc_expected_rt_dist_m(double v, double a, double w, double t_nd) {
    double z = w * a;
    if (std::abs(v) < 1e-4) {
        return t_nd + z * (a - z);
    } else {
        return t_nd + (z / v) - (a / v) * ((std::exp(2.0 * v * z) - 1.0) / (std::exp(2.0 * v * a) - 1.0));
    }
}

// [[Rcpp::export]]
NumericVector extract_magi_residual(const std::vector<double>& phi, const std::vector<int>& genes, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt, const NumericVector& iti, const NumericVector& f_dur, const NumericVector& ttp, bool return_ll) {
    
    double alpha_ctx_base = 1.0 / (1.0 + std::exp(-phi[0]));
    double eta_gc_base = 0.5 + std::exp(phi[1]);
    double lambda_gc = std::exp(phi[2]);
    double theta_cb = std::exp(phi[3]);
    double kappa_v = std::exp(phi[4]);
    double gamma_suppress = std::exp(phi[5]);
    double a_base = std::exp(phi[6]);
    double tau_nd = 1.0 / (1.0 + std::exp(-phi[7]));
    double beta_a = std::exp(phi[8]);
    double u = phi[9] * 0.001; 
    double alpha_vol = 1.0 / (1.0 + std::exp(-phi[10]));
    double beta_vol_a = std::exp(phi[11]);
    double beta_vol_v = std::exp(phi[12]);
    double beta_ph_cb = std::exp(phi[13]);
    double beta_ph_ctx = std::exp(phi[14]);
    
    int G1 = genes[0]; 
    int G2 = genes[1]; 
    int G3 = genes[2]; 
    int G4 = genes[3]; 
    int N_MF = genes[4]; 
    int G6 = genes[5]; 
    
    double a_tau = 0.05, b_tau = 5.0;
    std::vector<double> tau_k(N_MF);
    for(int k=0; k<N_MF; ++k) tau_k[k] = a_tau * std::pow(b_tau / a_tau, (double)k / (N_MF - 1.0));
    
    double l_gc_eff = lambda_gc + 1e-8;
    double l_mli_eff = (lambda_gc * 1.5) + 1e-8;
    double inv_l_gc_eff = 1.0 / l_gc_eff;
    double inv_l_mli_eff = 1.0 / l_mli_eff;

    double Q_ctx[2] = {0.5, 0.5};
    std::vector<double> mf_state(N_MF, 0.0);
    std::vector<double> w_gc1(N_MF, 0.0), w_gc2(N_MF, 0.0);
    std::vector<double> w_mli1(N_MF, 0.0), w_mli2(N_MF, 0.0);
    
    int T = resp.size();
    NumericVector out_vec(T);
    double Omega_t = 0.0;
    
    for (int t = 0; t < T; ++t) {
        int ch = resp[t] - 1; 
        double reward = (out[t] == 1) ? 1.0 : 0.0;
        
        if (iti[t] > 0.01) {
            for(int k=0; k<N_MF; ++k) mf_state[k] = mf_state[k] * std::exp(-iti[t] / tau_k[k]);
            double decay_gc_iti = std::exp(-lambda_gc * iti[t]);
            double decay_mli_iti = std::exp(-(lambda_gc * 1.5) * iti[t]);
            for (int k=0; k<N_MF; ++k) {
                w_gc1[k] *= decay_gc_iti; w_gc2[k] *= decay_gc_iti;
                w_mli1[k] *= decay_mli_iti; w_mli2[k] *= decay_mli_iti;
            }
        }
        
        double Q_cb_1 = 0.0, Q_cb_2 = 0.0;
        for (int k=0; k<N_MF; ++k) {
            Q_cb_1 += w_gc1[k] * mf_state[k] - w_mli1[k] * mf_state[k];
            Q_cb_2 += w_gc2[k] * mf_state[k] - w_mli2[k] * mf_state[k];
        }
        
        double U_1 = Q_ctx[0] + Q_cb_1;
        double U_2 = Q_ctx[1] + Q_cb_2;
        double delta_U = U_2 - U_1;
        
        double delta_Q_ctx = Q_ctx[1] - Q_ctx[0];
        double delta_Q_cb = Q_cb_2 - Q_cb_1;
        
        double w_bias = 0.5 + 0.45 * std::tanh(theta_cb * delta_Q_cb);
        
        double conflict = 0.0;
        if (G4 == 0) {
            conflict = 0.5 * (1.0 - std::tanh(10.0 * delta_Q_ctx) * std::tanh(10.0 * delta_Q_cb));
        } else {
            double RPE_ctx_approx = reward - Q_ctx[ch];
            double RPE_cb_approx = reward - ((ch == 1) ? Q_cb_2 : Q_cb_1);
            conflict = std::tanh(std::abs(RPE_ctx_approx) + std::abs(RPE_cb_approx));
        }
        
        double v_effective = kappa_v * delta_U;
        if (G2 == 1) {
            v_effective *= std::exp(-beta_vol_v * Omega_t);
        } else {
            v_effective *= std::exp(-gamma_suppress * conflict);
        }
        double safe_v = std::abs(v_effective) < 1e-4 ? (v_effective >= 0 ? 1e-4 : -1e-4) : v_effective;
        
        double t_elapsed = ttp[t] / 1000.0;
        double a_effective = a_base + beta_a * conflict;
        if (G1 == 1) a_effective += beta_vol_a * Omega_t;
        a_effective *= std::exp(-u * t_elapsed);
        
        if (a_effective < 0.01) a_effective = 0.01;
        if (a_effective > 10.0) a_effective = 10.0;
        
        int w_choice = (resp[t] == 2) ? 1 : 2;
        
        if (return_ll) {
            out_vec[t] = std::log(wiener_pdf_w(rt[t], w_choice, safe_v, a_effective, tau_nd, w_bias));
        } else {
            out_vec[t] = calc_expected_rt_dist_m(safe_v, a_effective, w_bias, tau_nd);
        }
        
        double RPE_ctx = reward - Q_ctx[ch];
        double cb_pred = (ch == 1) ? Q_cb_2 : Q_cb_1;
        double RPE_cb = reward - (Q_ctx[ch] + cb_pred); 
        
        double current_alpha_ctx = alpha_ctx_base;
        if (G6 == 1) current_alpha_ctx = alpha_ctx_base + beta_ph_ctx * Omega_t;
        if (current_alpha_ctx > 1.0) current_alpha_ctx = 1.0;
        
        Q_ctx[ch] += current_alpha_ctx * RPE_ctx;
        
        Omega_t = (1.0 - alpha_vol) * Omega_t + alpha_vol * std::abs(RPE_cb);
        
        double E_cb1 = (ch == 0) ? RPE_cb : 0.0;
        double E_cb2 = (ch == 1) ? RPE_cb : 0.0;
        
        double current_eta_gc = eta_gc_base;
        if (G3 == 1) current_eta_gc += beta_ph_cb * Omega_t;
        double eta_mli = current_eta_gc;
        
        if (f_dur[t] > 0.01) {
            for(int k=0; k<N_MF; ++k) mf_state[k] = reward + (mf_state[k] - reward) * std::exp(-f_dur[t] / tau_k[k]);
            double decay_gc_f = std::exp(-l_gc_eff * f_dur[t]);
            double decay_mli_f = std::exp(-l_mli_eff * f_dur[t]);
            double int_gc_f = (1.0 - decay_gc_f) * inv_l_gc_eff;
            double int_mli_f = (1.0 - decay_mli_f) * inv_l_mli_eff;
            double scale_gc1 = current_eta_gc * E_cb1 * int_gc_f, scale_gc2 = current_eta_gc * E_cb2 * int_gc_f;
            double scale_mli1 = -eta_mli * E_cb1 * int_mli_f, scale_mli2 = -eta_mli * E_cb2 * int_mli_f;
            for (int k=0; k<N_MF; ++k) {
                w_gc1[k] = w_gc1[k] * decay_gc_f + scale_gc1 * mf_state[k];
                w_gc2[k] = w_gc2[k] * decay_gc_f + scale_gc2 * mf_state[k];
                w_mli1[k] = w_mli1[k] * decay_mli_f + scale_mli1 * mf_state[k];
                w_mli2[k] = w_mli2[k] * decay_mli_f + scale_mli2 * mf_state[k];
            }
        }
    }
    return out_vec;
}

// [[Rcpp::export]]
inline double eval_magi_wrapper(const std::vector<double>& phi, const std::vector<int>& genes, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt, const NumericVector& iti, const NumericVector& f_dur, const NumericVector& ttp) {
    NumericVector ll = extract_magi_residual(phi, genes, resp, out, rt, iti, f_dur, ttp, true);
    double nll = 0.0;
    for(int i=0; i<ll.size(); ++i) nll -= 2.0 * ll[i];
    return (std::isnan(nll) || std::isinf(nll)) ? 1e9 : nll;
}
