#include <Rcpp.h>
#include "shared_utils.h"

using namespace Rcpp;

inline std::vector<double> exact_mf_step2(double dt, const std::vector<double>& mf, double tau_m, double I_drive, int N_MF) {
    std::vector<double> mf_next(N_MF, 0.0);
    std::vector<double> d(N_MF, 0.0);
    for (int i=0; i<N_MF; ++i) d[i] = mf[i] - I_drive;
    
    double x = dt / tau_m;
    double decay = std::exp(-x);

    std::vector<double> w(N_MF, 0.0);
    w[0] = 1.0;
    for (int i=1; i<N_MF; ++i) {
        w[i] = w[i-1] * x / (double)i;
    }

    for (int k=0; k<N_MF; ++k) {
        double conv_sum = 0.0;
        for (int j=0; j<=k; ++j) {
            conv_sum += d[k - j] * w[j];
        }
        mf_next[k] = I_drive + conv_sum * decay;
    }
    return mf_next;
}

inline double calc_expected_rt_seq(double v, double a, double w, double t_nd) {
    double z = w * a;
    if (std::abs(v) < 1e-4) {
        return t_nd + z * (a - z);
    } else {
        return t_nd + (z / v) - (a / v) * ((std::exp(2.0 * v * z) - 1.0) / (std::exp(2.0 * v * a) - 1.0));
    }
}

// [[Rcpp::export]]
inline double eval_bvk_full_gating_seq(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt, const NumericVector& iti, const NumericVector& f_dur, int N_MF) {
    double alpha_ctx = 1.0 / (1.0 + std::exp(-phi[0]));
    double tau_m = 1.0 + std::exp(phi[1]);
    double eta_gc = 0.5 + std::exp(phi[2]);
    double lambda_gc = std::exp(phi[3]);
    double theta_cb = std::exp(phi[4]);
    double kappa_ctx = std::exp(phi[5]);
    double gamma_suppress = std::exp(phi[6]);
    double a_s = std::exp(phi[7]);
    double tau_nd = 1.0 / (1.0 + std::exp(-phi[8]));
    double beta_a = std::exp(phi[9]);
    double kappa_cb = std::exp(phi[10]);
    
    double l_gc_eff = lambda_gc + 1e-8;
    double l_mli_eff = (lambda_gc * 1.5) + 1e-8;
    double eta_mli = eta_gc;
    
    double inv_l_gc_eff = 1.0 / l_gc_eff;
    double inv_l_mli_eff = 1.0 / l_mli_eff;

    double Q_ctx[2] = {0.5, 0.5};
    std::vector<double> mf_state(N_MF, 0.0);
    std::vector<double> w_gc1(N_MF, 0.0), w_gc2(N_MF, 0.0);
    std::vector<double> w_mli1(N_MF, 0.0), w_mli2(N_MF, 0.0);
    
    std::vector<double> D_vec;
    
    for (int t = 0; t < resp.size(); ++t) {
        int ch = resp[t] - 1; 
        double reward = (out[t] == 1) ? 1.0 : 0.0;
        
        if (iti[t] > 0.01) {
            mf_state = exact_mf_step2(iti[t], mf_state, tau_m, 0.0, N_MF);
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
        
        double delta_Q_ctx = Q_ctx[1] - Q_ctx[0];
        double delta_Q_cb = Q_cb_2 - Q_cb_1;
        
        double w_bias = 0.5 + 0.45 * std::tanh(theta_cb * delta_Q_cb);
        double conflict = 0.5 * (1.0 - std::tanh(10.0 * delta_Q_ctx) * std::tanh(10.0 * delta_Q_cb));
        
        double v_base = kappa_ctx * delta_Q_ctx + kappa_cb * delta_Q_cb;
        double v_effective = v_base * std::exp(-gamma_suppress * conflict);
        double safe_v = std::abs(v_effective) < 1e-4 ? (v_effective >= 0 ? 1e-4 : -1e-4) : v_effective;
        
        double a_effective = a_s + beta_a * conflict;
        if (a_effective < 0.01) a_effective = 0.01;
        if (a_effective > 10.0) a_effective = 10.0;
        
        int w_choice = (resp[t] == 2) ? 1 : 2;
        D_vec.push_back(-2.0 * std::log(wiener_pdf_w(rt[t], w_choice, safe_v, a_effective, tau_nd, w_bias)));
        
        double RPE_ctx = reward - Q_ctx[ch];
        Q_ctx[ch] += alpha_ctx * RPE_ctx;
        double cb_pred = (ch == 1) ? Q_cb_2 : Q_cb_1;
        double RPE_cb = reward - cb_pred;
        double E_cb1 = (ch == 0) ? RPE_cb : 0.0;
        double E_cb2 = (ch == 1) ? RPE_cb : 0.0;
        
        if (f_dur[t] > 0.01) {
            mf_state = exact_mf_step2(f_dur[t], mf_state, tau_m, reward, N_MF);
            double decay_gc_f = std::exp(-l_gc_eff * f_dur[t]);
            double decay_mli_f = std::exp(-l_mli_eff * f_dur[t]);
            double int_gc_f = (1.0 - decay_gc_f) * inv_l_gc_eff;
            double int_mli_f = (1.0 - decay_mli_f) * inv_l_mli_eff;
            double scale_gc1 = eta_gc * E_cb1 * int_gc_f, scale_gc2 = eta_gc * E_cb2 * int_gc_f;
            double scale_mli1 = -eta_mli * E_cb1 * int_mli_f, scale_mli2 = -eta_mli * E_cb2 * int_mli_f;
            for (int k=0; k<N_MF; ++k) {
                w_gc1[k] = w_gc1[k] * decay_gc_f + scale_gc1 * mf_state[k];
                w_gc2[k] = w_gc2[k] * decay_gc_f + scale_gc2 * mf_state[k];
                w_mli1[k] = w_mli1[k] * decay_mli_f + scale_mli1 * mf_state[k];
                w_mli2[k] = w_mli2[k] * decay_mli_f + scale_mli2 * mf_state[k];
            }
        }
    }
    double res = calc_pen_ll(D_vec);
    return (std::isnan(res) || std::isinf(res)) ? 1e9 : res;
}

// [[Rcpp::export]]
NumericVector extract_rt_bvk_full_gating_seq(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt, const NumericVector& iti, const NumericVector& f_dur, int N_MF) {
    double alpha_ctx = 1.0 / (1.0 + std::exp(-phi[0]));
    double tau_m = 1.0 + std::exp(phi[1]);
    double eta_gc = 0.5 + std::exp(phi[2]);
    double lambda_gc = std::exp(phi[3]);
    double theta_cb = std::exp(phi[4]);
    double kappa_ctx = std::exp(phi[5]);
    double gamma_suppress = std::exp(phi[6]);
    double a_s = std::exp(phi[7]);
    double tau_nd = 1.0 / (1.0 + std::exp(-phi[8]));
    double beta_a = std::exp(phi[9]);
    double kappa_cb = std::exp(phi[10]);
    
    double l_gc_eff = lambda_gc + 1e-8;
    double l_mli_eff = (lambda_gc * 1.5) + 1e-8;
    double eta_mli = eta_gc;
    double inv_l_gc_eff = 1.0 / l_gc_eff;
    double inv_l_mli_eff = 1.0 / l_mli_eff;

    double Q_ctx[2] = {0.5, 0.5};
    std::vector<double> mf_state(N_MF, 0.0);
    std::vector<double> w_gc1(N_MF, 0.0), w_gc2(N_MF, 0.0);
    std::vector<double> w_mli1(N_MF, 0.0), w_mli2(N_MF, 0.0);
    
    int T = resp.size();
    NumericVector pred_rt(T);
    
    for (int t = 0; t < T; ++t) {
        int ch = resp[t] - 1; 
        double reward = (out[t] == 1) ? 1.0 : 0.0;
        
        if (iti[t] > 0.01) {
            mf_state = exact_mf_step2(iti[t], mf_state, tau_m, 0.0, N_MF);
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
        
        double delta_Q_ctx = Q_ctx[1] - Q_ctx[0];
        double delta_Q_cb = Q_cb_2 - Q_cb_1;
        
        double w_bias = 0.5 + 0.45 * std::tanh(theta_cb * delta_Q_cb);
        double conflict = 0.5 * (1.0 - std::tanh(10.0 * delta_Q_ctx) * std::tanh(10.0 * delta_Q_cb));
        
        double v_base = kappa_ctx * delta_Q_ctx + kappa_cb * delta_Q_cb;
        double v_effective = v_base * std::exp(-gamma_suppress * conflict);
        double safe_v = std::abs(v_effective) < 1e-4 ? (v_effective >= 0 ? 1e-4 : -1e-4) : v_effective;
        
        double a_effective = a_s + beta_a * conflict;
        if (a_effective < 0.01) a_effective = 0.01;
        if (a_effective > 10.0) a_effective = 10.0;
        
        pred_rt[t] = calc_expected_rt_seq(safe_v, a_effective, w_bias, tau_nd);
        
        double RPE_ctx = reward - Q_ctx[ch];
        Q_ctx[ch] += alpha_ctx * RPE_ctx;
        double cb_pred = (ch == 1) ? Q_cb_2 : Q_cb_1;
        double RPE_cb = reward - cb_pred;
        double E_cb1 = (ch == 0) ? RPE_cb : 0.0;
        double E_cb2 = (ch == 1) ? RPE_cb : 0.0;
        
        if (f_dur[t] > 0.01) {
            mf_state = exact_mf_step2(f_dur[t], mf_state, tau_m, reward, N_MF);
            double decay_gc_f = std::exp(-l_gc_eff * f_dur[t]);
            double decay_mli_f = std::exp(-l_mli_eff * f_dur[t]);
            double int_gc_f = (1.0 - decay_gc_f) * inv_l_gc_eff;
            double int_mli_f = (1.0 - decay_mli_f) * inv_l_mli_eff;
            double scale_gc1 = eta_gc * E_cb1 * int_gc_f, scale_gc2 = eta_gc * E_cb2 * int_gc_f;
            double scale_mli1 = -eta_mli * E_cb1 * int_mli_f, scale_mli2 = -eta_mli * E_cb2 * int_mli_f;
            for (int k=0; k<N_MF; ++k) {
                w_gc1[k] = w_gc1[k] * decay_gc_f + scale_gc1 * mf_state[k];
                w_gc2[k] = w_gc2[k] * decay_gc_f + scale_gc2 * mf_state[k];
                w_mli1[k] = w_mli1[k] * decay_mli_f + scale_mli1 * mf_state[k];
                w_mli2[k] = w_mli2[k] * decay_mli_f + scale_mli2 * mf_state[k];
            }
        }
    }
    return pred_rt;
}
