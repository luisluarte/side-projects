#include <Rcpp.h>
using namespace Rcpp;

inline double pnorm_c(double x) { return R::pnorm(x, 0.0, 1.0, 1, 0); }
inline double dnorm_c(double x) { return R::dnorm(x, 0.0, 1.0, 0); }

inline double lba_F(double t, double A, double b, double v, double s) {
    if (t <= 0.0) return 0.0;
    double term1 = ((b - A - v * t) / A) * pnorm_c((b - A - v * t) / (s * t));
    double term2 = ((b - v * t) / A) * pnorm_c((b - v * t) / (s * t));
    double term3 = ((s * t) / A) * dnorm_c((b - A - v * t) / (s * t));
    double term4 = ((s * t) / A) * dnorm_c((b - v * t) / (s * t));
    double res = 1.0 + term1 - term2 + term3 - term4;
    return (res < 0.0) ? 0.0 : ((res > 1.0) ? 1.0 : res);
}

inline double lba_f(double t, double A, double b, double v, double s) {
    if (t <= 0.0) return 0.0;
    double term1 = -v * pnorm_c((b - A - v * t) / (s * t));
    double term2 = s * dnorm_c((b - A - v * t) / (s * t));
    double term3 = v * pnorm_c((b - v * t) / (s * t));
    double term4 = -s * dnorm_c((b - v * t) / (s * t));
    double res = (1.0 / A) * (term1 + term2 + term3 + term4);
    return (res < 1e-10) ? 1e-10 : res;
}

// [[Rcpp::export]]
NumericVector extract_baseline_lba(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt, bool return_ll) {
    double b_base = std::exp(phi[0]) + 0.1;
    double A = std::exp(phi[1]);
    if (A > b_base - 0.1) A = b_base - 0.1; 
    double t_nd = 1.0 / (1.0 + std::exp(-phi[2]));
    double beta_v = std::exp(phi[3]);
    double alpha = 1.0 / (1.0 + std::exp(-phi[4]));
    double alpha_c = 1.0 / (1.0 + std::exp(-phi[5]));
    double rho = 1.0 / (1.0 + std::exp(-phi[6])) * 0.05; 
    double beta_str_b = phi[7]; // Striatum modifies threshold
    double s = 1.0; // scaling parameter fixed
    
    int T = resp.size();
    NumericVector out_vec(T);
    double Q[2] = {0.5, 0.5};
    double R_bar = 0.5;
    
    for (int t=0; t<T; ++t) {
        int ch = resp[t] - 1; 
        double R = (out[t] == 1) ? 1.0 : 0.0;
        
        double v1 = beta_v * Q[0];
        double v2 = beta_v * Q[1];
        
        double b = b_base + beta_str_b * (0.5 - R_bar);
        if (b < A + 0.1) b = A + 0.1;
        
        double t_decision = rt[t] - t_nd;
        if (return_ll) {
            double ll = 1e-10;
            if (t_decision > 0.0) {
                if (ch == 0) {
                    ll = lba_f(t_decision, A, b, v1, s) * (1.0 - lba_F(t_decision, A, b, v2, s));
                } else {
                    ll = lba_f(t_decision, A, b, v2, s) * (1.0 - lba_F(t_decision, A, b, v1, s));
                }
            }
            out_vec[t] = std::log(std::max(ll, 1e-10));
        } else {
            double chosen_v = (ch == 0) ? v1 : v2;
            out_vec[t] = t_nd + (b - A/2.0) / std::max(chosen_v, 0.1);
        }
        
        Q[ch] = Q[ch] + alpha * (R - Q[ch]);
        int unch = (ch == 1) ? 0 : 1;
        Q[unch] = Q[unch] + alpha_c * ((1.0 - R) - Q[unch]);
        R_bar = (1.0 - rho) * R_bar + rho * R;
    }
    return out_vec;
}

// [[Rcpp::export]]
inline double eval_baseline_lba(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    NumericVector ll = extract_baseline_lba(phi, resp, out, rt, true);
    double nll = 0.0;
    for(int i=0; i<ll.size(); ++i) nll -= 2.0 * ll[i];
    return (std::isnan(nll) || std::isinf(nll)) ? 1e9 : nll;
}

// [[Rcpp::export]]
NumericVector extract_topo_11(const std::vector<double>& phi, const std::vector<int>& genes, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt, const NumericVector& iti, const NumericVector& f_dur, bool return_ll) {
    // Topo 11: LBA Cortico-Cerebellar Race
    double alpha_ctx = 1.0 / (1.0 + std::exp(-phi[0]));
    double lambda_gc = std::exp(phi[1]);
    double b_base = std::exp(phi[2]) + 0.1;
    double A = std::exp(phi[3]);
    if (A > b_base - 0.1) A = b_base - 0.1;
    double t_nd = 1.0 / (1.0 + std::exp(-phi[4]));
    
    double beta_v_ctx = std::exp(phi[5]);
    double beta_v_cb = std::exp(phi[6]);
    
    double rho_str = 1.0 / (1.0 + std::exp(-phi[7])) * 0.05; 
    double beta_str_b = phi[8];
    double s = 1.0;
    
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
        
        // Linear Ballistic Race: Cortical value + Cerebellar Temporal Expectation
        double v1 = beta_v_ctx * Q_ctx[0];
        double v2 = beta_v_ctx * Q_ctx[1];
        if (genes[0] == 1) { v1 += beta_v_cb * Q_cb_1; }
        if (genes[1] == 1) { v2 += beta_v_cb * Q_cb_2; }
        
        // Striatum modulates threshold for macroscopic fatigue
        double b = b_base;
        if (genes[2] == 1) b += beta_str_b * (0.5 - Str_bar);
        if (b < A + 0.1) b = A + 0.1;
        
        double t_decision = rt[t] - t_nd;
        if (return_ll) {
            double ll = 1e-10;
            if (t_decision > 0.0) {
                if (ch == 0) {
                    ll = lba_f(t_decision, A, b, v1, s) * (1.0 - lba_F(t_decision, A, b, v2, s));
                } else {
                    ll = lba_f(t_decision, A, b, v2, s) * (1.0 - lba_F(t_decision, A, b, v1, s));
                }
            }
            out_vec[t] = std::log(std::max(ll, 1e-10));
        } else {
            double chosen_v = (ch == 0) ? v1 : v2;
            out_vec[t] = t_nd + (b - A/2.0) / std::max(chosen_v, 0.1);
        }
        
        double RPE_ctx = R - Q_ctx[ch];
        Q_ctx[ch] += alpha_ctx * RPE_ctx;
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
inline double eval_topo_11(const std::vector<double>& phi, const std::vector<int>& genes, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt, const NumericVector& iti, const NumericVector& f_dur) {
    NumericVector ll = extract_topo_11(phi, genes, resp, out, rt, iti, f_dur, true);
    double nll = 0.0;
    for(int i=0; i<ll.size(); ++i) nll -= 2.0 * ll[i];
    return (std::isnan(nll) || std::isinf(nll)) ? 1e9 : nll;
}
