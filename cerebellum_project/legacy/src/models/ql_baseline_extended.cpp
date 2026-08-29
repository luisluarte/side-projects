#include <Rcpp.h>
#include "shared_utils.h"
using namespace Rcpp;

inline double calc_expected_rt_dist_t(double v, double a, double w, double t_nd) {
    double z = w * a;
    if (std::abs(v) < 1e-4) return t_nd + z * (a - z);
    return t_nd + (z / v) - (a / v) * ((std::exp(2.0 * v * z) - 1.0) / (std::exp(2.0 * v * a) - 1.0));
}

// [[Rcpp::export]]
NumericVector extract_rt_ql_baseline_extended(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt, bool return_ll) {
    double a = std::exp(phi[0]);
    double t_nd = 1.0 / (1.0 + std::exp(-phi[1]));
    double beta_v = std::exp(phi[2]);
    double alpha = 1.0 / (1.0 + std::exp(-phi[3]));
    double alpha_c = 1.0 / (1.0 + std::exp(-phi[4]));
    double theta_ctx = std::exp(phi[5]); 
    double beta_a = std::exp(phi[6]);    
    double theta_v = std::exp(phi[7]); 
    double u = phi[8] * 0.001;
    
    // The requested polynomial parameters added to the drift rate magnitude
    double b_v1 = phi[9];
    double b_v2 = phi[10];
    double b_v3 = phi[11];
    
    int T = resp.size();
    NumericVector out_vec(T);
    double Q[2] = {0.5, 0.5};
    
    for (int t=0; t<T; ++t) {
        int ch = resp[t] - 1; 
        double R = (out[t] == 1) ? 1.0 : 0.0;
        
        double U_base = beta_v * (Q[1] - Q[0]);
        double v = (U_base >= 0 ? 1.0 : -1.0) * std::pow(std::abs(U_base), theta_v);
        
        // Add polynomial drift fatigue to the magnitude of v
        double norm_t = (double)t / (double)T;
        double poly_drift = b_v1 * norm_t + b_v2 * std::pow(norm_t, 2) + b_v3 * std::pow(norm_t, 3);
        
        double sign_v = (v >= 0) ? 1.0 : -1.0;
        v = v + sign_v * poly_drift; 
        
        double safe_v = std::abs(v) < 1e-4 ? (v >= 0 ? 1e-4 : -1e-4) : v;
        
        double prev_rt = (t == 0) ? 0.5 : rt[t-1];
        double w_bias = 0.5 + 0.45 * std::tanh(theta_ctx * prev_rt);
        double a_t = (a + beta_a * std::tanh(theta_ctx * prev_rt)) * std::exp(-u * (double)t);
        
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
    }
    return out_vec;
}

// [[Rcpp::export]]
inline double eval_ql_baseline_extended(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    NumericVector ll = extract_rt_ql_baseline_extended(phi, resp, out, rt, true);
    double nll = 0.0;
    for(int i=0; i<ll.size(); ++i) nll -= 2.0 * ll[i];
    return (std::isnan(nll) || std::isinf(nll)) ? 1e9 : nll;
}
