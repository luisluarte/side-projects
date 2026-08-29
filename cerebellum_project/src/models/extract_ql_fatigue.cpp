#include <Rcpp.h>
#include "shared_utils.h"
using namespace Rcpp;

inline double calc_expected_rt_dist_ql_fatigue(double v, double a, double w, double t_nd) {
    double z = w * a;
    if (std::abs(v) < 1e-4) {
        return t_nd + z * (a - z);
    } else {
        return t_nd + (z / v) - (a / v) * ((std::exp(2.0 * v * z) - 1.0) / (std::exp(2.0 * v * a) - 1.0));
    }
}

// [[Rcpp::export]]
NumericVector extract_ll_ql_dynamic_poly_fatigue(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    double a = std::exp(phi[0]);
    double t_nd = 1.0 / (1.0 + std::exp(-phi[1]));
    double beta_v = std::exp(phi[2]);
    double alpha = 1.0 / (1.0 + std::exp(-phi[3]));
    double alpha_c = 1.0 / (1.0 + std::exp(-phi[4]));
    double theta_ctx = std::exp(phi[5]); 
    double beta_a = std::exp(phi[6]);    
    double theta_v = std::exp(phi[7]); 
    double u = phi[8] * 0.001;
    
    int T = resp.size();
    NumericVector ll(T);
    double Q[2] = {0.5, 0.5};
    
    for (int t=0; t<T; ++t) {
        int ch = resp[t] - 1; 
        double R = (out[t] == 1) ? 1.0 : 0.0;
        
        double U_base = beta_v * (Q[1] - Q[0]);
        double v = (U_base >= 0 ? 1.0 : -1.0) * std::pow(std::abs(U_base), theta_v);
        double safe_v = std::abs(v) < 1e-4 ? (v >= 0 ? 1e-4 : -1e-4) : v;
        
        double prev_rt = (t == 0) ? 0.5 : rt[t-1];
        double w_bias = 0.5 + 0.45 * std::tanh(theta_ctx * prev_rt);
        double a_t = (a + beta_a * std::tanh(theta_ctx * prev_rt)) * std::exp(-u * (double)t);
        if (a_t < 0.01) a_t = 0.01;
        if (a_t > 10.0) a_t = 10.0;
        
        int w_choice = (resp[t] == 2) ? 1 : 2;
        ll[t] = std::log(wiener_pdf_w(rt[t], w_choice, safe_v, a_t, t_nd, w_bias));
        
        Q[ch] = Q[ch] + alpha * (R - Q[ch]);
        int unch = (ch == 1) ? 0 : 1;
        Q[unch] = Q[unch] + alpha_c * ((1.0 - R) - Q[unch]);
    }
    return ll;
}

// [[Rcpp::export]]
NumericVector extract_rt_ql_dynamic_poly_fatigue(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    double a = std::exp(phi[0]);
    double t_nd = 1.0 / (1.0 + std::exp(-phi[1]));
    double beta_v = std::exp(phi[2]);
    double alpha = 1.0 / (1.0 + std::exp(-phi[3]));
    double alpha_c = 1.0 / (1.0 + std::exp(-phi[4]));
    double theta_ctx = std::exp(phi[5]); 
    double beta_a = std::exp(phi[6]);    
    double theta_v = std::exp(phi[7]); 
    double u = phi[8] * 0.001;
    
    int T = resp.size();
    NumericVector pred_rt(T);
    double Q[2] = {0.5, 0.5};
    
    for (int t=0; t<T; ++t) {
        int ch = resp[t] - 1; 
        double R = (out[t] == 1) ? 1.0 : 0.0;
        
        double U_base = beta_v * (Q[1] - Q[0]);
        double v = (U_base >= 0 ? 1.0 : -1.0) * std::pow(std::abs(U_base), theta_v);
        double safe_v = std::abs(v) < 1e-4 ? (v >= 0 ? 1e-4 : -1e-4) : v;
        
        double prev_rt = (t == 0) ? 0.5 : rt[t-1];
        double w_bias = 0.5 + 0.45 * std::tanh(theta_ctx * prev_rt);
        double a_t = (a + beta_a * std::tanh(theta_ctx * prev_rt)) * std::exp(-u * (double)t);
        if (a_t < 0.01) a_t = 0.01;
        if (a_t > 10.0) a_t = 10.0;
        
        pred_rt[t] = calc_expected_rt_dist_ql_fatigue(safe_v, a_t, w_bias, t_nd);
        
        Q[ch] = Q[ch] + alpha * (R - Q[ch]);
        int unch = (ch == 1) ? 0 : 1;
        Q[unch] = Q[unch] + alpha_c * ((1.0 - R) - Q[unch]);
    }
    return pred_rt;
}
