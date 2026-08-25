#include <Rcpp.h>
#include "shared_utils.h"

using namespace Rcpp;

// [[Rcpp::export]]
inline double eval_ql_ddm_dynamic(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    double a = std::exp(phi[0]);
    double t_nd = 1.0 / (1.0 + std::exp(-phi[1]));
    double beta_v = std::exp(phi[2]);
    double alpha = 1.0 / (1.0 + std::exp(-phi[3]));
    double alpha_c = 1.0 / (1.0 + std::exp(-phi[4]));
    double theta_ctx = std::exp(phi[5]); 
    double beta_a = std::exp(phi[6]);    
    
    std::vector<double> D_vec;
    double Q[2] = {0.5, 0.5};
    
    for (int t=0; t<resp.size(); ++t) {
        int ch = resp[t] - 1; 
        double R = (out[t] == 1) ? 1.0 : 0.0;
        
        double v = beta_v * (Q[1] - Q[0]); // Q[1] is Upper, Q[0] is Lower. Positive v = towards Upper.
        double safe_v = std::abs(v) < 1e-4 ? (v >= 0 ? 1e-4 : -1e-4) : v;
        
        double prev_rt = (t == 0) ? 0.5 : rt[t-1];
        double w_bias = 0.5 + 0.45 * std::tanh(theta_ctx * prev_rt); // >0.5 means bias towards Upper
        double a_t = a + beta_a * std::tanh(theta_ctx * prev_rt);
        if (a_t < 0.01) a_t = 0.01;
        if (a_t > 10.0) a_t = 10.0;
        
        // resp=2 is Upper, resp=1 is Lower. wiener_pdf_w expects choice=1 for Upper, choice=2 for Lower.
        int w_choice = (resp[t] == 2) ? 1 : 2;
        
        D_vec.push_back(-2.0 * std::log(wiener_pdf_w(rt[t], w_choice, safe_v, a_t, t_nd, w_bias)));
        
        Q[ch] = Q[ch] + alpha * (R - Q[ch]);
        int unch = (ch == 1) ? 0 : 1;
        Q[unch] = Q[unch] + alpha_c * ((1.0 - R) - Q[unch]);
    }
    double res = calc_pen_ll(D_vec);
    return (std::isnan(res) || std::isinf(res)) ? 1e9 : res;
}

// [[Rcpp::export]]
NumericVector extract_ll_ql_dynamic_point(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    double a = std::exp(phi[0]);
    double t_nd = 1.0 / (1.0 + std::exp(-phi[1]));
    double beta_v = std::exp(phi[2]);
    double alpha = 1.0 / (1.0 + std::exp(-phi[3]));
    double alpha_c = 1.0 / (1.0 + std::exp(-phi[4]));
    double theta_ctx = std::exp(phi[5]);
    double beta_a = std::exp(phi[6]);
    
    int T = resp.size();
    NumericVector ll(T);
    double Q[2] = {0.5, 0.5};
    
    for (int t=0; t<T; ++t) {
        int ch = resp[t] - 1; 
        double R = (out[t] == 1) ? 1.0 : 0.0;
        
        double v = beta_v * (Q[1] - Q[0]);
        double safe_v = std::abs(v) < 1e-4 ? (v >= 0 ? 1e-4 : -1e-4) : v;
        
        double prev_rt = (t == 0) ? 0.5 : rt[t-1];
        double w_bias = 0.5 + 0.45 * std::tanh(theta_ctx * prev_rt);
        double a_t = a + beta_a * std::tanh(theta_ctx * prev_rt);
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
inline double eval_ql_ddm_dynamic_poly(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    double a = std::exp(phi[0]);
    double t_nd = 1.0 / (1.0 + std::exp(-phi[1]));
    double beta_v = std::exp(phi[2]);
    double alpha = 1.0 / (1.0 + std::exp(-phi[3]));
    double alpha_c = 1.0 / (1.0 + std::exp(-phi[4]));
    double theta_ctx = std::exp(phi[5]); 
    double beta_a = std::exp(phi[6]);    
    double theta_v = std::exp(phi[7]); 
    
    std::vector<double> D_vec;
    double Q[2] = {0.5, 0.5};
    
    for (int t=0; t<resp.size(); ++t) {
        int ch = resp[t] - 1; 
        double R = (out[t] == 1) ? 1.0 : 0.0;
        
        double U_base = beta_v * (Q[1] - Q[0]);
        double v = (U_base >= 0 ? 1.0 : -1.0) * std::pow(std::abs(U_base), theta_v);
        double safe_v = std::abs(v) < 1e-4 ? (v >= 0 ? 1e-4 : -1e-4) : v;
        
        double prev_rt = (t == 0) ? 0.5 : rt[t-1];
        double w_bias = 0.5 + 0.45 * std::tanh(theta_ctx * prev_rt);
        double a_t = a + beta_a * std::tanh(theta_ctx * prev_rt);
        if (a_t < 0.01) a_t = 0.01;
        if (a_t > 10.0) a_t = 10.0;
        
        int w_choice = (resp[t] == 2) ? 1 : 2;
        
        D_vec.push_back(-2.0 * std::log(wiener_pdf_w(rt[t], w_choice, safe_v, a_t, t_nd, w_bias)));
        
        Q[ch] = Q[ch] + alpha * (R - Q[ch]);
        int unch = (ch == 1) ? 0 : 1;
        Q[unch] = Q[unch] + alpha_c * ((1.0 - R) - Q[unch]);
    }
    double res = calc_pen_ll(D_vec);
    return (std::isnan(res) || std::isinf(res)) ? 1e9 : res;
}

// [[Rcpp::export]]
NumericVector extract_ll_ql_dynamic_poly_point(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    double a = std::exp(phi[0]);
    double t_nd = 1.0 / (1.0 + std::exp(-phi[1]));
    double beta_v = std::exp(phi[2]);
    double alpha = 1.0 / (1.0 + std::exp(-phi[3]));
    double alpha_c = 1.0 / (1.0 + std::exp(-phi[4]));
    double theta_ctx = std::exp(phi[5]);
    double beta_a = std::exp(phi[6]);
    double theta_v = std::exp(phi[7]); 
    
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
        double a_t = a + beta_a * std::tanh(theta_ctx * prev_rt);
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
inline double eval_ql_ddm_dynamic_poly_fatigue(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    double a = std::exp(phi[0]);
    double t_nd = 1.0 / (1.0 + std::exp(-phi[1]));
    double beta_v = std::exp(phi[2]);
    double alpha = 1.0 / (1.0 + std::exp(-phi[3]));
    double alpha_c = 1.0 / (1.0 + std::exp(-phi[4]));
    double theta_ctx = std::exp(phi[5]); 
    double beta_a = std::exp(phi[6]);    
    double theta_v = std::exp(phi[7]); 
    double u = phi[8] * 0.001;
    
    std::vector<double> D_vec;
    double Q[2] = {0.5, 0.5};
    
    for (int t=0; t<resp.size(); ++t) {
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
        
        D_vec.push_back(-2.0 * std::log(wiener_pdf_w(rt[t], w_choice, safe_v, a_t, t_nd, w_bias)));
        
        Q[ch] = Q[ch] + alpha * (R - Q[ch]);
        int unch = (ch == 1) ? 0 : 1;
        Q[unch] = Q[unch] + alpha_c * ((1.0 - R) - Q[unch]);
    }
    double res = calc_pen_ll(D_vec);
    return (std::isnan(res) || std::isinf(res)) ? 1e9 : res;
}
// [[Rcpp::export]]
inline double eval_ql_ddm_dynamic_poly_tnd(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    double a = std::exp(phi[0]);
    double tau_min = 0.5 / (1.0 + std::exp(-phi[1]));
    double beta_v = std::exp(phi[2]);
    double alpha = 1.0 / (1.0 + std::exp(-phi[3]));
    double alpha_c = 1.0 / (1.0 + std::exp(-phi[4]));
    double theta_ctx = std::exp(phi[5]); 
    double beta_a = std::exp(phi[6]);    
    double theta_v = std::exp(phi[7]); 
    double delta_tau_base = std::exp(phi[8]);
    double kappa_auto = std::exp(phi[9]);
    double beta_pause = std::exp(phi[10]);
    
    std::vector<double> D_vec;
    double Q[2] = {0.5, 0.5};
    
    for (int t=0; t<resp.size(); ++t) {
        int ch = resp[t] - 1; 
        double R = (out[t] == 1) ? 1.0 : 0.0;
        
        double U_base = beta_v * (Q[1] - Q[0]);
        double abs_Q_diff = std::abs(Q[1] - Q[0]);
        double v = (U_base >= 0 ? 1.0 : -1.0) * std::pow(std::abs(U_base), theta_v);
        double safe_v = std::abs(v) < 1e-4 ? (v >= 0 ? 1e-4 : -1e-4) : v;
        
        double prev_rt = (t == 0) ? 0.5 : rt[t-1];
        double w_bias = 0.5 + 0.45 * std::tanh(theta_ctx * prev_rt);
        double a_t = a + beta_a * std::tanh(theta_ctx * prev_rt);
        if (a_t < 0.01) a_t = 0.01;
        if (a_t > 10.0) a_t = 10.0;
        
        double t_nd_t = tau_min + delta_tau_base * std::exp(-kappa_auto * abs_Q_diff) + beta_pause * prev_rt;
        if (t_nd_t >= rt[t]) t_nd_t = rt[t] - 0.001;
        if (t_nd_t < 0.001) t_nd_t = 0.001;
        
        int w_choice = (resp[t] == 2) ? 1 : 2;
        
        D_vec.push_back(-2.0 * std::log(wiener_pdf_w(rt[t], w_choice, safe_v, a_t, t_nd_t, w_bias)));
        
        Q[ch] = Q[ch] + alpha * (R - Q[ch]);
        int unch = (ch == 1) ? 0 : 1;
        Q[unch] = Q[unch] + alpha_c * ((1.0 - R) - Q[unch]);
    }
    double res = calc_pen_ll(D_vec);
    return (std::isnan(res) || std::isinf(res)) ? 1e9 : res;
}
