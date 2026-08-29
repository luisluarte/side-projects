#include <Rcpp.h>
#include <random>
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

inline double signum(double x) {
    if (x > 0) return 1.0;
    if (x < 0) return -1.0;
    return 0.0;
}

// [[Rcpp::export]]
NumericVector extract_epoch2_hybrid(const std::vector<double>& phi, const std::vector<double>& hyper, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt, bool return_ll) {
    // Hyperparameters bounding the Manifold's geometry
    int N = 100; // Fixed as per Variant 10 success
    double lambda_min = hyper[0];
    double lambda_max = hyper[1];
    double poisson_rate = hyper[2];
    
    // Continuous Parameters (optimized by L-BFGS-B)
    double a_base = std::exp(phi[0]);
    double t_nd_max = 1.0 / (1.0 + std::exp(-phi[1]));
    double kappa_v = std::exp(phi[2]); 
    double alpha_ctx = 1.0 / (1.0 + std::exp(-phi[3])); 
    double alpha_cb = 1.0 / (1.0 + std::exp(-phi[4])); 
    double lambda_lasso = std::exp(phi[5]); 
    double gamma_cb = 0.2 / (1.0 + std::exp(-phi[6])); // Bound between 0 and 0.2
    double scale_I = std::exp(phi[7]); 
    
    std::vector<double> Lambda(N);
    for(int i=0; i<N; ++i) {
        Lambda[i] = lambda_min * std::pow(lambda_max / lambda_min, (double)i / (double)(N - 1));
    }
    
    std::vector<double> h(N, 0.0);
    std::vector<double> W(N, 0.0);
    
    double Q_ctx[2] = {0.5, 0.5};
    double RPE_ctx_last = 0.5;
    
    int T = resp.size();
    NumericVector out_vec(T);
    
    // Deterministic seed for L-BFGS-B stability
    std::mt19937 gen(42);
    std::poisson_distribution<> d_pois(poisson_rate);
    
    for (int t = 0; t < T; ++t) {
        int ch = resp[t] - 1; 
        double R = (out[t] == 1) ? 1.0 : 0.0;
        
        // I. Cortical Deterministic Basis (Drift)
        double delta_Q = Q_ctx[1] - Q_ctx[0];
        double v_effective = kappa_v * delta_Q;
        double safe_v = std::abs(v_effective) < 1e-4 ? (v_effective >= 0 ? 1e-4 : -1e-4) : v_effective;
        
        // II. Sigmoidal Golgi Transfer Function
        double I_t = std::abs(RPE_ctx_last);
        double theta_t = 5.0 / (1.0 + std::exp(-scale_I * (I_t - 0.5)));
        
        // III. Stochastic Manifold Update
        for(int i=0; i<N; ++i) {
            double zeta = (double)d_pois(gen);
            double h_tilde = zeta - theta_t;
            if (h_tilde < 0.0) h_tilde = 0.0; // ReLU
            h[i] = (1.0 - Lambda[i]) * h[i] + Lambda[i] * h_tilde;
        }
        
        // IV. Cerebellar Readout (t_nd gating)
        double V_cb = 0.0;
        for(int i=0; i<N; ++i) {
            V_cb += W[i] * h[i];
        }
        
        double t_nd_eff = t_nd_max - gamma_cb * std::abs(V_cb);
        if (t_nd_eff < 0.01) t_nd_eff = 0.01;
        
        double a_effective = a_base;
        if (a_effective < 0.01) a_effective = 0.01;
        if (a_effective > 10.0) a_effective = 10.0;
        
        int w_choice = (resp[t] == 2) ? 1 : 2;
        if (return_ll) {
            out_vec[t] = std::log(wiener_pdf_w(rt[t], w_choice, safe_v, a_effective, t_nd_eff, 0.5));
        } else {
            out_vec[t] = calc_expected_rt_dist_t(safe_v, a_effective, 0.5, t_nd_eff);
        }
        
        // Post-Trial Cortical and Lasso Updates
        double RPE_ctx = R - Q_ctx[ch];
        Q_ctx[ch] += alpha_ctx * RPE_ctx;
        
        for(int i=0; i<N; ++i) {
            W[i] += alpha_cb * RPE_ctx * h[i] - lambda_lasso * signum(W[i]);
        }
        
        RPE_ctx_last = RPE_ctx;
    }
    return out_vec;
}

// [[Rcpp::export]]
inline double eval_epoch2_hybrid(const std::vector<double>& phi, const std::vector<double>& hyper, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    NumericVector ll = extract_epoch2_hybrid(phi, hyper, resp, out, rt, true);
    double nll = 0.0;
    for(int i=0; i<ll.size(); ++i) nll -= 2.0 * ll[i];
    return (std::isnan(nll) || std::isinf(nll)) ? 1e9 : nll;
}
