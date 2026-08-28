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
NumericVector extract_mutant_manifold(const std::vector<double>& phi, const std::vector<double>& hyper, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt, bool return_ll) {
    // Hyperparameters bounded by variants
    int N = (int)hyper[0];
    int golgi_type = (int)hyper[1];
    double lambda_min = hyper[2];
    double lambda_max = hyper[3];
    double poisson_rate = hyper[4];
    
    // Continuous Parameters (optimized by L-BFGS-B)
    double kappa_v = std::exp(phi[0]); // Drift scaling
    double a_base = std::exp(phi[1]);
    double t_nd = 1.0 / (1.0 + std::exp(-phi[2]));
    double alpha = 1.0 / (1.0 + std::exp(-phi[3])); // Learning rate
    double lambda_lasso = std::exp(phi[4]); // L1 pruning rate
    double gamma_I = 1.0 / (1.0 + std::exp(-phi[5])); // Volatility integration
    double scale_I = std::exp(phi[6]); // Golgi mapping
    double beta_str_a = phi[7]; // Macroscopic boundary fatigue
    
    std::vector<double> Lambda(N);
    for(int i=0; i<N; ++i) {
        Lambda[i] = lambda_min * std::pow(lambda_max / lambda_min, (double)i / (double)(N - 1));
    }
    
    std::vector<double> h(N, 0.0);
    std::vector<double> W_1(N, 0.0);
    std::vector<double> W_2(N, 0.0);
    
    double I_t = 0.5;
    double Str_bar = 0.5;
    double rho_str = 0.05;
    
    int T = resp.size();
    NumericVector out_vec(T);
    
    // Deterministic seed for L-BFGS-B stability
    std::mt19937 gen(42);
    std::poisson_distribution<> d_pois(poisson_rate);
    
    for (int t = 0; t < T; ++t) {
        int ch = resp[t] - 1; 
        double R = (out[t] == 1) ? 1.0 : 0.0;
        
        // 2. Variance Regulator (Golgi)
        double theta_t = 0.0;
        if (golgi_type == 0) {
            theta_t = 1.0 / (scale_I * I_t + 0.01);
        } else {
            theta_t = 1.0 / (1.0 + std::exp(-scale_I * (I_t - 0.5))) * 5.0;
        }
        
        // 1 & 3. Variance Generator & Mutant Manifold Update
        for(int i=0; i<N; ++i) {
            double zeta = (double)d_pois(gen);
            double h_tilde = zeta - theta_t;
            if (h_tilde < 0.0) h_tilde = 0.0; // ReLU
            h[i] = (1.0 - Lambda[i]) * h[i] + Lambda[i] * h_tilde;
        }
        
        // 4. Unified Lasso Readout
        double V_1 = 0.0;
        double V_2 = 0.0;
        for(int i=0; i<N; ++i) {
            V_1 += W_1[i] * h[i];
            V_2 += W_2[i] * h[i];
        }
        
        // DDM Output Mapping
        double v_effective = kappa_v * (V_2 - V_1);
        double safe_v = std::abs(v_effective) < 1e-4 ? (v_effective >= 0 ? 1e-4 : -1e-4) : v_effective;
        
        double a_effective = a_base + beta_str_a * (0.5 - Str_bar);
        if (a_effective < 0.01) a_effective = 0.01;
        if (a_effective > 10.0) a_effective = 10.0;
        
        int w_choice = (resp[t] == 2) ? 1 : 2;
        if (return_ll) {
            out_vec[t] = std::log(wiener_pdf_w(rt[t], w_choice, safe_v, a_effective, t_nd, 0.5));
        } else {
            out_vec[t] = calc_expected_rt_dist_t(safe_v, a_effective, 0.5, t_nd);
        }
        
        // Post-Trial Updates
        double V_chosen = (ch == 0) ? V_1 : V_2;
        double RPE = R - V_chosen;
        
        for(int i=0; i<N; ++i) {
            if (ch == 0) {
                W_1[i] += alpha * RPE * h[i] - lambda_lasso * signum(W_1[i]);
            } else {
                W_2[i] += alpha * RPE * h[i] - lambda_lasso * signum(W_2[i]);
            }
        }
        
        I_t = (1.0 - gamma_I) * I_t + gamma_I * std::abs(RPE);
        Str_bar = (1.0 - rho_str) * Str_bar + rho_str * R;
    }
    return out_vec;
}

// [[Rcpp::export]]
inline double eval_mutant_manifold(const std::vector<double>& phi, const std::vector<double>& hyper, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    NumericVector ll = extract_mutant_manifold(phi, hyper, resp, out, rt, true);
    double nll = 0.0;
    for(int i=0; i<ll.size(); ++i) nll -= 2.0 * ll[i];
    return (std::isnan(nll) || std::isinf(nll)) ? 1e9 : nll;
}
