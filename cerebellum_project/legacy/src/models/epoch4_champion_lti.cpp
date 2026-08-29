#include <Rcpp.h>
#include <random>
#include <algorithm>
#include <cmath>
using namespace Rcpp;

inline double signum(double x) {
    if (x > 0) return 1.0;
    if (x < 0) return -1.0;
    return 0.0;
}

// [[Rcpp::export]]
NumericVector extract_epoch4_lti(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    int N = 20; 
    
    double a_base = std::exp(phi[0]);
    double t_nd_max = 1.0 / (1.0 + std::exp(-phi[1]));
    double kappa_v = std::exp(phi[2]); 
    double alpha_ctx = 1.0 / (1.0 + std::exp(-phi[3])); 
    double alpha_cb = 1.0 / (1.0 + std::exp(-phi[4])); 
    double beta_a = std::exp(phi[5]);
    double gamma_suppress = std::exp(phi[6]);
    double tau_m = std::exp(phi[7]); 
    double gamma_cb = 0.5 / (1.0 + std::exp(-phi[8])); 
    double eta_var = std::exp(phi[9]);
    double lambda_gc = 0.001; 
    
    std::vector<double> m(N, 0.0);
    std::vector<double> w_gc(N, 0.0);
    std::vector<double> w_mli(N, 0.0);
    
    double Q_ctx[2] = {0.5, 0.5};
    
    int T = resp.size();
    NumericVector rt_sim(T);
    
    std::mt19937 gen(42);
    std::normal_distribution<> d_norm(0.0, 1.0);
    std::exponential_distribution<> d_exp(1.0);
    
    for (int t = 0; t < T; ++t) {
        int ch = resp[t] - 1; 
        double R = (out[t] == 1) ? 1.0 : 0.0;
        
        // I. Temporal Cascade (LTI Poisson Integration)
        double dt = (t == 0) ? 0.5 : rt[t-1] + 0.5;
        double x = dt / tau_m;
        if (x > 50.0) x = 50.0; // bounds check
        double exp_x = std::exp(-x);
        
        std::vector<double> m_new(N, 0.0);
        for(int k=0; k<N; ++k) {
            double sum_val = 0.0;
            double x_pow = 1.0;
            double fact = 1.0;
            for(int j=1; j<=k+1; ++j) {
                sum_val += m[k-j+1] * (x_pow / fact);
                x_pow *= x;
                fact *= (double)j;
            }
            m_new[k] = exp_x * sum_val;
        }
        m_new[0] += 1.0; // Trial onset impulse
        
        double L1_norm = 0.0;
        for(int k=0; k<N; ++k) {
            m[k] = m_new[k];
            L1_norm += std::abs(m[k]);
        }
        
        // III. Cerebellar Readout
        double Q_cb = 0.0;
        for(int k=0; k<N; ++k) {
            Q_cb += w_gc[k] * m[k] - w_mli[k] * m[k];
        }
        
        // II. Cortical Conflict & Decision Tail
        double delta_Q_ctx = Q_ctx[1] - Q_ctx[0];
        double conflict = 0.5 * (1.0 - std::tanh(10.0 * delta_Q_ctx) * std::tanh(10.0 * Q_cb));
        
        double v_base = kappa_v * std::abs(delta_Q_ctx);
        if (v_base < 1e-4) v_base = 1e-4;
        
        double a_t = a_base + beta_a * conflict;
        double v_eff = v_base * std::exp(-gamma_suppress * conflict);
        
        double tau = a_t / v_eff;
        if (tau < 1e-4) tau = 1e-4;
        if (tau > 10.0) tau = 10.0;
        
        double mu_tnd = t_nd_max - gamma_cb * std::abs(Q_cb);
        if (mu_tnd < 0.01) mu_tnd = 0.01;
        
        double sigma_tnd = eta_var * L1_norm;
        
        // IV. Generative LFI Evaluation
        double N_samp = d_norm(gen);
        double E_samp = d_exp(gen);
        rt_sim[t] = mu_tnd + sigma_tnd * N_samp + tau * E_samp;
        if (rt_sim[t] < 0.01) rt_sim[t] = 0.01;
        
        // Update Plasticity
        double RPE = R - Q_ctx[ch];
        Q_ctx[ch] += alpha_ctx * RPE;
        
        for(int k=0; k<N; ++k) {
            w_gc[k] += alpha_cb * RPE * m[k] - lambda_gc * signum(w_gc[k]);
            w_mli[k] -= alpha_cb * RPE * m[k] - lambda_gc * signum(w_mli[k]);
        }
    }
    return rt_sim;
}
