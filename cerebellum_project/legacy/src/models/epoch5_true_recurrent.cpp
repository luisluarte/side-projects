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
NumericVector extract_baseline_exgauss_sim(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    double a_base = std::exp(phi[0]);
    double mu_tnd = 1.0 / (1.0 + std::exp(-phi[1]));
    double kappa_v = std::exp(phi[2]); 
    double alpha_ctx = 1.0 / (1.0 + std::exp(-phi[3])); 
    
    int T = resp.size();
    NumericVector rt_sim(T);
    double Q_ctx[2] = {0.5, 0.5};
    
    std::mt19937 gen(42);
    std::normal_distribution<> d_norm(0.0, 1.0);
    std::exponential_distribution<> d_exp(1.0);
    
    for (int t = 0; t < T; ++t) {
        int ch = resp[t] - 1; 
        double R = (out[t] == 1) ? 1.0 : 0.0;
        
        double Q_active = Q_ctx[ch];
        if (Q_active < 0.01) Q_active = 0.01;
        double tau = a_base / (kappa_v * Q_active);
        if (tau < 1e-4) tau = 1e-4;
        
        double E_samp = d_exp(gen);
        
        rt_sim[t] = mu_tnd + tau * E_samp; 
        if (rt_sim[t] < 0.01) rt_sim[t] = 0.01;
        
        double RPE_ctx = R - Q_ctx[ch];
        Q_ctx[ch] += alpha_ctx * RPE_ctx;
    }
    return rt_sim;
}

// [[Rcpp::export]]
NumericVector extract_epoch5_true_recurrent(const std::vector<double>& phi, const std::vector<double>& hyper, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    int N = 50; // N=50 is sufficient for spatial dynamics without enormous O(N^2) overhead
    double lambda_min = hyper[0];
    double lambda_max = hyper[1];
    double poisson_rate = hyper[2];
    
    double a_base = std::exp(phi[0]);
    double t_nd_max = 1.0 / (1.0 + std::exp(-phi[1]));
    double kappa_v = std::exp(phi[2]); 
    double alpha_ctx = 1.0 / (1.0 + std::exp(-phi[3])); 
    double alpha_cb = 1.0 / (1.0 + std::exp(-phi[4])); 
    double lambda_lasso = std::exp(phi[5]); 
    double gamma_cb = 0.5 / (1.0 + std::exp(-phi[6])); 
    double scale_I = std::exp(phi[7]); 
    double eta_var = std::exp(phi[8]); 
    double rho_res = 1.5 / (1.0 + std::exp(-phi[9])); // Spatial recurrent weight scaling
    
    std::vector<double> Lambda(N);
    for(int i=0; i<N; ++i) {
        Lambda[i] = lambda_min * std::pow(lambda_max / lambda_min, (double)i / (double)(N - 1));
    }
    
    std::vector<double> h(N, 0.0);
    std::vector<double> W_readout(N, 0.0);
    
    // Fixed Spatial Recurrent Topology
    std::mt19937 init_gen(12345);
    std::normal_distribution<> d_w(0.0, 1.0 / std::sqrt((double)N));
    std::vector<std::vector<double>> W_res(N, std::vector<double>(N));
    for(int i=0; i<N; ++i) {
        for(int j=0; j<N; ++j) {
            W_res[i][j] = d_w(init_gen);
        }
    }
    
    double Q_ctx[2] = {0.5, 0.5};
    double RPE_ctx_last = 0.5;
    
    int T = resp.size();
    NumericVector rt_sim(T);
    
    // Reparameterization Trick / Fixed Sequence Generative Draw
    std::mt19937 gen(42);
    std::poisson_distribution<> d_pois(poisson_rate);
    std::normal_distribution<> d_norm(0.0, 1.0);
    std::exponential_distribution<> d_exp(1.0);
    
    for (int t = 0; t < T; ++t) {
        int ch = resp[t] - 1; 
        double R = (out[t] == 1) ? 1.0 : 0.0;
        
        // I. Cortical Basis
        double Q_active = Q_ctx[ch];
        if (Q_active < 0.01) Q_active = 0.01;
        double tau = a_base / (kappa_v * Q_active);
        if (tau < 1e-4) tau = 1e-4;
        
        // II. Golgi Gate
        double I_t = std::abs(RPE_ctx_last);
        double theta_t = 5.0 / (1.0 + std::exp(-scale_I * (I_t - 0.5)));
        
        // III. True Spatial Recurrence (O(N^2))
        std::vector<double> h_new(N, 0.0);
        double L1_norm = 0.0;
        
        for(int i=0; i<N; ++i) {
            double rec_input = 0.0;
            for(int j=0; j<N; ++j) {
                rec_input += W_res[i][j] * h[j];
            }
            
            double zeta = (double)d_pois(gen); // Biological Stochasticity
            double h_tilde = (rho_res * rec_input) + zeta - theta_t;
            if (h_tilde < 0.0) h_tilde = 0.0; 
            
            h_new[i] = (1.0 - Lambda[i]) * h[i] + Lambda[i] * h_tilde;
            L1_norm += std::abs(h_new[i]);
        }
        
        // Update h
        for(int i=0; i<N; ++i) {
            h[i] = h_new[i];
        }
        
        // IV. Cerebellar Readout
        double V_cb = 0.0;
        for(int i=0; i<N; ++i) {
            V_cb += W_readout[i] * h[i];
        }
        
        double mu_tnd = t_nd_max - gamma_cb * std::abs(V_cb);
        if (mu_tnd < 0.01) mu_tnd = 0.01;
        double sigma_tnd = eta_var * L1_norm;
        
        // V. Generative Draw
        double N_samp = d_norm(gen);
        double E_samp = d_exp(gen);
        rt_sim[t] = mu_tnd + sigma_tnd * N_samp + tau * E_samp;
        if (rt_sim[t] < 0.01) rt_sim[t] = 0.01;
        
        // VI. Plasticity Updates
        double RPE = R - Q_ctx[ch];
        Q_ctx[ch] += alpha_ctx * RPE;
        
        for(int i=0; i<N; ++i) {
            W_readout[i] += alpha_cb * RPE * h[i] - lambda_lasso * signum(W_readout[i]);
        }
        RPE_ctx_last = RPE;
    }
    return rt_sim;
}
