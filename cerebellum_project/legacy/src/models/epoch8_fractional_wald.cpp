#include <Rcpp.h>
#include <random>
#include <algorithm>
#include <cmath>
using namespace Rcpp;

inline double signum(double x) { return (x > 0) ? 1.0 : ((x < 0) ? -1.0 : 0.0); }

// Fast Inverse Gaussian (Wald) Sampler
double sample_wald(double mu, double lambda, std::mt19937& gen, std::normal_distribution<>& d_norm, std::uniform_real_distribution<>& d_unif) {
    double nu = d_norm(gen);
    double y = nu * nu;
    double mu_y_over_2l = (mu * mu * y) / (2.0 * lambda);
    double x = mu + mu_y_over_2l - (mu / (2.0 * lambda)) * std::sqrt(4.0 * mu * lambda * y + mu * mu * y * y);
    double z = d_unif(gen);
    if (z <= (mu / (mu + x))) {
        return x;
    } else {
        return (mu * mu) / x;
    }
}

// [[Rcpp::export]]
NumericVector extract_baseline_wald_sim(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    double a_base = std::exp(phi[0]);
    double mu_tnd = 1.0 / (1.0 + std::exp(-phi[1]));
    double kappa_v = std::exp(phi[2]); 
    double alpha_ctx = 1.0 / (1.0 + std::exp(-phi[3])); 
    
    int T = resp.size();
    NumericVector rt_sim(T);
    double Q_ctx[2] = {0.5, 0.5};
    
    std::mt19937 gen(42);
    std::normal_distribution<> d_norm(0.0, 1.0);
    std::uniform_real_distribution<> d_unif(0.0, 1.0);
    
    for (int t = 0; t < T; ++t) {
        int ch = resp[t] - 1; 
        double R = (out[t] == 1) ? 1.0 : 0.0;
        
        double Q_active = Q_ctx[ch];
        if (Q_active < 0.01) Q_active = 0.01;
        
        double v_t = kappa_v * Q_active;
        double mu_wald = a_base / v_t;
        double lambda_wald = a_base * a_base;
        
        double t_cog = sample_wald(mu_wald, lambda_wald, gen, d_norm, d_unif);
        
        rt_sim[t] = mu_tnd + t_cog; 
        if (rt_sim[t] < 0.01) rt_sim[t] = 0.01;
        
        double RPE_ctx = R - Q_ctx[ch];
        Q_ctx[ch] += alpha_ctx * RPE_ctx;
    }
    return rt_sim;
}

// [[Rcpp::export]]
NumericVector extract_epoch8_fractional(const std::vector<double>& phi, const std::vector<double>& hyper, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    int N = 100; 
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
    double gamma_spectral = std::exp(phi[9]); // Spectral scaling exponent
    
    std::vector<double> Lambda(N);
    std::vector<double> Tau(N);
    for(int i=0; i<N; ++i) {
        Lambda[i] = lambda_min * std::pow(lambda_max / lambda_min, (double)i / (double)(N - 1));
        Tau[i] = 1.0 / Lambda[i];
    }
    
    std::vector<double> h(N, 0.0);
    std::vector<double> W_readout(N, 0.0);
    
    // Spectral Mossy Fiber Projection Matrix
    std::mt19937 init_gen(12345);
    std::normal_distribution<> d_w(0.0, 1.0);
    std::vector<std::vector<double>> W_MF(N, std::vector<double>(3));
    for(int i=0; i<N; ++i) {
        double std_scale = std::sqrt(std::pow(Tau[i], -gamma_spectral));
        for(int j=0; j<3; ++j) {
            W_MF[i][j] = d_w(init_gen) * std_scale;
        }
    }
    
    double Q_ctx[2] = {0.5, 0.5};
    double RPE_ctx_last = 0.5;
    
    double prev_choice = 0.5;
    double prev_rt = 0.5;
    double prev_reward = 0.5;
    
    int T = resp.size();
    NumericVector rt_sim(T);
    
    std::mt19937 gen(42);
    std::poisson_distribution<> d_pois(poisson_rate);
    std::normal_distribution<> d_norm(0.0, 1.0);
    std::uniform_real_distribution<> d_unif(0.0, 1.0);
    
    for (int t = 0; t < T; ++t) {
        int ch = resp[t] - 1; 
        double R = (out[t] == 1) ? 1.0 : 0.0;
        
        // I. Cortical Basis
        double Q_active = Q_ctx[ch];
        if (Q_active < 0.01) Q_active = 0.01;
        
        // II. Golgi Gate
        double I_t = std::abs(RPE_ctx_last);
        double theta_t = 5.0 / (1.0 + std::exp(-scale_I * (I_t - 0.5)));
        
        // III. Fractional Contextual Manifold
        double L1_norm = 0.0;
        for(int i=0; i<N; ++i) {
            double u_i = W_MF[i][0] * prev_choice + W_MF[i][1] * prev_rt + W_MF[i][2] * prev_reward;
            double zeta = (double)d_pois(gen);
            
            double h_tilde = u_i + zeta - theta_t;
            if (h_tilde < 0.0) h_tilde = 0.0; 
            
            h[i] = (1.0 - Lambda[i]) * h[i] + Lambda[i] * h_tilde;
            L1_norm += std::abs(h[i]);
        }
        
        // IV. Cerebellar Readout
        double V_cb = 0.0;
        for(int i=0; i<N; ++i) {
            V_cb += W_readout[i] * h[i];
        }
        
        double mu_exec = t_nd_max - gamma_cb * std::abs(V_cb);
        if (mu_exec < 0.01) mu_exec = 0.01;
        double sigma_exec = eta_var * L1_norm;
        
        // V. Generative Draw (Wald-Gaussian Superposition)
        double v_t = kappa_v * Q_active;
        double mu_wald = a_base / v_t;
        double lambda_wald = a_base * a_base;
        double t_cog = sample_wald(mu_wald, lambda_wald, gen, d_norm, d_unif);
        
        double t_exec = mu_exec + sigma_exec * d_norm(gen);
        
        rt_sim[t] = t_cog + t_exec;
        if (rt_sim[t] < 0.01) rt_sim[t] = 0.01;
        
        // VI. Plasticity Updates
        double RPE = R - Q_ctx[ch];
        Q_ctx[ch] += alpha_ctx * RPE;
        
        for(int i=0; i<N; ++i) {
            W_readout[i] += alpha_cb * RPE * h[i] - lambda_lasso * signum(W_readout[i]);
        }
        RPE_ctx_last = RPE;
        
        // VII. Context Update
        prev_choice = (double)ch;
        prev_rt = rt[t]; 
        prev_reward = R;
    }
    return rt_sim;
}
