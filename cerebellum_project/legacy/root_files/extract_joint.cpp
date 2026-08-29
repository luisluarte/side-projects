#include <Rcpp.h>
#include <random>
#include <cmath>
#include <vector>
using namespace Rcpp;

inline double signum(double x) { return (x > 0) ? 1.0 : ((x < 0) ? -1.0 : 0.0); }

double sample_w(double mu, double lambda, std::mt19937& gen, std::normal_distribution<>& d_norm, std::uniform_real_distribution<>& d_unif) {
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
NumericMatrix extract_base_joint(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    double a_base = std::exp(phi[0]);
    double mu_tnd = 1.0 / (1.0 + std::exp(-phi[1]));
    double kappa_v = std::exp(phi[2]); 
    double alpha_ctx = 1.0 / (1.0 + std::exp(-phi[3])); 
    
    int T = resp.size();
    NumericMatrix res(T, 3);
    double Q_ctx[2] = {0.5, 0.5};
    
    std::mt19937 gen(42);
    std::normal_distribution<> d_norm(0.0, 1.0);
    std::uniform_real_distribution<> d_unif(0.0, 1.0);
    
    for (int t = 0; t < T; ++t) {
        int ch = resp[t] - 1; 
        double R = (out[t] == 1) ? 1.0 : 0.0;
        
        double v0 = kappa_v * std::max(Q_ctx[0], 0.01);
        double v1 = kappa_v * std::max(Q_ctx[1], 0.01);
        double p1 = std::exp(v1) / (std::exp(v0) + std::exp(v1));
        
        double Q_active = Q_ctx[ch];
        if (Q_active < 0.01) Q_active = 0.01;
        
        double v_t = kappa_v * Q_active;
        double mu_wald = a_base / v_t;
        double lambda_wald = a_base * a_base;
        
        double t_cog = sample_w(mu_wald, lambda_wald, gen, d_norm, d_unif);
        double rt_sim = mu_tnd + t_cog; 
        if (rt_sim < 0.01) rt_sim = 0.01;
        
        res(t, 0) = rt_sim;
        res(t, 1) = mu_wald + mu_tnd;
        res(t, 2) = p1;
        
        double RPE_ctx = R - Q_ctx[ch];
        Q_ctx[ch] += alpha_ctx * RPE_ctx;
    }
    return res;
}

// [[Rcpp::export]]
NumericMatrix extract_hybrid_joint(const std::vector<double>& phi, const std::vector<double>& hyper, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    int N = 100; 
    double lambda_min = hyper[0], lambda_max = hyper[1], poisson_rate = hyper[2];
    double a_base = std::exp(phi[0]), t_nd = 1.0 / (1.0 + std::exp(-phi[1])), kappa_v = std::exp(phi[2]); 
    double alpha_ctx = 1.0 / (1.0 + std::exp(-phi[3])), alpha_cb = 1.0 / (1.0 + std::exp(-phi[4])); 
    double lambda_lasso = std::exp(phi[5]), gamma_perturb = 1.0 / (1.0 + std::exp(-phi[6])); 
    double scale_I = std::exp(phi[7]), gamma_spectral = std::exp(phi[8]); 
    double sigma_diff = std::exp(phi[9]), sigma_tnd = std::exp(phi[10]), delta_cb = phi[11];
    
    std::vector<double> Lambda(N), Tau(N);
    for(int i=0; i<N; ++i) {
        Lambda[i] = lambda_min * std::pow(lambda_max / lambda_min, (double)i / (double)(N - 1));
        Tau[i] = 1.0 / Lambda[i];
    }
    
    double Q_ctx[2] = {0.5, 0.5};
    std::vector<double> W_cb(N, 0.0), Z_trace(N, 0.0);
    double cb_bias = 0.0;
    
    int T = resp.size();
    NumericMatrix res(T, 3);
    
    std::mt19937 gen(42);
    std::normal_distribution<> d_norm(0.0, 1.0);
    std::uniform_real_distribution<> d_unif(0.0, 1.0);
    
    for (int t = 0; t < T; ++t) {
        int ch = resp[t] - 1; 
        double R = (out[t] == 1) ? 1.0 : 0.0;
        
        double v0 = kappa_v * std::max(Q_ctx[0], 0.01);
        double v1 = kappa_v * std::max(Q_ctx[1], 0.01);
        
        // Add cerebellar bias to choice 1 vs 0
        // Cerebellar bias is added to the active drift. We approximate its effect on P(Ch=1) 
        // by pushing v1 with cb_bias and v0 with -cb_bias
        v1 += cb_bias;
        v0 -= cb_bias;
        if(v1 < 0.01) v1 = 0.01;
        if(v0 < 0.01) v0 = 0.01;
        
        double p1 = std::exp(v1) / (std::exp(v0) + std::exp(v1));
        
        double Q_active = Q_ctx[ch];
        if (Q_active < 0.01) Q_active = 0.01;
        
        double v_t = kappa_v * Q_active + signum((double)ch - 0.5) * cb_bias;
        if (v_t < 0.01) v_t = 0.01;
        
        double a_t = a_base;
        double a_mod = 1.0 + gamma_spectral * std::abs(cb_bias);
        a_t *= a_mod; 
        
        double mu_wald = a_t / v_t;
        double lambda_wald = a_t * a_t;
        
        double t_cog = sample_w(mu_wald, lambda_wald, gen, d_norm, d_unif);
        double rt_sim = t_nd + t_cog; 
        if (rt_sim < 0.01) rt_sim = 0.01;
        
        res(t, 0) = rt_sim;
        res(t, 1) = mu_wald + t_nd;
        res(t, 2) = p1;
        
        double RPE_ctx = R - Q_ctx[ch];
        Q_ctx[ch] += alpha_ctx * RPE_ctx;
        
        double RPE_cb = R - Q_active;
        cb_bias = 0.0;
        for(int i=0; i<N; ++i) {
            double z_old = Z_trace[i];
            Z_trace[i] = z_old * std::exp(-1.0 / Tau[i]) + 1.0;
            W_cb[i] += alpha_cb * RPE_cb * Z_trace[i] - lambda_lasso * signum(W_cb[i]);
            cb_bias += W_cb[i] * Z_trace[i] * delta_cb;
        }
    }
    return res;
}
