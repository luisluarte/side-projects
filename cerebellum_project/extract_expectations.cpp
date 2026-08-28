#include <Rcpp.h>
#include <random>
#include <cmath>
using namespace Rcpp;

inline double signum(double x) { return (x > 0) ? 1.0 : ((x < 0) ? -1.0 : 0.0); }

double sample_wald2(double mu, double lambda, std::mt19937& gen, std::normal_distribution<>& d_norm, std::uniform_real_distribution<>& d_unif) {
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
NumericMatrix get_base_expect(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    double a_base = std::exp(phi[0]);
    double mu_tnd = 1.0 / (1.0 + std::exp(-phi[1]));
    double kappa_v = std::exp(phi[2]); 
    double alpha_ctx = 1.0 / (1.0 + std::exp(-phi[3])); 
    int T = resp.size();
    NumericMatrix res(T, 2);
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
        
        double t_cog = sample_wald2(mu_wald, lambda_wald, gen, d_norm, d_unif);
        double rt_sim = mu_tnd + t_cog; 
        if (rt_sim < 0.01) rt_sim = 0.01;
        
        res(t, 0) = rt_sim;
        res(t, 1) = mu_wald + mu_tnd; // Baseline deterministic expectation
        
        double RPE_ctx = R - Q_ctx[ch];
        Q_ctx[ch] += alpha_ctx * RPE_ctx;
    }
    return res;
}

// [[Rcpp::export]]
NumericMatrix get_hybrid_expect(const std::vector<double>& phi, const std::vector<double>& hyper, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
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
    
    std::vector<double> h(N, 0.0), W_cb(N, 0.0);
    std::mt19937 init_gen(12345);
    std::normal_distribution<> d_w(0.0, 1.0);
    std::vector<std::vector<double>> W_MF(N, std::vector<double>(3));
    for(int i=0; i<N; ++i) {
        double std_scale = std::sqrt(std::pow(Tau[i], -gamma_spectral));
        for(int j=0; j<3; ++j) { W_MF[i][j] = d_w(init_gen) * std_scale; }
    }
    
    double Q_ctx[2] = {0.5, 0.5}, RPE_ctx_last = 0.5, prev_choice = 0.5, prev_rt = 0.5, prev_reward = 0.5;
    int T = resp.size();
    NumericMatrix res(T, 2);
    
    std::mt19937 gen(42);
    std::poisson_distribution<> d_pois(poisson_rate);
    std::normal_distribution<> d_norm(0.0, 1.0);
    std::uniform_real_distribution<> d_unif(0.0, 1.0);
    
    for (int t = 0; t < T; ++t) {
        int ch = resp[t] - 1; double R = (out[t] == 1) ? 1.0 : 0.0;
        double I_t = std::abs(RPE_ctx_last);
        double theta_t = 5.0 / (1.0 + std::exp(-scale_I * (I_t - 0.5)));
        
        for(int i=0; i<N; ++i) {
            double u_i = W_MF[i][0] * prev_choice + W_MF[i][1] * prev_rt + W_MF[i][2] * prev_reward;
            double zeta = (double)d_pois(gen);
            double h_tilde = u_i + zeta - theta_t;
            if (h_tilde < 0.0) h_tilde = 0.0; 
            h[i] = (1.0 - Lambda[i]) * h[i] + Lambda[i] * h_tilde;
        }
        
        double Q_cb = 0.0;
        for(int i=0; i<N; ++i) Q_cb += W_cb[i] * h[i];
        
        double V_eff = Q_ctx[ch] * (1.0 + gamma_perturb * std::tanh(Q_cb));
        double v_t = kappa_v * std::max(V_eff, 1e-4);
        double a_t = a_base * std::exp(delta_cb * std::tanh(Q_cb));
        double mu_wald = a_t / v_t;
        double lambda_wald = (a_t * a_t) / (sigma_diff * sigma_diff);
        
        double t_cog = sample_wald2(mu_wald, lambda_wald, gen, d_norm, d_unif);
        double t_nd_sample = t_nd * std::exp(sigma_tnd * d_norm(gen));
        double rt_sim = t_cog + t_nd_sample;
        if (rt_sim < 0.01) rt_sim = 0.01;
        
        res(t, 0) = rt_sim;
        
        // Mathematical Expectation of Wald + LogNormal
        double expected_tnd = t_nd * std::exp((sigma_tnd * sigma_tnd) / 2.0);
        res(t, 1) = mu_wald + expected_tnd; 
        
        double RPE = R - Q_ctx[ch];
        Q_ctx[ch] += alpha_ctx * RPE;
        for(int i=0; i<N; ++i) W_cb[i] += alpha_cb * RPE * h[i] - lambda_lasso * signum(W_cb[i]);
        RPE_ctx_last = RPE;
        prev_choice = (double)ch; prev_rt = rt[t]; prev_reward = R;
    }
    return res;
}
