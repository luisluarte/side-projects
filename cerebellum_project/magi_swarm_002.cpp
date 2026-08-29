#include <Rcpp.h>
#include <cmath>
#include <vector>
#include <random>

using namespace Rcpp;

static double wfpt_pdf(double t, int resp, double v, double a, double t0, double w) {
    if (t <= t0 || a < 1e-6) return 1e-10;
    double tt = t - t0;
    if (resp == 2) { v = -v; w = 1.0 - w; }
    double tau = tt / (a * a);
    if (tau < 1e-6) return 1e-10;
    
    double p = 0.0;
    if (tau > 0.15) {
        for(int k=1; k<50; ++k) {
            double term = k * std::exp(-0.5*k*k*M_PI*M_PI*tau) * std::sin(k*M_PI*w);
            p += term;
            if (std::abs(term) < 1e-10) break;
        }
        p *= M_PI;
    } else {
        for(int k=-10; k<=10; ++k) {
            double r = w + 2.0*k;
            p += r * std::exp(-0.5*r*r/tau);
        }
        p /= std::sqrt(2.0*M_PI*tau*tau*tau);
    }
    double res = (p/(a*a)) * std::exp(-v*a*w - 0.5*v*v*tt);
    return res > 1e-10 ? res : 1e-10;
}

// hyper: [0]lambda_sparse, [1]beta_ising, [2]K_sa, [3]theta_io
// phi: [0]a, [1]tnd, [2]v_ctx, [3]alpha_ctx, [4]gamma, [5]tau_omega, [6]lambda_sa

// [[Rcpp::export]]
double get_nll_swarm_002(const std::vector<double>& phi, const std::vector<double>& hyper,
                      const IntegerVector& resp, const NumericVector& reward, const NumericVector& rt) {
    double lambda_sparse = hyper[0], beta_ising = hyper[1];
    int K_sa = (int)hyper[2];
    double theta_io = hyper[3];
    
    double a = std::exp(phi[0]), tnd = 0.8 / (1.0 + std::exp(-phi[1]));
    double v_ctx = std::exp(phi[2]), alpha_ctx = 1.0 / (1.0 + std::exp(-phi[3]));
    double gamma = std::exp(phi[4]), tau_omega = 1.0 / (1.0 + std::exp(-phi[5]));
    double lambda_sa_temp = std::exp(phi[6]);
    
    int N = 32;
    std::vector<double> h(N,0), Z(N,0), W(N,0), W_exp(N,0), frac_mem(N,0);
    std::vector<int> S(N,0);
    double Omega = 0.0;
    int last_spike = -10; // Refractory tracker
    
    std::mt19937 gen(42);
    std::uniform_real_distribution<> unif(0.0,1.0);
    std::normal_distribution<> norm(0.0,1.0);
    for(int i=0;i<N;++i){W_exp[i]=norm(gen);W[i]=norm(gen);}
    
    // Spatio-Temporal Gradients
    std::vector<double> frac_alpha(N, 0.0), kappa_vec(N, 0.0);
    for(int i=0; i<N; ++i) {
        frac_alpha[i] = 0.1 + 0.8 * (double(i) / double(N - 1));
        kappa_vec[i] = 0.1 + 0.89 * (double(i) / double(N - 1)); // 0.1 to 0.99
    }
    
    double nll=0.0, prev_E=0.0;
    double Q[2]={0.5,0.5};
    
    for(int t=0;t<resp.size();++t){
        int ch=resp[t]-1; double R=(reward[t]==1.0)?1.0:0.0;
        
        // 1. Spatio-Temporal Expansion & Trace
        for(int i=0;i<N;++i){
            frac_mem[i] = frac_alpha[i] * frac_mem[i] + (1.0 - frac_alpha[i]) * W_exp[i] * Q[ch];
            h[i] = std::tanh(frac_mem[i]);
            Z[i] = kappa_vec[i] * Z[i] + h[i];
        }
        
        // 2. IO Refractory Gating
        double e_io = std::abs(prev_E);
        bool io_spike = (e_io > theta_io) && ((t - last_spike) >= 1); // 1-trial absolute refractory
        if (io_spike) last_spike = t;
        double f_io = io_spike ? e_io : 0.0;
        
        Omega = Omega * (1.0 - tau_omega) + gamma * f_io;
        
        double cb=0.0;
        for(int i=0; i<N; ++i) if(S[i]==1) cb += W[i]*Z[i];
        
        // 3. Drift Integration
        double v_eff = v_ctx*(Q[1]-Q[0]) + Omega*cb;
        if(std::abs(v_eff)<1e-4) v_eff = v_eff>=0?1e-4:-1e-4;
        
        nll -= std::log(wfpt_pdf(rt[t], resp[t], v_eff, a, tnd, 0.5));
        
        prev_E = R - Q[ch];
        Q[ch] += alpha_ctx * prev_E;
        
        // 4. Thermodynamics with Gap-Junction Ising Prior
        if(std::abs(prev_E)>0.01){
            auto calc_F = [&](const std::vector<int>& S_vec) {
                double F_cb = 0.0, L1 = 0.0, Ising = 0.0;
                for(int i=0; i<N; ++i) {
                    if (S_vec[i]) {
                        F_cb += W[i]*Z[i];
                        L1 += 1.0;
                    }
                    if (i < N-1 && S_vec[i]==1 && S_vec[i+1]==1) Ising += 1.0;
                }
                return std::pow(-prev_E - F_cb, 2) + lambda_sparse * L1 - beta_ising * Ising;
            };
            
            double cF = calc_F(S);
            
            for(int k=1;k<=K_sa;++k){
                double temp = std::abs(prev_E) * std::exp(-lambda_sa_temp*k);
                int idx = gen()%N;
                S[idx] = 1-S[idx];
                
                double nF = calc_F(S);
                
                if(nF - cF < 0 || unif(gen) < std::exp(-(nF - cF)/std::max(temp,1e-10))) cF = nF;
                else S[idx] = 1-S[idx]; // Revert
            }
        }
    }
    return nll;
}
