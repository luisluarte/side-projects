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

// ==========================================
// BASELINE WALD (WFPT)
// ==========================================
// [[Rcpp::export]]
double get_nll_base_w(const std::vector<double>& phi, const IntegerVector& resp, const NumericVector& reward, const NumericVector& rt) {
    double a = std::exp(phi[0]), tnd = 0.8 / (1.0 + std::exp(-phi[1]));
    double k_v = std::exp(phi[2]), alpha = 1.0 / (1.0 + std::exp(-phi[3]));
    double nll = 0.0; double Q[2] = {0.5, 0.5};
    for(int t=0; t<resp.size(); ++t) {
        int ch = resp[t]-1; double R = (reward[t]==1.0)?1.0:0.0;
        double v = k_v*(Q[1]-Q[0]);
        if(std::abs(v)<1e-4) v = v>=0?1e-4:-1e-4;
        nll -= std::log(wfpt_pdf(rt[t], resp[t], v, a, tnd, 0.5));
        Q[ch] += alpha*(R - Q[ch]);
    }
    return nll;
}

// ==========================================
// SWARM 001: HETEROGENEOUS FRAC + DUAL EXP + MODULAR + THRESHOLD OMEGA
// ==========================================
// hyper: [0]k_fast, [1]k_slow, [2]lambda_sparse, [3]K_sa
// phi: [0]a, [1]tnd, [2]v_ctx, [3]alpha_ctx, [4]gamma, [5]tau_omega, [6]lambda_sa, [7]omega_thresh

// [[Rcpp::export]]
double get_nll_swarm_001(const std::vector<double>& phi, const std::vector<double>& hyper,
                      const IntegerVector& resp, const NumericVector& reward, const NumericVector& rt) {
    double k_fast = hyper[0], k_slow = hyper[1];
    double lambda_sparse = hyper[2];
    int K_sa = (int)hyper[3];
    
    double a = std::exp(phi[0]), tnd = 0.8 / (1.0 + std::exp(-phi[1]));
    double v_ctx = std::exp(phi[2]), alpha_ctx = 1.0 / (1.0 + std::exp(-phi[3]));
    double gamma = std::exp(phi[4]), tau_omega = 1.0 / (1.0 + std::exp(-phi[5]));
    double lambda_sa_temp = std::exp(phi[6]);
    double omega_thresh = 1.0 / (1.0 + std::exp(-phi[7])); // Sigmoid bounded (0, 1)
    
    int N = 32;
    int num_blocks = 4;
    int block_size = N / num_blocks;
    
    std::vector<double> h(N,0), Z_fast(N,0), Z_slow(N,0), Z(N,0), W(N,0), W_exp(N,0), frac_mem(N,0);
    std::vector<int> S(num_blocks,0);
    double Omega = 0.0;
    
    std::mt19937 gen(42);
    std::uniform_real_distribution<> unif(0.0,1.0);
    std::normal_distribution<> norm(0.0,1.0);
    for(int i=0;i<N;++i){W_exp[i]=norm(gen);W[i]=norm(gen);}
    
    // Heterogeneous Fractional Alphas
    std::vector<double> frac_alpha(N, 0.0);
    for(int i=0; i<N; ++i) frac_alpha[i] = 0.1 + 0.8 * (double(i) / double(N - 1));
    
    double nll=0.0, prev_E=0.0;
    double Q[2]={0.5,0.5};
    
    for(int t=0;t<resp.size();++t){
        int ch=resp[t]-1; double R=(reward[t]==1.0)?1.0:0.0;
        
        // Non-homogeneous Fractional Expansion & Dual Trace
        for(int i=0;i<N;++i){
            frac_mem[i] = frac_alpha[i] * frac_mem[i] + (1.0 - frac_alpha[i]) * W_exp[i] * Q[ch];
            h[i] = std::tanh(frac_mem[i]);
            Z_fast[i] = Z_fast[i] * k_fast + h[i];
            Z_slow[i] = Z_slow[i] * k_slow + h[i];
            Z[i] = 0.5 * Z_fast[i] + 0.5 * Z_slow[i];
        }
        
        // Thresholded Leaky Integrator for Coupling
        double e_io = std::abs(prev_E);
        double f_io = (e_io > omega_thresh) ? (e_io - omega_thresh) : 0.0;
        Omega = Omega * (1.0 - tau_omega) + gamma * f_io;
        
        double cb=0.0;
        for(int b=0; b<num_blocks; ++b){
            if(S[b]==1){
                for(int i=b*block_size; i<(b+1)*block_size; ++i) cb += W[i]*Z[i];
            }
        }
        
        // Drift Gating
        double v_eff = v_ctx*(Q[1]-Q[0]) + Omega*cb;
        if(std::abs(v_eff)<1e-4) v_eff = v_eff>=0?1e-4:-1e-4;
        
        nll -= std::log(wfpt_pdf(rt[t], resp[t], v_eff, a, tnd, 0.5));
        
        prev_E = R - Q[ch];
        Q[ch] += alpha_ctx * prev_E;
        
        // Modular SA Masking with Sparsity Constraint
        if(std::abs(prev_E)>0.01){
            double cF=0.0, cL1=0.0;
            for(int b=0; b<num_blocks; ++b) {
                if(S[b]==1){
                    cL1 += 1.0; // Block-level sparsity
                    for(int i=b*block_size; i<(b+1)*block_size; ++i) cF += W[i]*Z[i];
                }
            }
            cF = std::pow(-prev_E - cF, 2) + lambda_sparse * cL1;
            
            for(int k=1;k<=K_sa;++k){
                double temp = std::abs(prev_E) * std::exp(-lambda_sa_temp*k);
                int idx = gen()%num_blocks; // Flip an entire sagittal band
                S[idx] = 1-S[idx];
                
                double nF=0.0, nL1=0.0;
                for(int b=0; b<num_blocks; ++b) {
                    if(S[b]==1){
                        nL1 += 1.0;
                        for(int i=b*block_size; i<(b+1)*block_size; ++i) nF += W[i]*Z[i];
                    }
                }
                nF = std::pow(-prev_E - nF, 2) + lambda_sparse * nL1;
                
                if(nF - cF < 0 || unif(gen) < std::exp(-(nF - cF)/std::max(temp,1e-10))) cF = nF;
                else S[idx] = 1-S[idx]; // Revert
            }
        }
    }
    return nll;
}
