#include <Rcpp.h>
#include <cmath>
#include <vector>
#include <random>

using namespace Rcpp;

// Navarro & Fuss (2009) WFPT Defective Density
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
    double a = std::exp(phi[0]);
    double tnd = 0.8 / (1.0 + std::exp(-phi[1]));
    double k_v = std::exp(phi[2]);
    double alpha = 1.0 / (1.0 + std::exp(-phi[3]));
    double nll = 0.0;
    double Q[2] = {0.5, 0.5};
    for(int t=0; t<resp.size(); ++t) {
        int ch = resp[t]-1;
        double R = (reward[t]==1.0)?1.0:0.0;
        double v = k_v*(Q[1]-Q[0]);
        if(std::abs(v)<1e-4) v = v>=0?1e-4:-1e-4;
        nll -= std::log(wfpt_pdf(rt[t], resp[t], v, a, tnd, 0.5));
        Q[ch] += alpha*(R - Q[ch]);
    }
    return nll;
}

// [[Rcpp::export]]
NumericMatrix ext_base_w(const std::vector<double>& phi, const IntegerVector& resp, const NumericVector& reward, const NumericVector& rt) {
    double a = std::exp(phi[0]);
    double tnd = 0.8 / (1.0 + std::exp(-phi[1]));
    double k_v = std::exp(phi[2]);
    double alpha = 1.0 / (1.0 + std::exp(-phi[3]));
    NumericMatrix res(resp.size(), 2);
    double Q[2] = {0.5, 0.5};
    for(int t=0; t<resp.size(); ++t) {
        int ch = resp[t]-1;
        double R = (reward[t]==1.0)?1.0:0.0;
        double v = k_v*(Q[1]-Q[0]);
        if(std::abs(v)<1e-4) v = v>=0?1e-4:-1e-4;
        res(t,1) = 1.0/(1.0+std::exp(-v*a));
        res(t,0) = tnd + (a/(2.0*v))*std::tanh(v*a/2.0);
        Q[ch] += alpha*(R - Q[ch]);
    }
    return res;
}

// ==========================================
// SPARSITY-CONSTRAINED CC_WFPT_075
// ==========================================
// hyper: [0]kappa, [1]K_sa, [2]lambda_sparse

// [[Rcpp::export]]
double get_nll_sparse(const std::vector<double>& phi, const std::vector<double>& hyper,
                      const IntegerVector& resp, const NumericVector& reward, const NumericVector& rt) {
    double kappa = hyper[0];
    int K_sa = (int)hyper[1];
    double lambda_sparse = hyper[2];
    
    double a = std::exp(phi[0]), tnd = 0.8 / (1.0 + std::exp(-phi[1]));
    double v_ctx = std::exp(phi[2]), alpha_ctx = 1.0 / (1.0 + std::exp(-phi[3]));
    double gamma = std::exp(phi[4]), tau_omega = 1.0 / (1.0 + std::exp(-phi[5]));
    double lambda_sa_temp = std::exp(phi[6]);
    
    int N = 32;
    std::vector<double> h(N,0), Z(N,0), W(N,0), W_exp(N,0), frac_mem(N,0);
    std::vector<int> S(N,0);
    double Omega = 0.0;
    
    std::mt19937 gen(42);
    std::uniform_real_distribution<> unif(0.0,1.0);
    std::normal_distribution<> norm(0.0,1.0);
    for(int i=0;i<N;++i){W_exp[i]=norm(gen);W[i]=norm(gen);}
    
    double nll=0.0, prev_E=0.0;
    double Q[2]={0.5,0.5};
    
    for(int t=0;t<resp.size();++t){
        int ch=resp[t]-1; double R=(reward[t]==1.0)?1.0:0.0;
        
        // Fractional Expansion
        double frac_alpha = 0.7;
        for(int i=0;i<N;++i){
            frac_mem[i] = frac_alpha * frac_mem[i] + (1.0 - frac_alpha) * W_exp[i] * Q[ch];
            h[i] = std::tanh(frac_mem[i]);
        }
        
        // Mono-exponential Trace
        for(int i=0;i<N;++i) Z[i] = Z[i] * kappa + h[i];
        
        // Coupling
        Omega = Omega * (1.0 - tau_omega) + gamma * std::abs(prev_E);
        double cb=0.0;
        for(int i=0;i<N;++i) if(S[i]==1) cb+=W[i]*Z[i];
        
        // Drift Gating
        double v_eff = v_ctx*(Q[1]-Q[0]) + Omega*cb;
        if(std::abs(v_eff)<1e-4) v_eff = v_eff>=0?1e-4:-1e-4;
        
        nll -= std::log(wfpt_pdf(rt[t], resp[t], v_eff, a, tnd, 0.5));
        
        prev_E = R - Q[ch];
        Q[ch] += alpha_ctx * prev_E;
        
        // SA Masking with Sparsity Constraint
        if(std::abs(prev_E)>0.01){
            double cF=0.0, cL1=0.0;
            for(int i=0;i<N;++i) if(S[i]==1){ cF+=W[i]*Z[i]; cL1+=1.0; }
            cF = std::pow(-prev_E - cF, 2) + lambda_sparse * cL1;
            
            for(int k=1;k<=K_sa;++k){
                double temp = std::abs(prev_E) * std::exp(-lambda_sa_temp*k);
                int idx = gen()%N;
                S[idx] = 1-S[idx];
                
                double nF=0.0, nL1=0.0;
                for(int i=0;i<N;++i) if(S[i]==1){ nF+=W[i]*Z[i]; nL1+=1.0; }
                nF = std::pow(-prev_E - nF, 2) + lambda_sparse * nL1;
                
                if(nF - cF < 0 || unif(gen) < std::exp(-(nF - cF)/std::max(temp,1e-10))) cF = nF;
                else S[idx] = 1-S[idx];
            }
        }
    }
    return nll;
}

// [[Rcpp::export]]
NumericMatrix ext_sparse(const std::vector<double>& phi, const std::vector<double>& hyper,
                         const IntegerVector& resp, const NumericVector& reward, const NumericVector& rt) {
    double kappa = hyper[0];
    int K_sa = (int)hyper[1];
    double lambda_sparse = hyper[2];
    
    double a = std::exp(phi[0]), tnd = 0.8 / (1.0 + std::exp(-phi[1]));
    double v_ctx = std::exp(phi[2]), alpha_ctx = 1.0 / (1.0 + std::exp(-phi[3]));
    double gamma = std::exp(phi[4]), tau_omega = 1.0 / (1.0 + std::exp(-phi[5]));
    double lambda_sa_temp = std::exp(phi[6]);
    
    int N = 32;
    std::vector<double> h(N,0), Z(N,0), W(N,0), W_exp(N,0), frac_mem(N,0);
    std::vector<int> S(N,0);
    double Omega = 0.0;
    
    std::mt19937 gen(42);
    std::uniform_real_distribution<> unif(0.0,1.0);
    std::normal_distribution<> norm(0.0,1.0);
    for(int i=0;i<N;++i){W_exp[i]=norm(gen);W[i]=norm(gen);}
    
    NumericMatrix res(resp.size(),2);
    double prev_E=0.0; double Q[2]={0.5,0.5};
    
    for(int t=0;t<resp.size();++t){
        int ch=resp[t]-1; double R=(reward[t]==1.0)?1.0:0.0;
        
        double frac_alpha = 0.7;
        for(int i=0;i<N;++i){
            frac_mem[i] = frac_alpha * frac_mem[i] + (1.0 - frac_alpha) * W_exp[i] * Q[ch];
            h[i] = std::tanh(frac_mem[i]);
        }
        for(int i=0;i<N;++i) Z[i] = Z[i] * kappa + h[i];
        
        Omega = Omega * (1.0 - tau_omega) + gamma * std::abs(prev_E);
        double cb=0.0;
        for(int i=0;i<N;++i) if(S[i]==1) cb+=W[i]*Z[i];
        
        double v_eff = v_ctx*(Q[1]-Q[0]) + Omega*cb;
        if(std::abs(v_eff)<1e-4) v_eff = v_eff>=0?1e-4:-1e-4;
        
        res(t,1) = 1.0/(1.0+std::exp(-v_eff*a));
        res(t,0) = tnd + (a/(2.0*v_eff))*std::tanh(v_eff*a/2.0);
        
        prev_E = R - Q[ch];
        Q[ch] += alpha_ctx * prev_E;
        
        if(std::abs(prev_E)>0.01){
            double cF=0.0, cL1=0.0;
            for(int i=0;i<N;++i) if(S[i]==1){ cF+=W[i]*Z[i]; cL1+=1.0; }
            cF = std::pow(-prev_E - cF, 2) + lambda_sparse * cL1;
            for(int k=1;k<=K_sa;++k){
                double temp = std::abs(prev_E) * std::exp(-lambda_sa_temp*k);
                int idx = gen()%N;
                S[idx] = 1-S[idx];
                double nF=0.0, nL1=0.0;
                for(int i=0;i<N;++i) if(S[i]==1){ nF+=W[i]*Z[i]; nL1+=1.0; }
                nF = std::pow(-prev_E - nF, 2) + lambda_sparse * nL1;
                if(nF - cF < 0 || unif(gen) < std::exp(-(nF - cF)/std::max(temp,1e-10))) cF = nF;
                else S[idx] = 1-S[idx];
            }
        }
    }
    return res;
}
