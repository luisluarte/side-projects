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
// UNIVERSAL 100-MODEL WFPT ENGINE
// ==========================================
// phi: [0]a, [1]tnd, [2]v_ctx, [3]alpha_ctx, [4]gamma, [5]tau_omega, [6]lambda_sa
// hyper: [0]expansion(1=delay,2=reservoir,3=fractional),
//        [1]trace(1=mono,2=dual), [2]mask(1=indep,2=modular),
//        [3]integration(1=drift,2=boundary), [4]kappa, [5]K_sa

// [[Rcpp::export]]
double get_nll_100(const std::vector<double>& phi, const std::vector<double>& hyper,
                   const IntegerVector& resp, const NumericVector& reward, const NumericVector& rt) {
    int exp_type = (int)hyper[0];
    int trace_type = (int)hyper[1];
    int mask_type = (int)hyper[2];
    int integ_type = (int)hyper[3];
    double kappa = hyper[4];
    int K_sa = (int)hyper[5];
    
    double a = std::exp(phi[0]);
    double tnd = 0.8 / (1.0 + std::exp(-phi[1]));
    double v_ctx = std::exp(phi[2]);
    double alpha_ctx = 1.0 / (1.0 + std::exp(-phi[3]));
    double gamma = std::exp(phi[4]);
    double tau_omega = 1.0 / (1.0 + std::exp(-phi[5]));
    double lambda_sa = std::exp(phi[6]);
    
    int N = 32;
    int n_modules = 4;
    int mod_sz = N / n_modules;
    std::vector<double> h(N,0), Z(N,0), Z_slow(N,0), W(N,0), W_exp(N,0);
    std::vector<int> S(N,0);
    std::vector<double> W_rec(N*N, 0.0);
    std::vector<double> frac_mem(N, 0.0);
    double Omega = 0.0;
    
    std::mt19937 gen(42);
    std::uniform_real_distribution<> unif(0.0,1.0);
    std::normal_distribution<> norm(0.0,1.0);
    for(int i=0;i<N;++i){W_exp[i]=norm(gen);W[i]=norm(gen);}
    if(exp_type==2){
        double sparsity=0.1;
        for(int i=0;i<N;++i)for(int j=0;j<N;++j)
            if(unif(gen)<sparsity) W_rec[i*N+j]=norm(gen)*0.9;
    }
    
    double nll=0.0, prev_E=0.0;
    double Q[2]={0.5,0.5};
    
    for(int t=0;t<resp.size();++t){
        int ch=resp[t]-1;
        double R=(reward[t]==1.0)?1.0:0.0;
        
        // === AXIS 1: Expansion ===
        if(exp_type==1){ // Delay-Line
            for(int i=N-1;i>0;--i) h[i]=h[i-1];
            h[0]=std::tanh(W_exp[0]*Q[ch]);
        } else if(exp_type==2){ // Recurrent Reservoir
            std::vector<double> h_new(N,0);
            for(int i=0;i<N;++i){
                double inp=W_exp[i]*Q[ch];
                for(int j=0;j<N;++j) inp+=W_rec[i*N+j]*h[j];
                h_new[i]=std::tanh(inp);
            }
            h=h_new;
        } else { // Fractional Derivative
            double frac_alpha=0.7;
            for(int i=0;i<N;++i){
                double inp=W_exp[i]*Q[ch];
                frac_mem[i]=frac_alpha*frac_mem[i]+(1.0-frac_alpha)*inp;
                h[i]=std::tanh(frac_mem[i]);
            }
        }
        
        // === AXIS 2: Trace ===
        if(trace_type==1){ // Mono-exponential
            for(int i=0;i<N;++i) Z[i]=Z[i]*kappa+h[i];
        } else { // Dual-Cascaded
            double kappa_fast=0.3, kappa_slow=kappa;
            for(int i=0;i<N;++i){
                Z[i]=Z[i]*kappa_fast+h[i];
                Z_slow[i]=Z_slow[i]*kappa_slow+Z[i];
            }
        }
        
        // Coupling
        Omega=Omega*(1.0-tau_omega)+gamma*std::abs(prev_E);
        double cb=0.0;
        double* Zptr=(trace_type==2)?Z_slow.data():Z.data();
        for(int i=0;i<N;++i) if(S[i]==1) cb+=W[i]*Zptr[i];
        
        // === AXIS 4: Cortical Integration ===
        double v_eff, a_eff=a;
        if(integ_type==1){ // Drift Gating
            v_eff=v_ctx*(Q[1]-Q[0])+Omega*cb;
        } else { // Boundary Expansion
            v_eff=v_ctx*(Q[1]-Q[0]);
            a_eff=a+std::abs(Omega*cb)*0.5;
        }
        if(std::abs(v_eff)<1e-4) v_eff=v_eff>=0?1e-4:-1e-4;
        
        nll -= std::log(wfpt_pdf(rt[t], resp[t], v_eff, a_eff, tnd, 0.5));
        
        prev_E=R-Q[ch];
        Q[ch]+=alpha_ctx*prev_E;
        
        // === AXIS 3: SA Mask ===
        if(std::abs(prev_E)>0.01){
            double cF=0;
            for(int i=0;i<N;++i) if(S[i]==1) cF+=W[i]*Zptr[i];
            cF=std::pow(-prev_E-cF,2);
            for(int k=1;k<=K_sa;++k){
                double temp=std::abs(prev_E)*std::exp(-lambda_sa*k);
                if(mask_type==1){ // Independent Node
                    int idx=gen()%N;
                    S[idx]=1-S[idx];
                    double nF=0;
                    for(int i=0;i<N;++i) if(S[i]==1) nF+=W[i]*Zptr[i];
                    nF=std::pow(-prev_E-nF,2);
                    if(nF-cF<0||unif(gen)<std::exp(-(nF-cF)/std::max(temp,1e-10))) cF=nF;
                    else S[idx]=1-S[idx];
                } else { // Modular Microzone
                    int mod=gen()%n_modules;
                    for(int i=mod*mod_sz;i<(mod+1)*mod_sz;++i) S[i]=1-S[i];
                    double nF=0;
                    for(int i=0;i<N;++i) if(S[i]==1) nF+=W[i]*Zptr[i];
                    nF=std::pow(-prev_E-nF,2);
                    if(nF-cF<0||unif(gen)<std::exp(-(nF-cF)/std::max(temp,1e-10))) cF=nF;
                    else for(int i=mod*mod_sz;i<(mod+1)*mod_sz;++i) S[i]=1-S[i];
                }
            }
        }
    }
    return nll;
}

// [[Rcpp::export]]
NumericMatrix ext_100(const std::vector<double>& phi, const std::vector<double>& hyper,
                      const IntegerVector& resp, const NumericVector& reward, const NumericVector& rt) {
    int exp_type=(int)hyper[0];int trace_type=(int)hyper[1];int mask_type=(int)hyper[2];
    int integ_type=(int)hyper[3];double kappa=hyper[4];int K_sa=(int)hyper[5];
    double a=std::exp(phi[0]);double tnd=0.8/(1.0+std::exp(-phi[1]));
    double v_ctx=std::exp(phi[2]);double alpha_ctx=1.0/(1.0+std::exp(-phi[3]));
    double gamma=std::exp(phi[4]);double tau_omega=1.0/(1.0+std::exp(-phi[5]));
    double lambda_sa=std::exp(phi[6]);
    int N=32;int n_modules=4;int mod_sz=N/n_modules;
    std::vector<double> h(N,0),Z(N,0),Z_slow(N,0),W(N,0),W_exp(N,0);
    std::vector<int> S(N,0);std::vector<double> W_rec(N*N,0.0);
    std::vector<double> frac_mem(N,0.0);double Omega=0.0;
    std::mt19937 gen(42);std::uniform_real_distribution<> unif(0.0,1.0);
    std::normal_distribution<> norm(0.0,1.0);
    for(int i=0;i<N;++i){W_exp[i]=norm(gen);W[i]=norm(gen);}
    if(exp_type==2){double sp=0.1;for(int i=0;i<N;++i)for(int j=0;j<N;++j)if(unif(gen)<sp)W_rec[i*N+j]=norm(gen)*0.9;}
    NumericMatrix res(resp.size(),2);double prev_E=0.0;double Q[2]={0.5,0.5};
    for(int t=0;t<resp.size();++t){
        int ch=resp[t]-1;double R=(reward[t]==1.0)?1.0:0.0;
        if(exp_type==1){for(int i=N-1;i>0;--i)h[i]=h[i-1];h[0]=std::tanh(W_exp[0]*Q[ch]);}
        else if(exp_type==2){std::vector<double> hn(N,0);for(int i=0;i<N;++i){double inp=W_exp[i]*Q[ch];for(int j=0;j<N;++j)inp+=W_rec[i*N+j]*h[j];hn[i]=std::tanh(inp);}h=hn;}
        else{double fa=0.7;for(int i=0;i<N;++i){frac_mem[i]=fa*frac_mem[i]+(1.0-fa)*W_exp[i]*Q[ch];h[i]=std::tanh(frac_mem[i]);}}
        if(trace_type==1){for(int i=0;i<N;++i)Z[i]=Z[i]*kappa+h[i];}
        else{double kf=0.3;for(int i=0;i<N;++i){Z[i]=Z[i]*kf+h[i];Z_slow[i]=Z_slow[i]*kappa+Z[i];}}
        Omega=Omega*(1.0-tau_omega)+gamma*std::abs(prev_E);
        double cb=0;double*Zp=(trace_type==2)?Z_slow.data():Z.data();
        for(int i=0;i<N;++i)if(S[i]==1)cb+=W[i]*Zp[i];
        double v_eff,a_eff=a;
        if(integ_type==1){v_eff=v_ctx*(Q[1]-Q[0])+Omega*cb;}else{v_eff=v_ctx*(Q[1]-Q[0]);a_eff=a+std::abs(Omega*cb)*0.5;}
        if(std::abs(v_eff)<1e-4)v_eff=v_eff>=0?1e-4:-1e-4;
        res(t,1)=1.0/(1.0+std::exp(-v_eff*a_eff));
        res(t,0)=tnd+(a_eff/(2.0*v_eff))*std::tanh(v_eff*a_eff/2.0);
        prev_E=R-Q[ch];Q[ch]+=alpha_ctx*prev_E;
        if(std::abs(prev_E)>0.01){
            double cF=0;for(int i=0;i<N;++i)if(S[i]==1)cF+=W[i]*Zp[i];cF=(prev_E+cF)*(prev_E+cF);
            for(int k=1;k<=K_sa;++k){double temp=std::abs(prev_E)*std::exp(-lambda_sa*k);
                if(mask_type==1){int idx=gen()%N;S[idx]=1-S[idx];double nF=0;for(int i=0;i<N;++i)if(S[i]==1)nF+=W[i]*Zp[i];nF=(prev_E+nF)*(prev_E+nF);if(nF-cF<0||unif(gen)<std::exp(-(nF-cF)/std::max(temp,1e-10)))cF=nF;else S[idx]=1-S[idx];}
                else{int mod=gen()%n_modules;for(int i=mod*mod_sz;i<(mod+1)*mod_sz;++i)S[i]=1-S[i];double nF=0;for(int i=0;i<N;++i)if(S[i]==1)nF+=W[i]*Zp[i];nF=(prev_E+nF)*(prev_E+nF);if(nF-cF<0||unif(gen)<std::exp(-(nF-cF)/std::max(temp,1e-10)))cF=nF;else for(int i=mod*mod_sz;i<(mod+1)*mod_sz;++i)S[i]=1-S[i];}
            }
        }
    }
    return res;
}
