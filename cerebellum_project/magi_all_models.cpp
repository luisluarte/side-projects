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

// ---------------------------------------------------------
// MODEL BASE: Q-Learning Wald
// ---------------------------------------------------------
// phi: [0]a, [1]tnd, [2]v_ctx, [3]alpha_ctx
// [[Rcpp::export]]
double get_nll_base(const std::vector<double>& phi, const IntegerVector& resp, const NumericVector& reward, const NumericVector& rt) {
    double a = std::exp(phi[0]), tnd = 0.8 / (1.0 + std::exp(-phi[1]));
    double v_ctx = std::exp(phi[2]), alpha_ctx = 1.0 / (1.0 + std::exp(-phi[3]));
    double nll = 0.0, Q[2] = {0.5, 0.5};
    for(int t=0; t<resp.size(); ++t) {
        int ch = resp[t]-1; double R = (reward[t]==1.0)?1.0:0.0;
        double v_eff = v_ctx * (Q[1] - Q[0]);
        if(std::abs(v_eff)<1e-4) v_eff = v_eff>=0?1e-4:-1e-4;
        nll -= std::log(wfpt_pdf(rt[t], resp[t], v_eff, a, tnd, 0.5));
        Q[ch] += alpha_ctx * (R - Q[ch]);
    }
    return nll;
}

// [[Rcpp::export]]
DataFrame sim_base(const std::vector<double>& phi, const IntegerVector& resp, const NumericVector& reward, const NumericVector& rt) {
    double a = std::exp(phi[0]), tnd = 0.8 / (1.0 + std::exp(-phi[1]));
    double v_ctx = std::exp(phi[2]), alpha_ctx = 1.0 / (1.0 + std::exp(-phi[3]));
    NumericVector pred_v(resp.size()), pred_a(resp.size()), Q_diff(resp.size());
    double Q[2] = {0.5, 0.5};
    for(int t=0; t<resp.size(); ++t) {
        int ch = resp[t]-1; double R = (reward[t]==1.0)?1.0:0.0;
        double v_eff = v_ctx * (Q[1] - Q[0]);
        pred_v[t] = v_eff; pred_a[t] = a; Q_diff[t] = Q[1]-Q[0];
        Q[ch] += alpha_ctx * (R - Q[ch]);
    }
    return DataFrame::create(_["v"]=pred_v, _["a"]=pred_a, _["tnd"]=tnd, _["Q_diff"]=Q_diff);
}


// ---------------------------------------------------------
// MODEL 005: Static Opponent Process
// ---------------------------------------------------------
// [[Rcpp::export]]
double get_nll_005(const std::vector<double>& phi, const std::vector<double>& hyper, const IntegerVector& resp, const NumericVector& reward, const NumericVector& rt) {
    double lambda_sparse = hyper[0], beta_ising = hyper[1]; int K_sa = (int)hyper[2];
    double a = std::exp(phi[0]), tnd = 0.8 / (1.0 + std::exp(-phi[1]));
    double v_ctx = std::exp(phi[2]), alpha_ctx = 1.0 / (1.0 + std::exp(-phi[3]));
    double alpha_pc = 1.0 / (1.0 + std::exp(-phi[4])); double gamma = std::exp(phi[5]);
    double lambda_sa_temp = std::exp(phi[6]);
    
    int N = 32; std::vector<double> h(N,0), Z(N,0), W(N,0), W_exp(N,0), frac_mem(N,0), W_PC(N,0);
    std::vector<int> S(N,0);
    std::mt19937 gen(42); std::uniform_real_distribution<> unif(0.0,1.0); std::normal_distribution<> norm(0.0,1.0);
    for(int i=0;i<N;++i){W_exp[i]=norm(gen);W[i]=norm(gen);}
    std::vector<double> frac_alpha(N, 0.0), kappa_vec(N, 0.0);
    for(int i=0; i<N; ++i) { frac_alpha[i] = 0.1+0.8*(i/31.0); kappa_vec[i] = 0.1+0.89*(i/31.0); }
    
    double nll=0.0, prev_E=0.0; double Q[2]={0.5,0.5}; int prev_ch = 0;
    
    for(int t=0;t<resp.size();++t){
        int ch=resp[t]-1; double R=(reward[t]==1.0)?1.0:0.0;
        for(int i=0;i<N;++i){
            frac_mem[i] = frac_alpha[i] * frac_mem[i] + (1.0 - frac_alpha[i]) * W_exp[i] * Q[ch];
            h[i] = std::tanh(frac_mem[i]); Z[i] = kappa_vec[i] * Z[i] + h[i];
        }
        double cb0 = 0.0, cb1 = 0.0;
        for(int i=0; i<16; ++i) if(S[i]==1) cb0 += W_PC[i]*Z[i];
        for(int i=16; i<32; ++i) if(S[i]==1) cb1 += W_PC[i]*Z[i];
        double v_eff = v_ctx*(Q[1]-Q[0]) + gamma*(cb1 - cb0);
        if(std::abs(v_eff)<1e-4) v_eff = v_eff>=0?1e-4:-1e-4;
        nll -= std::log(wfpt_pdf(rt[t], resp[t], v_eff, a, tnd, 0.5));
        
        prev_E = R - Q[ch]; Q[ch] += alpha_ctx * prev_E; prev_ch = ch;
        double err0 = (prev_ch == 0) ? prev_E : 0.0; double err1 = (prev_ch == 1) ? prev_E : 0.0;
        for(int i=0; i<16; ++i) { W_PC[i] += alpha_pc * Z[i] * err0; }
        for(int i=16; i<32; ++i) { W_PC[i] += alpha_pc * Z[i] * err1; }
    }
    return nll;
}

// [[Rcpp::export]]
DataFrame sim_005(const std::vector<double>& phi, const std::vector<double>& hyper, const IntegerVector& resp, const NumericVector& reward, const NumericVector& rt) {
    double a = std::exp(phi[0]), tnd = 0.8 / (1.0 + std::exp(-phi[1]));
    double v_ctx = std::exp(phi[2]), gamma = std::exp(phi[5]);
    double alpha_ctx = 1.0 / (1.0 + std::exp(-phi[3]));
    double alpha_pc = 1.0 / (1.0 + std::exp(-phi[4]));
    int N = 32; std::vector<double> h(N,0), Z(N,0), W(N,0), W_exp(N,0), frac_mem(N,0), W_PC(N,0);
    std::vector<int> S(N,1);
    std::mt19937 gen(42); std::normal_distribution<> norm(0.0,1.0);
    for(int i=0;i<N;++i){W_exp[i]=norm(gen);W[i]=norm(gen);}
    std::vector<double> frac_alpha(N, 0.0), kappa_vec(N, 0.0);
    for(int i=0; i<N; ++i) { frac_alpha[i] = 0.1+0.8*(i/31.0); kappa_vec[i] = 0.1+0.89*(i/31.0); }
    
    NumericVector pred_v(resp.size()), pred_a(resp.size()), Q_diff(resp.size()), cb_diff(resp.size());
    double Q[2]={0.5,0.5}; int prev_ch = 0; double prev_E=0.0;
    for(int t=0;t<resp.size();++t){
        int ch=resp[t]-1; double R=(reward[t]==1.0)?1.0:0.0;
        for(int i=0;i<N;++i){
            frac_mem[i] = frac_alpha[i] * frac_mem[i] + (1.0 - frac_alpha[i]) * W_exp[i] * Q[ch];
            Z[i] = kappa_vec[i] * Z[i] + std::tanh(frac_mem[i]);
        }
        double cb0 = 0.0, cb1 = 0.0;
        for(int i=0; i<16; ++i) if(S[i]==1) cb0 += W_PC[i]*Z[i];
        for(int i=16; i<32; ++i) if(S[i]==1) cb1 += W_PC[i]*Z[i];
        
        pred_v[t] = v_ctx*(Q[1]-Q[0]) + gamma*(cb1 - cb0);
        pred_a[t] = a; Q_diff[t] = Q[1]-Q[0]; cb_diff[t] = cb1 - cb0;
        
        prev_E = R - Q[ch]; Q[ch] += alpha_ctx * prev_E; prev_ch = ch;
        for(int i=0; i<16; ++i) { W_PC[i] += alpha_pc * Z[i] * ((prev_ch == 0) ? prev_E : 0.0); }
        for(int i=16; i<32; ++i) { W_PC[i] += alpha_pc * Z[i] * ((prev_ch == 1) ? prev_E : 0.0); }
    }
    return DataFrame::create(_["v"]=pred_v, _["a"]=pred_a, _["tnd"]=tnd, _["Q_diff"]=Q_diff, _["cb_diff"]=cb_diff);
}


// ---------------------------------------------------------
// MODEL 006: Continuous ITI + Epistemic Boundary
// ---------------------------------------------------------
// [[Rcpp::export]]
double get_nll_006(const std::vector<double>& phi, const std::vector<double>& hyper, const IntegerVector& resp, const NumericVector& reward, const NumericVector& rt, const NumericVector& iti) {
    double a_base = std::exp(phi[0]), tnd = 0.8 / (1.0 + std::exp(-phi[1]));
    double v_ctx = std::exp(phi[2]), alpha_ctx = 1.0 / (1.0 + std::exp(-phi[3]));
    double alpha_pc = 1.0 / (1.0 + std::exp(-phi[4])); double gamma = std::exp(phi[5]);
    double tau_decay = std::exp(phi[7]), w_u = std::exp(phi[8]);
    
    int N = 32; std::vector<double> h(N,0), Z(N,0), W(N,0), W_exp(N,0), frac_mem(N,0), W_PC(N,0);
    std::vector<int> S(N,0);
    std::mt19937 gen(42); std::normal_distribution<> norm(0.0,1.0);
    for(int i=0;i<N;++i){W_exp[i]=norm(gen);W[i]=norm(gen);}
    std::vector<double> frac_alpha(N, 0.0), kappa_vec(N, 0.0);
    for(int i=0; i<N; ++i) { frac_alpha[i] = 0.1+0.8*(i/31.0); kappa_vec[i] = 0.1+0.89*(i/31.0); }
    
    double nll=0.0, prev_E=0.0; double Q[2]={0.5,0.5}; int prev_ch = 0;
    
    for(int t=0;t<resp.size();++t){
        int ch=resp[t]-1; double R=(reward[t]==1.0)?1.0:0.0;
        double current_iti = (t == 0 || std::isnan(iti[t])) ? 1.0 : iti[t];
        double physical_decay = std::exp(-current_iti / tau_decay);
        
        for(int i=0;i<N;++i){
            frac_mem[i] = frac_alpha[i] * frac_mem[i] + (1.0 - frac_alpha[i]) * W_exp[i] * Q[ch];
            Z[i] = physical_decay * kappa_vec[i] * Z[i] + std::tanh(frac_mem[i]);
        }
        double cb0 = 0.0, cb1 = 0.0;
        for(int i=0; i<16; ++i) if(S[i]==1) cb0 += W_PC[i]*Z[i];
        for(int i=16; i<32; ++i) if(S[i]==1) cb1 += W_PC[i]*Z[i];
        
        double v_eff = v_ctx*(Q[1]-Q[0]) + gamma*(cb1 - cb0);
        if(std::abs(v_eff)<1e-4) v_eff = v_eff>=0?1e-4:-1e-4;
        double a_dyn = a_base + w_u * std::abs(cb0) * std::abs(cb1);
        
        nll -= std::log(wfpt_pdf(rt[t], resp[t], v_eff, a_dyn, tnd, 0.5));
        
        prev_E = R - Q[ch]; Q[ch] += alpha_ctx * prev_E; prev_ch = ch;
        for(int i=0; i<16; ++i) { W_PC[i] += alpha_pc * Z[i] * ((prev_ch == 0) ? prev_E : 0.0); }
        for(int i=16; i<32; ++i) { W_PC[i] += alpha_pc * Z[i] * ((prev_ch == 1) ? prev_E : 0.0); }
    }
    return nll;
}

// [[Rcpp::export]]
DataFrame sim_006(const std::vector<double>& phi, const std::vector<double>& hyper, const IntegerVector& resp, const NumericVector& reward, const NumericVector& rt, const NumericVector& iti) {
    double a_base = std::exp(phi[0]), tnd = 0.8 / (1.0 + std::exp(-phi[1]));
    double v_ctx = std::exp(phi[2]), alpha_ctx = 1.0 / (1.0 + std::exp(-phi[3]));
    double alpha_pc = 1.0 / (1.0 + std::exp(-phi[4])); double gamma = std::exp(phi[5]);
    double tau_decay = std::exp(phi[7]), w_u = std::exp(phi[8]);
    
    int N = 32; std::vector<double> h(N,0), Z(N,0), W(N,0), W_exp(N,0), frac_mem(N,0), W_PC(N,0);
    std::vector<int> S(N,1);
    std::mt19937 gen(42); std::normal_distribution<> norm(0.0,1.0);
    for(int i=0;i<N;++i){W_exp[i]=norm(gen);W[i]=norm(gen);}
    std::vector<double> frac_alpha(N, 0.0), kappa_vec(N, 0.0);
    for(int i=0; i<N; ++i) { frac_alpha[i] = 0.1+0.8*(i/31.0); kappa_vec[i] = 0.1+0.89*(i/31.0); }
    
    NumericVector pred_v(resp.size()), pred_a(resp.size()), conflict(resp.size());
    double Q[2]={0.5,0.5}; int prev_ch = 0; double prev_E=0.0;
    
    for(int t=0;t<resp.size();++t){
        int ch=resp[t]-1; double R=(reward[t]==1.0)?1.0:0.0;
        double current_iti = (t == 0 || std::isnan(iti[t])) ? 1.0 : iti[t];
        double physical_decay = std::exp(-current_iti / tau_decay);
        
        for(int i=0;i<N;++i){
            frac_mem[i] = frac_alpha[i] * frac_mem[i] + (1.0 - frac_alpha[i]) * W_exp[i] * Q[ch];
            Z[i] = physical_decay * kappa_vec[i] * Z[i] + std::tanh(frac_mem[i]);
        }
        double cb0 = 0.0, cb1 = 0.0;
        for(int i=0; i<16; ++i) if(S[i]==1) cb0 += W_PC[i]*Z[i];
        for(int i=16; i<32; ++i) if(S[i]==1) cb1 += W_PC[i]*Z[i];
        
        pred_v[t] = v_ctx*(Q[1]-Q[0]) + gamma*(cb1 - cb0);
        conflict[t] = std::abs(cb0) * std::abs(cb1);
        pred_a[t] = a_base + w_u * conflict[t];
        
        prev_E = R - Q[ch]; Q[ch] += alpha_ctx * prev_E; prev_ch = ch;
        for(int i=0; i<16; ++i) { W_PC[i] += alpha_pc * Z[i] * ((prev_ch == 0) ? prev_E : 0.0); }
        for(int i=16; i<32; ++i) { W_PC[i] += alpha_pc * Z[i] * ((prev_ch == 1) ? prev_E : 0.0); }
    }
    return DataFrame::create(_["v"]=pred_v, _["a"]=pred_a, _["tnd"]=tnd, _["conflict"]=conflict);
}
