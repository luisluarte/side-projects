#include <Rcpp.h>
#include <cmath>
#include <vector>
#include <random>

using namespace Rcpp;

// [[Rcpp::export]]
NumericMatrix ext_thermo_sudoku(const std::vector<double>& phi, const std::vector<double>& hyper, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    int struct_id = (int)hyper[0];
    double kappa = hyper[1];
    int K_sa = (int)hyper[2];
    
    double a = std::exp(phi[0]), tnd = 1.0 / (1.0 + std::exp(-phi[1]));
    double v_ctx = std::exp(phi[2]);
    double gamma = std::exp(phi[4]), tau_omega = 1.0 / (1.0 + std::exp(-phi[5]));
    double lambda_sa = std::exp(phi[6]);
    
    int N = 32; 
    std::vector<double> h(N, 0.0), Z(N, 0.0);
    std::vector<int> S(N, 0); 
    std::vector<double> W(N, 0.0);
    double Omega = 0.0;
    
    std::mt19937 gen(42);
    std::uniform_real_distribution<> unif(0.0, 1.0);
    std::normal_distribution<> norm(0.0, 1.0);
    
    std::vector<double> W_exp(N);
    for(int i=0; i<N; ++i) { W_exp[i] = norm(gen); W[i] = norm(gen); }
    
    double prev_E_IO = 0.0;
    double Q_ctx[2] = {0.5, 0.5};
    NumericMatrix res(resp.size(), 2);
    
    for(int t=0; t<resp.size(); ++t) {
        int ch = resp[t] - 1; double R = (out[t] == 1) ? 1.0 : 0.0;
        
        for(int i=0; i<N; ++i) {
            double act = W_exp[i] * Q_ctx[ch] + 0.1 * h[i];
            if(struct_id == 1) h[i] = std::tanh(act);
            else if(struct_id == 2) h[i] = act > 0 ? act : 0.0;
            else if(struct_id == 3) h[i] = 1.0 / (1.0 + std::exp(-act));
            else h[i] = act;
        }
        
        for(int i=0; i<N; ++i) Z[i] = Z[i] * kappa + h[i];
        Omega = Omega * (1.0 - tau_omega) + gamma * std::abs(prev_E_IO);
        
        double cb_drive = 0.0;
        for(int i=0; i<N; ++i) if(S[i] == 1) cb_drive += W[i] * Z[i];
        
        double v_eff = v_ctx * (Q_ctx[1] - Q_ctx[0]) + Omega * cb_drive;
        double safe_v = std::abs(v_eff) < 1e-4 ? (v_eff >= 0 ? 1e-4 : -1e-4) : v_eff;
        
        double exp_rt = tnd + (a / std::abs(safe_v)) * std::tanh(a * std::abs(safe_v) / 2.0);
        if(std::isnan(exp_rt)) exp_rt = tnd + (a*a)/4.0;
        double p1 = 1.0 / (1.0 + std::exp(-a * safe_v)); // upper boundary
        
        res(t, 0) = exp_rt;
        res(t, 1) = p1;
        
        prev_E_IO = R - std::max(Q_ctx[ch], 0.01);
        Q_ctx[ch] += 0.1 * prev_E_IO;
        
        if(std::abs(prev_E_IO) > 0.01) {
            double current_F = 0.0;
            for(int i=0; i<N; ++i) if(S[i] == 1) current_F += W[i] * Z[i];
            current_F = std::pow(-prev_E_IO - current_F, 2);
            
            for(int k=1; k<=K_sa; ++k) {
                double temp = std::abs(prev_E_IO) * std::exp(-lambda_sa * k);
                int idx = gen() % N;
                S[idx] = 1 - S[idx]; 
                
                double new_F = 0.0;
                for(int i=0; i<N; ++i) if(S[i] == 1) new_F += W[i] * Z[i];
                new_F = std::pow(-prev_E_IO - new_F, 2);
                
                double dF = new_F - current_F;
                if(dF < 0 || unif(gen) < std::exp(-dF / temp)) {
                    current_F = new_F; 
                } else {
                    S[idx] = 1 - S[idx]; 
                }
            }
        }
    }
    return res;
}
