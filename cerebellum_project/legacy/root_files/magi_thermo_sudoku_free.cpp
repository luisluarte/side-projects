#include <Rcpp.h>
#include <cmath>
#include <vector>
#include <random>

using namespace Rcpp;

double wiener_pdf2(double t, int resp, double v, double a, double t0, double w) {
    if (t <= t0) return 1e-10;
    double tt = t - t0; double k = 0.0; double p = 0.0; double err = 1e-10;
    if (resp == 2) { v = -v; w = 1.0 - w; }
    while (true) {
        k++;
        double term = k * std::sin(k * M_PI * w) * std::exp(-0.5 * (v * v * tt) - 0.5 * (k * k * M_PI * M_PI * tt / (a * a))) * std::exp(v * a * w) * M_PI / (a * a);
        p += term;
        if (std::abs(term) < err || k > 50) break;
    }
    return p > 1e-10 ? p : 1e-10;
}

// [[Rcpp::export]]
double get_nll_thermo_sudoku_free(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    int struct_id = 1; // CC_Model_069 is Tanh
    
    double a = std::exp(phi[0]), tnd = 1.0 / (1.0 + std::exp(-phi[1]));
    double v_ctx = std::exp(phi[2]);
    double gamma = std::exp(phi[4]), tau_omega = 1.0 / (1.0 + std::exp(-phi[5]));
    double lambda_sa = std::exp(phi[6]);
    
    // Free parameters for kappa and K_sa
    double kappa = 1.0 / (1.0 + std::exp(-phi[7]));
    int K_sa = (int)(std::round(50.0 / (1.0 + std::exp(-phi[8])))) + 1;
    if (K_sa < 1) K_sa = 1;
    if (K_sa > 50) K_sa = 50;

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
    
    double nll = 0.0;
    double prev_E_IO = 0.0;
    double Q_ctx[2] = {0.5, 0.5};
    
    for(int t=0; t<resp.size(); ++t) {
        int ch = resp[t] - 1; double R = (out[t] == 1) ? 1.0 : 0.0;
        
        for(int i=0; i<N; ++i) {
            double act = W_exp[i] * Q_ctx[ch] + 0.1 * h[i];
            h[i] = std::tanh(act);
        }
        for(int i=0; i<N; ++i) Z[i] = Z[i] * kappa + h[i];
        
        Omega = Omega * (1.0 - tau_omega) + gamma * std::abs(prev_E_IO);
        
        double cb_drive = 0.0;
        for(int i=0; i<N; ++i) if(S[i] == 1) cb_drive += W[i] * Z[i];
        
        double v_eff = v_ctx * (Q_ctx[1] - Q_ctx[0]) + Omega * cb_drive;
        double safe_v = std::abs(v_eff) < 1e-4 ? (v_eff >= 0 ? 1e-4 : -1e-4) : v_eff;
        
        int w_choice = (resp[t] == 2) ? 1 : 2;
        double pdf = wiener_pdf2(rt[t], w_choice, safe_v, a, tnd, 0.5);
        double p_ch = (ch == 1) ? (std::exp(safe_v) / (std::exp(-safe_v) + std::exp(safe_v))) : (std::exp(-safe_v) / (std::exp(-safe_v) + std::exp(safe_v)));
        if (p_ch < 1e-5) p_ch = 1e-5;
        nll -= (std::log(pdf) + std::log(p_ch)); // Use Joint Density
        
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
    return nll;
}

// [[Rcpp::export]]
NumericMatrix ext_thermo_sudoku_free(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    double a = std::exp(phi[0]), tnd = 1.0 / (1.0 + std::exp(-phi[1]));
    double v_ctx = std::exp(phi[2]);
    double gamma = std::exp(phi[4]), tau_omega = 1.0 / (1.0 + std::exp(-phi[5]));
    double lambda_sa = std::exp(phi[6]);
    
    double kappa = 1.0 / (1.0 + std::exp(-phi[7]));
    int K_sa = (int)(std::round(50.0 / (1.0 + std::exp(-phi[8])))) + 1;
    if (K_sa < 1) K_sa = 1;
    if (K_sa > 50) K_sa = 50;

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
    
    NumericMatrix res(resp.size(), 2);
    double prev_E_IO = 0.0;
    double Q_ctx[2] = {0.5, 0.5};
    
    for(int t=0; t<resp.size(); ++t) {
        int ch = resp[t] - 1; double R = (out[t] == 1) ? 1.0 : 0.0;
        
        for(int i=0; i<N; ++i) {
            double act = W_exp[i] * Q_ctx[ch] + 0.1 * h[i];
            h[i] = std::tanh(act);
        }
        for(int i=0; i<N; ++i) Z[i] = Z[i] * kappa + h[i];
        
        Omega = Omega * (1.0 - tau_omega) + gamma * std::abs(prev_E_IO);
        
        double cb_drive = 0.0;
        for(int i=0; i<N; ++i) if(S[i] == 1) cb_drive += W[i] * Z[i];
        
        double v_eff = v_ctx * (Q_ctx[1] - Q_ctx[0]) + Omega * cb_drive;
        double safe_v = std::abs(v_eff) < 1e-4 ? (v_eff >= 0 ? 1e-4 : -1e-4) : v_eff;
        
        res(t, 0) = tnd + (a / std::abs(safe_v)); // expected RT
        res(t, 1) = std::exp(safe_v) / (std::exp(-safe_v) + std::exp(safe_v)); // probability of 1
        
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
                if(new_F - current_F < 0 || unif(gen) < std::exp(-(new_F - current_F) / temp)) current_F = new_F; 
                else S[idx] = 1 - S[idx]; 
            }
        }
    }
    return res;
}
