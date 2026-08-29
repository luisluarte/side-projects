#include <Rcpp.h>
#include <cmath>
#include <vector>
#include <random>

using namespace Rcpp;

// Navarro & Fuss (2009) WFPT Defective Density
double wiener_pdf_nf(double t, int resp, double v, double a, double t0, double w) {
    if (t <= t0) return 1e-10;
    double tt = t - t0;
    // resp == 2 means hitting the UPPER boundary
    if (resp == 2) {
        v = -v;
        w = 1.0 - w;
    }
    double tau = tt / (a * a);
    if (tau < 1e-5) return 1e-10;
    
    double p = 0.0;
    if (tau > 0.15) { // Large time
        for(int k=1; k<50; ++k) {
            double term = k * std::exp(-0.5 * k * k * M_PI * M_PI * tau) * std::sin(k * M_PI * w);
            p += term;
            if (std::abs(term) < 1e-10) break;
        }
        p *= M_PI;
    } else { // Small time
        for(int k=-10; k<=10; ++k) {
            double term = (w + 2.0*k) * std::exp(-0.5 * (w + 2.0*k) * (w + 2.0*k) / tau);
            p += term;
        }
        p /= std::sqrt(2.0 * M_PI * tau * tau * tau);
    }
    double res = (p / (a * a)) * std::exp(-v * a * w - 0.5 * v * v * tt);
    return res > 1e-10 ? res : 1e-10;
}

// ==========================================
// BASELINE WALD (WFPT CORRECTED)
// ==========================================
// [[Rcpp::export]]
double get_nll_base_nf(const std::vector<double>& phi, const IntegerVector& resp, const NumericVector& reward, const NumericVector& rt) {
    double a = std::exp(phi[0]);
    double tnd = 1.0 / (1.0 + std::exp(-phi[1]));
    double k_v = std::exp(phi[2]);
    double alpha = 1.0 / (1.0 + std::exp(-phi[3]));
    
    double nll = 0.0;
    double Q[2] = {0.5, 0.5};
    
    for(int t=0; t<resp.size(); ++t) {
        int ch = resp[t] - 1; 
        double R = (reward[t] == 1.0) ? 1.0 : 0.0;
        
        double v = k_v * (Q[1] - Q[0]);
        double safe_v = std::abs(v) < 1e-4 ? (v >= 0 ? 1e-4 : -1e-4) : v;
        
        double pdf = wiener_pdf_nf(rt[t], resp[t], safe_v, a, tnd, 0.5);
        nll -= std::log(pdf);
        
        Q[ch] += alpha * (R - std::max(Q[ch], 0.01));
    }
    return nll;
}

// [[Rcpp::export]]
NumericMatrix ext_base_nf(const std::vector<double>& phi, const IntegerVector& resp, const NumericVector& reward, const NumericVector& rt) {
    double a = std::exp(phi[0]);
    double tnd = 1.0 / (1.0 + std::exp(-phi[1]));
    double k_v = std::exp(phi[2]);
    double alpha = 1.0 / (1.0 + std::exp(-phi[3]));
    
    NumericMatrix res(resp.size(), 2);
    double Q[2] = {0.5, 0.5};
    
    for(int t=0; t<resp.size(); ++t) {
        int ch = resp[t] - 1; 
        double R = (reward[t] == 1.0) ? 1.0 : 0.0;
        
        double v = k_v * (Q[1] - Q[0]);
        double safe_v = std::abs(v) < 1e-4 ? (v >= 0 ? 1e-4 : -1e-4) : v;
        
        double P_upper = 1.0 / (1.0 + std::exp(-safe_v * a));
        double E_rt = tnd;
        if (std::abs(safe_v) < 1e-4) E_rt += (a * a) / 4.0;
        else E_rt += (a / (2.0 * safe_v)) * std::tanh(safe_v * a / 2.0);
        
        res(t, 0) = E_rt;
        res(t, 1) = P_upper;
        
        Q[ch] += alpha * (R - std::max(Q[ch], 0.01));
    }
    return res;
}


// ==========================================
// THERMODYNAMIC SUDOKU (WFPT CORRECTED)
// ==========================================
// [[Rcpp::export]]
double get_nll_thermo_sudoku_nf(const std::vector<double>& phi, const std::vector<double>& hyper, const IntegerVector& resp, const NumericVector& reward, const NumericVector& rt) {
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
    
    double nll = 0.0;
    double prev_E_IO = 0.0;
    double Q_ctx[2] = {0.5, 0.5};
    
    for(int t=0; t<resp.size(); ++t) {
        int ch = resp[t] - 1; double R = (reward[t] == 1.0) ? 1.0 : 0.0;
        
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
        
        double pdf = wiener_pdf_nf(rt[t], resp[t], safe_v, a, tnd, 0.5);
        nll -= std::log(pdf); // EXACT JOINT DENSITY, NO SOFTMAX EXPLOIT
        
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
                if(dF < 0 || unif(gen) < std::exp(-dF / temp)) current_F = new_F; 
                else S[idx] = 1 - S[idx]; 
            }
        }
    }
    return nll;
}

// [[Rcpp::export]]
NumericMatrix ext_thermo_sudoku_nf(const std::vector<double>& phi, const std::vector<double>& hyper, const IntegerVector& resp, const NumericVector& reward, const NumericVector& rt) {
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
    
    NumericMatrix res(resp.size(), 2);
    double prev_E_IO = 0.0;
    double Q_ctx[2] = {0.5, 0.5};
    
    for(int t=0; t<resp.size(); ++t) {
        int ch = resp[t] - 1; double R = (reward[t] == 1.0) ? 1.0 : 0.0;
        
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
        
        double P_upper = 1.0 / (1.0 + std::exp(-safe_v * a));
        double E_rt = tnd;
        if (std::abs(safe_v) < 1e-4) E_rt += (a * a) / 4.0;
        else E_rt += (a / (2.0 * safe_v)) * std::tanh(safe_v * a / 2.0);
        
        res(t, 0) = E_rt;
        res(t, 1) = P_upper;
        
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
