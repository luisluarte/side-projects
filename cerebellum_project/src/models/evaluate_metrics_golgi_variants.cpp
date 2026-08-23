#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace Rcpp;

// 1. GOLGI CEILING RELU
// [[Rcpp::export]]
List eval_metrics_eccm_golgi_ceiling(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt, const NumericVector& delta_t) {
    double a = std::exp(phi[0]), t_nd = 1.0 / (1.0 + std::exp(-phi[1])), beta_v = std::exp(phi[2]);
    double eta_LTP = 1.0 / (1.0 + std::exp(-phi[3])), eta_LTD = 1.0 / (1.0 + std::exp(-phi[4])), w_cb = phi[5]; 
    double lambda_shift = std::exp(phi[6]); 
    double gamma_suppress = std::exp(phi[7]); 
    double theta_ceiling = std::exp(phi[8]); 
    
    int k = 80; int N_MF = 240, N_GC = 1024, N_MLI = 256;
    std::vector<double> W_PF1(N_GC, 0.0), W_PF2(N_GC, 0.0), W_MLI1(N_MLI, 0.0), W_MLI2(N_MLI, 0.0), mf(N_MF, 0.0);
    std::vector<std::vector<double>> W_MF_GC(N_GC, std::vector<double>(N_MF, 0.0)), W_GC_MLI(N_MLI, std::vector<double>(N_GC, 0.0));
    
    uint32_t state = 42;
    auto next_rnd = [&]() { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state; };
    auto runif = [&]() { return (next_rnd() % 1000000) / 1000000.0; };
    auto rnorm = [&]() { double u1 = std::max(1e-6, runif()); double u2 = runif(); return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2); };

    for (int i=0; i<N_GC; ++i) for (int j=0; j<N_MF; ++j) W_MF_GC[i][j] = rnorm() / std::sqrt((double)N_MF);
    for (int i=0; i<N_MLI; ++i) for (int j=0; j<N_GC; ++j) W_GC_MLI[i][j] = rnorm() / std::sqrt((double)N_GC);
    
    std::vector<double> prob_ch1, exp_rt;
    double Q1_CTX = 0.0, Q2_CTX = 0.0;
    
    for (int t = 0; t < resp.size(); ++t) {
        int ch = resp[t]; double R_raw = (out[t] == 1) ? 1.0 : -1.0;
        double Q1_CB = 0.0, Q2_CB = 0.0;
        
        double mf_energy = 0.0;
        for (int j=0; j<N_MF; ++j) mf_energy += mf[j] * mf[j];
        double ceiling = std::exp(-theta_ceiling * std::sqrt(mf_energy));
        
        std::vector<double> gc(N_GC, 0.0), mli(N_MLI, 0.0);
        for (int i=0; i<N_GC; ++i) {
            double act = 0.0; for (int j=0; j<N_MF; ++j) act += W_MF_GC[i][j] * mf[j];
            gc[i] = std::min(ceiling, std::max(0.0, act));
        }
        for (int i=0; i<N_MLI; ++i) {
            double act = 0.0; for (int j=0; j<N_GC; ++j) act += W_GC_MLI[i][j] * gc[j];
            mli[i] = std::tanh(act);
        }
        for (int i=0; i<N_GC; ++i) { Q1_CB += W_PF1[i]*gc[i]; Q2_CB += W_PF2[i]*gc[i]; }
        for (int i=0; i<N_MLI; ++i) { Q1_CB -= W_MLI1[i]*mli[i]; Q2_CB -= W_MLI2[i]*mli[i]; }
        
        double delta_cc = (ch == 1) ? std::abs(Q1_CTX - Q1_CB) : std::abs(Q2_CTX - Q2_CB);
        double v_t = beta_v * ( (Q1_CTX * (1.0 + w_cb * Q1_CB)) - (Q2_CTX * (1.0 + w_cb * Q2_CB)) );
        double v_effective = v_t * std::exp(-gamma_suppress * delta_cc);
        
        prob_ch1.push_back(1.0 / (1.0 + std::exp(-v_effective * a)));
        
        double lr = (out[t] == 1) ? eta_LTP : eta_LTD;
        double lr_cb = lr / (double)N_GC, lr_mli = lr / (double)N_MLI;
        double err_cb = (ch == 1) ? (R_raw - Q1_CB) : (R_raw - Q2_CB);
        double err_ctx = (ch == 1) ? (R_raw - Q1_CTX) : (R_raw - Q2_CTX);
        
        if (ch == 1) {
            Q1_CTX += (out[t] == 1) ? eta_LTP*(1.0-Q1_CTX) : eta_LTD*(-1.0-Q1_CTX);
            for (int i=0; i<N_GC; ++i) W_PF1[i] += lr_cb * err_cb * gc[i];
            for (int i=0; i<N_MLI; ++i) W_MLI1[i] -= lr_mli * err_cb * mli[i];
        } else {
            Q2_CTX += (out[t] == 1) ? eta_LTP*(1.0-Q2_CTX) : eta_LTD*(-1.0-Q2_CTX);
            for (int i=0; i<N_GC; ++i) W_PF2[i] += lr_cb * err_cb * gc[i];
            for (int i=0; i<N_MLI; ++i) W_MLI2[i] -= lr_mli * err_cb * mli[i];
        }
        
        int shifts = (delta_t[t] > 2.0 && !std::isnan(delta_t[t])) ? std::min(k, 1 + (int)std::floor(lambda_shift * (delta_t[t] - 2.0))) : 1;
        for (int s = 0; s < shifts; ++s) {
            for(int j=k-2; j>=0; --j) { mf[j+1] = mf[j]; mf[k+j+1] = mf[k+j]; mf[2*k+j+1] = mf[2*k+j]; }
            if (s == 0) { mf[0] = (ch == 1) ? 1.0 : -1.0; mf[k] = R_raw; mf[2*k] = err_ctx; } 
            else { mf[0] = 0.0; mf[k] = 0.0; mf[2*k] = 0.0; }
        }
    }
    return List::create(Named("prob_ch1") = prob_ch1);
}

// 2. GOLGI TEMPERATURE SOFTMAX
// [[Rcpp::export]]
List eval_metrics_eccm_golgi_softmax(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt, const NumericVector& delta_t) {
    double a = std::exp(phi[0]), t_nd = 1.0 / (1.0 + std::exp(-phi[1])), beta_v = std::exp(phi[2]);
    double eta_LTP = 1.0 / (1.0 + std::exp(-phi[3])), eta_LTD = 1.0 / (1.0 + std::exp(-phi[4])), w_cb = phi[5]; 
    double lambda_shift = std::exp(phi[6]); 
    double gamma_suppress = std::exp(phi[7]); 
    double theta_temp = std::exp(phi[8]); 
    
    int k = 80; int N_MF = 240, N_GC = 1024, N_MLI = 256;
    std::vector<double> W_PF1(N_GC, 0.0), W_PF2(N_GC, 0.0), W_MLI1(N_MLI, 0.0), W_MLI2(N_MLI, 0.0), mf(N_MF, 0.0);
    std::vector<std::vector<double>> W_MF_GC(N_GC, std::vector<double>(N_MF, 0.0)), W_GC_MLI(N_MLI, std::vector<double>(N_GC, 0.0));
    
    uint32_t state = 42;
    auto next_rnd = [&]() { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state; };
    auto runif = [&]() { return (next_rnd() % 1000000) / 1000000.0; };
    auto rnorm = [&]() { double u1 = std::max(1e-6, runif()); double u2 = runif(); return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2); };

    for (int i=0; i<N_GC; ++i) for (int j=0; j<N_MF; ++j) W_MF_GC[i][j] = rnorm() / std::sqrt((double)N_MF);
    for (int i=0; i<N_MLI; ++i) for (int j=0; j<N_GC; ++j) W_GC_MLI[i][j] = rnorm() / std::sqrt((double)N_GC);
    
    std::vector<double> prob_ch1, exp_rt;
    double Q1_CTX = 0.0, Q2_CTX = 0.0;
    
    for (int t = 0; t < resp.size(); ++t) {
        int ch = resp[t]; double R_raw = (out[t] == 1) ? 1.0 : -1.0;
        double Q1_CB = 0.0, Q2_CB = 0.0;
        
        double mf_energy = 0.0;
        for (int j=0; j<N_MF; ++j) mf_energy += mf[j] * mf[j];
        double tau = 1.0 + theta_temp * std::sqrt(mf_energy);
        
        std::vector<double> gc(N_GC, 0.0), mli(N_MLI, 0.0);
        double max_act = -1e9;
        std::vector<double> raw_acts(N_GC, 0.0);
        for (int i=0; i<N_GC; ++i) {
            double act = 0.0; for (int j=0; j<N_MF; ++j) act += W_MF_GC[i][j] * mf[j];
            raw_acts[i] = act / tau;
            if (raw_acts[i] > max_act) max_act = raw_acts[i];
        }
        
        double sum_exp = 0.0;
        for (int i=0; i<N_GC; ++i) {
            gc[i] = std::exp(raw_acts[i] - max_act);
            sum_exp += gc[i];
        }
        for (int i=0; i<N_GC; ++i) gc[i] /= sum_exp; 
        
        for (int i=0; i<N_MLI; ++i) {
            double act = 0.0; for (int j=0; j<N_GC; ++j) act += W_GC_MLI[i][j] * gc[j];
            mli[i] = std::tanh(act);
        }
        for (int i=0; i<N_GC; ++i) { Q1_CB += W_PF1[i]*gc[i]; Q2_CB += W_PF2[i]*gc[i]; }
        for (int i=0; i<N_MLI; ++i) { Q1_CB -= W_MLI1[i]*mli[i]; Q2_CB -= W_MLI2[i]*mli[i]; }
        
        double delta_cc = (ch == 1) ? std::abs(Q1_CTX - Q1_CB) : std::abs(Q2_CTX - Q2_CB);
        double v_t = beta_v * ( (Q1_CTX * (1.0 + w_cb * Q1_CB)) - (Q2_CTX * (1.0 + w_cb * Q2_CB)) );
        double v_effective = v_t * std::exp(-gamma_suppress * delta_cc);
        
        prob_ch1.push_back(1.0 / (1.0 + std::exp(-v_effective * a)));
        
        double lr = (out[t] == 1) ? eta_LTP : eta_LTD;
        double lr_cb = lr / (double)N_GC, lr_mli = lr / (double)N_MLI;
        double err_cb = (ch == 1) ? (R_raw - Q1_CB) : (R_raw - Q2_CB);
        double err_ctx = (ch == 1) ? (R_raw - Q1_CTX) : (R_raw - Q2_CTX);
        
        if (ch == 1) {
            Q1_CTX += (out[t] == 1) ? eta_LTP*(1.0-Q1_CTX) : eta_LTD*(-1.0-Q1_CTX);
            for (int i=0; i<N_GC; ++i) W_PF1[i] += lr_cb * err_cb * gc[i];
            for (int i=0; i<N_MLI; ++i) W_MLI1[i] -= lr_mli * err_cb * mli[i];
        } else {
            Q2_CTX += (out[t] == 1) ? eta_LTP*(1.0-Q2_CTX) : eta_LTD*(-1.0-Q2_CTX);
            for (int i=0; i<N_GC; ++i) W_PF2[i] += lr_cb * err_cb * gc[i];
            for (int i=0; i<N_MLI; ++i) W_MLI2[i] -= lr_mli * err_cb * mli[i];
        }
        
        int shifts = (delta_t[t] > 2.0 && !std::isnan(delta_t[t])) ? std::min(k, 1 + (int)std::floor(lambda_shift * (delta_t[t] - 2.0))) : 1;
        for (int s = 0; s < shifts; ++s) {
            for(int j=k-2; j>=0; --j) { mf[j+1] = mf[j]; mf[k+j+1] = mf[k+j]; mf[2*k+j+1] = mf[2*k+j]; }
            if (s == 0) { mf[0] = (ch == 1) ? 1.0 : -1.0; mf[k] = R_raw; mf[2*k] = err_ctx; } 
            else { mf[0] = 0.0; mf[k] = 0.0; mf[2*k] = 0.0; }
        }
    }
    return List::create(Named("prob_ch1") = prob_ch1);
}
