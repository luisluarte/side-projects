#include <Rcpp.h>
#include "shared_utils.h"

using namespace Rcpp;

// Full Gating Model
// phi[0] = a
// phi[1] = t_nd
// phi[2] = beta_v
// phi[3] = eta_LTP
// phi[4] = eta_LTD
// phi[5] = lambda_shift
// phi[6] = gamma_v
// phi[7] = beta_a
// phi[8] = theta_cb

// [[Rcpp::export]]
inline double eval_eccm_full_gating(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt, const NumericVector& delta_t) {
    double a = std::exp(phi[0]);
    double t_nd = 1.0 / (1.0 + std::exp(-phi[1]));
    double beta_v = std::exp(phi[2]);
    double eta_LTP = 1.0 / (1.0 + std::exp(-phi[3]));
    double eta_LTD = 1.0 / (1.0 + std::exp(-phi[4]));
    double lambda_shift = std::exp(phi[5]); 
    double gamma_v = std::exp(phi[6]); 
    double beta_a = std::exp(phi[7]);
    double theta_cb = std::exp(phi[8]);
    
    int k = 80; int N_MF = 240, N_GC = 1024, N_MLI = 256; 
    
    std::vector<double> W_PF1(N_GC, 0.0), W_PF2(N_GC, 0.0);
    std::vector<double> W_MLI1(N_MLI, 0.0), W_MLI2(N_MLI, 0.0);
    std::vector<std::vector<double>> W_MF_GC(N_GC, std::vector<double>(N_MF, 0.0));
    std::vector<std::vector<double>> W_GC_MLI(N_MLI, std::vector<double>(N_GC, 0.0));
    std::vector<double> mf(N_MF, 0.0);
    
    SimpleRNG rng(42);
    for (int i=0; i<N_GC; ++i) for (int j=0; j<N_MF; ++j) W_MF_GC[i][j] = rng.rnorm() / std::sqrt((double)N_MF);
    for (int i=0; i<N_MLI; ++i) for (int j=0; j<N_GC; ++j) W_GC_MLI[i][j] = rng.rnorm() / std::sqrt((double)N_GC);
    
    std::vector<double> D_vec;
    double Q1_CTX = 0.5, Q2_CTX = 0.5;
    
    for (int t = 0; t < resp.size(); ++t) {
        int ch = resp[t]; double R_raw = (out[t] == 1) ? 1.0 : -1.0;
        double Q1_CB = 0.0, Q2_CB = 0.0;
        
        std::vector<double> gc(N_GC, 0.0), mli(N_MLI, 0.0);
        for (int i=0; i<N_GC; ++i) {
            double act = 0.0; for (int j=0; j<N_MF; ++j) act += W_MF_GC[i][j] * mf[j];
            gc[i] = std::tanh(act);
        }
        for (int i=0; i<N_MLI; ++i) {
            double act = 0.0; for (int j=0; j<N_GC; ++j) act += W_GC_MLI[i][j] * gc[j];
            mli[i] = std::tanh(act);
        }
        for (int i=0; i<N_GC; ++i) { Q1_CB += W_PF1[i]*gc[i]; Q2_CB += W_PF2[i]*gc[i]; }
        for (int i=0; i<N_MLI; ++i) { Q1_CB -= W_MLI1[i]*mli[i]; Q2_CB -= W_MLI2[i]*mli[i]; }
        
        double d_ctx = Q1_CTX - Q2_CTX;
        double d_cb = Q1_CB - Q2_CB;
        
        double conflict = 0.5 * (1.0 - std::tanh(10.0 * d_ctx) * std::tanh(10.0 * d_cb));
        
        double w_bias = 0.5 + 0.45 * std::tanh(theta_cb * d_cb);
        if (w_bias < 0.05) w_bias = 0.05;
        if (w_bias > 0.95) w_bias = 0.95;
        
        double v_base = beta_v * d_ctx;
        double v_effective = v_base * std::exp(-gamma_v * conflict);
        
        double a_effective = a + beta_a * conflict;
        if (a_effective < 0.01) a_effective = 0.01;
        if (a_effective > 10.0) a_effective = 10.0;
        
        double safe_v = std::abs(v_effective) < 1e-4 ? (v_effective >= 0 ? 1e-4 : -1e-4) : v_effective;
        
        D_vec.push_back(-2.0 * std::log(wiener_pdf_w(rt[t], ch, safe_v, a_effective, t_nd, w_bias)));
        
        double lr = (out[t] == 1) ? eta_LTP : eta_LTD;
        double lr_cb = lr / (double)N_GC, lr_mli = lr / (double)N_MLI;
        double err_cb = 0.0;
        double err_ctx = 0.0;
        
        if (ch == 1) {
            err_cb = R_raw - Q1_CB;
            err_ctx = (out[t] == 1 ? 1.0 : 0.0) - Q1_CTX;
            Q1_CTX += (out[t] == 1) ? eta_LTP*(1.0-Q1_CTX) : eta_LTD*(0.0-Q1_CTX);
            for (int i=0; i<N_GC; ++i) W_PF1[i] += lr_cb * err_cb * gc[i];
            for (int i=0; i<N_MLI; ++i) W_MLI1[i] -= lr_mli * err_cb * mli[i];
        } else {
            err_cb = R_raw - Q2_CB;
            err_ctx = (out[t] == 1 ? 1.0 : 0.0) - Q2_CTX;
            Q2_CTX += (out[t] == 1) ? eta_LTP*(1.0-Q2_CTX) : eta_LTD*(0.0-Q2_CTX);
            for (int i=0; i<N_GC; ++i) W_PF2[i] += lr_cb * err_cb * gc[i];
            for (int i=0; i<N_MLI; ++i) W_MLI2[i] -= lr_mli * err_cb * mli[i];
        }
        
        double dt = delta_t[t];
        int shifts = 1;
        if (dt > 2.0 && !std::isnan(dt)) shifts = 1 + std::floor(lambda_shift * (dt - 2.0));
        shifts = std::min(shifts, k);

        for (int s = 0; s < shifts; ++s) {
            for(int j=k-2; j>=0; --j) { 
                mf[j+1] = mf[j]; 
                mf[k+j+1] = mf[k+j]; 
                mf[2*k+j+1] = mf[2*k+j];
            }
            if (s == 0) {
                mf[0] = (ch == 1) ? 1.0 : -1.0; 
                mf[k] = R_raw;
                mf[2*k] = err_ctx;
            } else {
                mf[0] = 0.0; 
                mf[k] = 0.0;
                mf[2*k] = 0.0;
            }
        }
    }
    double res = calc_pen_ll(D_vec);
    return (std::isnan(res) || std::isinf(res)) ? 1e9 : res;
}

// [[Rcpp::export]]
NumericVector extract_ll_eccm_full_gating(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt, const NumericVector& delta_t) {
    double a = std::exp(phi[0]);
    double t_nd = 1.0 / (1.0 + std::exp(-phi[1]));
    double beta_v = std::exp(phi[2]);
    double eta_LTP = 1.0 / (1.0 + std::exp(-phi[3]));
    double eta_LTD = 1.0 / (1.0 + std::exp(-phi[4]));
    double lambda_shift = std::exp(phi[5]); 
    double gamma_v = std::exp(phi[6]); 
    double beta_a = std::exp(phi[7]);
    double theta_cb = std::exp(phi[8]);
    
    int k = 80; int N_MF = 240, N_GC = 1024, N_MLI = 256; 
    
    std::vector<double> W_PF1(N_GC, 0.0), W_PF2(N_GC, 0.0);
    std::vector<double> W_MLI1(N_MLI, 0.0), W_MLI2(N_MLI, 0.0);
    std::vector<std::vector<double>> W_MF_GC(N_GC, std::vector<double>(N_MF, 0.0));
    std::vector<std::vector<double>> W_GC_MLI(N_MLI, std::vector<double>(N_GC, 0.0));
    std::vector<double> mf(N_MF, 0.0);
    
    SimpleRNG rng(42);
    for (int i=0; i<N_GC; ++i) for (int j=0; j<N_MF; ++j) W_MF_GC[i][j] = rng.rnorm() / std::sqrt((double)N_MF);
    for (int i=0; i<N_MLI; ++i) for (int j=0; j<N_GC; ++j) W_GC_MLI[i][j] = rng.rnorm() / std::sqrt((double)N_GC);
    
    int T = resp.size();
    NumericVector ll(T);
    double Q1_CTX = 0.5, Q2_CTX = 0.5;
    
    for (int t = 0; t < T; ++t) {
        int ch = resp[t]; double R_raw = (out[t] == 1) ? 1.0 : -1.0;
        double Q1_CB = 0.0, Q2_CB = 0.0;
        
        std::vector<double> gc(N_GC, 0.0), mli(N_MLI, 0.0);
        for (int i=0; i<N_GC; ++i) {
            double act = 0.0; for (int j=0; j<N_MF; ++j) act += W_MF_GC[i][j] * mf[j];
            gc[i] = std::tanh(act);
        }
        for (int i=0; i<N_MLI; ++i) {
            double act = 0.0; for (int j=0; j<N_GC; ++j) act += W_GC_MLI[i][j] * gc[j];
            mli[i] = std::tanh(act);
        }
        for (int i=0; i<N_GC; ++i) { Q1_CB += W_PF1[i]*gc[i]; Q2_CB += W_PF2[i]*gc[i]; }
        for (int i=0; i<N_MLI; ++i) { Q1_CB -= W_MLI1[i]*mli[i]; Q2_CB -= W_MLI2[i]*mli[i]; }
        
        double d_ctx = Q1_CTX - Q2_CTX;
        double d_cb = Q1_CB - Q2_CB;
        
        double conflict = 0.5 * (1.0 - std::tanh(10.0 * d_ctx) * std::tanh(10.0 * d_cb));
        
        double w_bias = 0.5 + 0.45 * std::tanh(theta_cb * d_cb);
        if (w_bias < 0.05) w_bias = 0.05;
        if (w_bias > 0.95) w_bias = 0.95;
        
        double v_base = beta_v * d_ctx;
        double v_effective = v_base * std::exp(-gamma_v * conflict);
        
        double a_effective = a + beta_a * conflict;
        if (a_effective < 0.01) a_effective = 0.01;
        if (a_effective > 10.0) a_effective = 10.0;
        
        double safe_v = std::abs(v_effective) < 1e-4 ? (v_effective >= 0 ? 1e-4 : -1e-4) : v_effective;
        
        ll[t] = std::log(wiener_pdf_w(rt[t], ch, safe_v, a_effective, t_nd, w_bias));
        
        double lr = (out[t] == 1) ? eta_LTP : eta_LTD;
        double lr_cb = lr / (double)N_GC, lr_mli = lr / (double)N_MLI;
        double err_cb = 0.0;
        double err_ctx = 0.0;
        
        if (ch == 1) {
            err_cb = R_raw - Q1_CB;
            err_ctx = (out[t] == 1 ? 1.0 : 0.0) - Q1_CTX;
            Q1_CTX += (out[t] == 1) ? eta_LTP*(1.0-Q1_CTX) : eta_LTD*(0.0-Q1_CTX);
            for (int i=0; i<N_GC; ++i) W_PF1[i] += lr_cb * err_cb * gc[i];
            for (int i=0; i<N_MLI; ++i) W_MLI1[i] -= lr_mli * err_cb * mli[i];
        } else {
            err_cb = R_raw - Q2_CB;
            err_ctx = (out[t] == 1 ? 1.0 : 0.0) - Q2_CTX;
            Q2_CTX += (out[t] == 1) ? eta_LTP*(1.0-Q2_CTX) : eta_LTD*(0.0-Q2_CTX);
            for (int i=0; i<N_GC; ++i) W_PF2[i] += lr_cb * err_cb * gc[i];
            for (int i=0; i<N_MLI; ++i) W_MLI2[i] -= lr_mli * err_cb * mli[i];
        }
        
        double dt = delta_t[t];
        int shifts = 1;
        if (dt > 2.0 && !std::isnan(dt)) shifts = 1 + std::floor(lambda_shift * (dt - 2.0));
        shifts = std::min(shifts, k);

        for (int s = 0; s < shifts; ++s) {
            for(int j=k-2; j>=0; --j) { 
                mf[j+1] = mf[j]; 
                mf[k+j+1] = mf[k+j]; 
                mf[2*k+j+1] = mf[2*k+j];
            }
            if (s == 0) {
                mf[0] = (ch == 1) ? 1.0 : -1.0; 
                mf[k] = R_raw;
                mf[2*k] = err_ctx;
            } else {
                mf[0] = 0.0; 
                mf[k] = 0.0;
                mf[2*k] = 0.0;
            }
        }
    }
    return ll;
}
