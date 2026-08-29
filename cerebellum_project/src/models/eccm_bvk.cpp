#include <Rcpp.h>
#include "shared_utils.h"

using namespace Rcpp;

// [[Rcpp::export]]
inline double eval_eccm_bvk(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt, const NumericVector& delta_t) {
    double a = std::exp(phi[0]), t_nd = 1.0 / (1.0 + std::exp(-phi[1])), beta_v = std::exp(phi[2]);
    double eta_LTP = 1.0 / (1.0 + std::exp(-phi[3])), eta_LTD = 1.0 / (1.0 + std::exp(-phi[4]));
    double lambda_shift = std::exp(phi[6]); 
    double theta_cb = 1.0 / (1.0 + std::exp(-phi[8]));
    double kappa = std::exp(phi[9]);
    
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
    double Q1_CTX = 0.0, Q2_CTX = 0.0;
    
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
        
        double delta_CC = std::abs((Q1_CTX - Q2_CTX) - (Q1_CB - Q2_CB));
        
        double w_raw = 0.5 + theta_cb * (Q1_CB - Q2_CB);
        double w_t = std::max(0.05, std::min(0.95, w_raw));
        
        double v_base = beta_v * (Q1_CTX - Q2_CTX);
        double v_t = v_base * std::exp(-kappa * delta_CC);
        
        double safe_v = std::abs(v_t) < 1e-4 ? (v_t >= 0 ? 1e-4 : -1e-4) : v_t;
        
        double a_t = a;
        if (a_t > 10.0) a_t = 10.0;
        if (a_t < 0.01) a_t = 0.01;
        
        D_vec.push_back(-2.0 * std::log(wiener_pdf_w(rt[t], ch, safe_v, a_t, t_nd, w_t)));
        
        double lr = (out[t] == 1) ? eta_LTP : eta_LTD;
        double lr_cb = lr / (double)N_GC, lr_mli = lr / (double)N_MLI;
        double err_cb = 0.0;
        double err_ctx = 0.0;
        
        if (ch == 1) {
            err_cb = R_raw - Q1_CB;
            err_ctx = R_raw - Q1_CTX;
            Q1_CTX += (out[t] == 1) ? eta_LTP*(1.0-Q1_CTX) : eta_LTD*(-1.0-Q1_CTX);
            for (int i=0; i<N_GC; ++i) W_PF1[i] += lr_cb * err_cb * gc[i];
            for (int i=0; i<N_MLI; ++i) W_MLI1[i] -= lr_mli * err_cb * mli[i];
        } else {
            err_cb = R_raw - Q2_CB;
            err_ctx = R_raw - Q2_CTX;
            Q2_CTX += (out[t] == 1) ? eta_LTP*(1.0-Q2_CTX) : eta_LTD*(-1.0-Q2_CTX);
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
NumericVector extract_ll_eccm_bvk(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt, const NumericVector& delta_t) {
    double a = std::exp(phi[0]), t_nd = 1.0 / (1.0 + std::exp(-phi[1])), beta_v = std::exp(phi[2]);
    double eta_LTP = 1.0 / (1.0 + std::exp(-phi[3])), eta_LTD = 1.0 / (1.0 + std::exp(-phi[4]));
    double lambda_shift = std::exp(phi[6]); 
    double theta_cb = 1.0 / (1.0 + std::exp(-phi[8]));
    double kappa = std::exp(phi[9]);
    
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
    double Q1_CTX = 0.0, Q2_CTX = 0.0;
    
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
        
        double delta_CC = std::abs((Q1_CTX - Q2_CTX) - (Q1_CB - Q2_CB));
        
        double w_raw = 0.5 + theta_cb * (Q1_CB - Q2_CB);
        double w_t = std::max(0.05, std::min(0.95, w_raw));
        
        double v_base = beta_v * (Q1_CTX - Q2_CTX);
        double v_t = v_base * std::exp(-kappa * delta_CC);
        
        double safe_v = std::abs(v_t) < 1e-4 ? (v_t >= 0 ? 1e-4 : -1e-4) : v_t;
        
        double a_t = a;
        if (a_t > 10.0) a_t = 10.0;
        if (a_t < 0.01) a_t = 0.01;
        
        ll[t] = std::log(wiener_pdf_w(rt[t], ch, safe_v, a_t, t_nd, w_t));
        
        double lr = (out[t] == 1) ? eta_LTP : eta_LTD;
        double lr_cb = lr / (double)N_GC, lr_mli = lr / (double)N_MLI;
        double err_cb = 0.0;
        double err_ctx = 0.0;
        
        if (ch == 1) {
            err_cb = R_raw - Q1_CB;
            err_ctx = R_raw - Q1_CTX;
            Q1_CTX += (out[t] == 1) ? eta_LTP*(1.0-Q1_CTX) : eta_LTD*(-1.0-Q1_CTX);
            for (int i=0; i<N_GC; ++i) W_PF1[i] += lr_cb * err_cb * gc[i];
            for (int i=0; i<N_MLI; ++i) W_MLI1[i] -= lr_mli * err_cb * mli[i];
        } else {
            err_cb = R_raw - Q2_CB;
            err_ctx = R_raw - Q2_CTX;
            Q2_CTX += (out[t] == 1) ? eta_LTP*(1.0-Q2_CTX) : eta_LTD*(-1.0-Q2_CTX);
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
