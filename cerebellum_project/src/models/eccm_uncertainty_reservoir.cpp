#include <Rcpp.h>
#include "shared_utils.h"

using namespace Rcpp;

// [[Rcpp::export]]
inline double eval_eccm_uncertainty_reservoir(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt, const NumericVector& delta_t) {
    double a = std::exp(phi[0]), t_nd = 1.0 / (1.0 + std::exp(-phi[1])), beta_v = std::exp(phi[2]);
    double eta_LTP = 1.0 / (1.0 + std::exp(-phi[3])), eta_LTD = 1.0 / (1.0 + std::exp(-phi[4])), w_cb = phi[5]; 
    double lambda_shift = std::exp(phi[6]); 
    double tau = 1.0 / (1.0 + std::exp(-phi[7])); // Time constant for C_RPE
    double alpha_out = 1.0 / (1.0 + std::exp(-phi[8])); // Learning rate for Readout
    
    int k = 80; int N_MF = 240, N_GC = 1024, N_MLI = 256; 
    
    std::vector<double> W_PF1(N_GC, 0.0), W_PF2(N_GC, 0.0);
    std::vector<double> W_MLI1(N_MLI, 0.0), W_MLI2(N_MLI, 0.0);
    std::vector<std::vector<double>> W_MF_GC(N_GC, std::vector<double>(N_MF, 0.0));
    std::vector<std::vector<double>> W_GC_MLI(N_MLI, std::vector<double>(N_GC, 0.0));
    std::vector<double> mf(N_MF, 0.0);
    
    // Uncertainty Reservoir (LC-NE Hub)
    int N_RES = 30;
    std::vector<std::vector<double>> W_res(N_RES, std::vector<double>(2, 0.0));
    std::vector<double> W_out(N_RES, 0.0);
    
    SimpleRNG rng(42);
    for (int i=0; i<N_GC; ++i) for (int j=0; j<N_MF; ++j) W_MF_GC[i][j] = rng.rnorm() / std::sqrt((double)N_MF);
    for (int i=0; i<N_MLI; ++i) for (int j=0; j<N_GC; ++j) W_GC_MLI[i][j] = rng.rnorm() / std::sqrt((double)N_GC);
    for (int i=0; i<N_RES; ++i) {
        W_res[i][0] = rng.rnorm(); // delta_cc weight
        W_res[i][1] = rng.rnorm(); // C_RPE weight
    }
    
    std::vector<double> D_vec;
    double Q1_CTX = 0.0, Q2_CTX = 0.0;
    double C_RPE = 0.0;
    
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
        
        double delta_cc = (ch == 1) ? std::abs(Q1_CTX - Q1_CB) : std::abs(Q2_CTX - Q2_CB);
        double v_t = beta_v * ( (Q1_CTX * (1.0 + w_cb * Q1_CB)) - (Q2_CTX * (1.0 + w_cb * Q2_CB)) );
        
        // UNCERTAINTY RESERVOIR
        std::vector<double> h(N_RES, 0.0);
        double U_t = 0.0;
        for (int i=0; i<N_RES; ++i) {
            h[i] = std::tanh(W_res[i][0] * delta_cc + W_res[i][1] * C_RPE);
            U_t += W_out[i] * h[i];
        }
        double epsilon = 1.0 / (1.0 + std::exp(-U_t)); // [0, 1]
        
        // Epsilon Modulation: if 0, v_effective = v_t; if 0.5, 0; if 1.0, -v_t
        double v_effective = v_t * (1.0 - 2.0 * epsilon);
        
        double safe_v = std::abs(v_effective) < 1e-4 ? (v_effective >= 0 ? 1e-4 : -1e-4) : v_effective;
        D_vec.push_back(-2.0 * std::log(wiener_pdf(rt[t], ch, safe_v, a, t_nd)));
        
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
        
        // Update C_RPE
        C_RPE = (1.0 - tau) * C_RPE + tau * std::abs(err_ctx);
        
        // Update W_out (Readout Layer)
        // If they switched, switch_flag = 1.0. If stayed, switch_flag = -1.0.
        double switch_flag = (t == 0 || ch == resp[t-1]) ? -1.0 : 1.0;
        for (int i=0; i<N_RES; ++i) {
            W_out[i] += alpha_out * err_ctx * switch_flag * h[i];
        }
        
        double dt = delta_t[t];
        int shifts = 1;
        if (dt > 2.0 && !std::isnan(dt)) {
            shifts = 1 + std::floor(lambda_shift * (dt - 2.0));
        }
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
    return calc_pen_ll(D_vec);
}
