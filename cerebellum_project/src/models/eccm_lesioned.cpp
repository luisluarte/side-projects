#include <Rcpp.h>
#include "shared_utils.h"

using namespace Rcpp;

// [[Rcpp::export]]
double eval_eccm_lesioned(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    double a = std::exp(phi[0]), t_nd = 1.0 / (1.0 + std::exp(-phi[1])), beta_v = std::exp(phi[2]);
    double eta_LTP = 1.0 / (1.0 + std::exp(-phi[3])), eta_LTD = 1.0 / (1.0 + std::exp(-phi[4])), w_cb = phi[5]; 
    int k = 80; int N_MF = 160;
    
    std::vector<double> W_PF1(N_MF, 0.0), W_PF2(N_MF, 0.0);
    std::vector<double> mf(N_MF, 0.0);
    
    std::vector<double> D_vec;
    double Q1_CTX = 0.0, Q2_CTX = 0.0;
    
    for (int t = 0; t < resp.size(); ++t) {
        int ch = resp[t]; double R_raw = (out[t] == 1) ? 1.0 : -1.0;
        double Q1_CB = 0.0, Q2_CB = 0.0;
        
        for (int i=0; i<N_MF; ++i) { Q1_CB += W_PF1[i]*mf[i]; Q2_CB += W_PF2[i]*mf[i]; }
        Q1_CB = std::max(-1.0, std::min(1.0, Q1_CB));
        Q2_CB = std::max(-1.0, std::min(1.0, Q2_CB));
        
        double v_t = beta_v * ( (Q1_CTX * (1.0 + w_cb * Q1_CB)) - (Q2_CTX * (1.0 + w_cb * Q2_CB)) );
        double safe_v = std::abs(v_t) < 1e-4 ? (v_t >= 0 ? 1e-4 : -1e-4) : v_t;
        D_vec.push_back(-2.0 * std::log(wiener_pdf(rt[t], ch, safe_v, a, t_nd)));
        
        double lr = (out[t] == 1) ? eta_LTP : eta_LTD;
        double lr_cb = lr / (double)N_MF;
        
        if (ch == 1) {
            double err_cb = R_raw - Q1_CB;
            Q1_CTX += (out[t] == 1) ? eta_LTP*(1.0-Q1_CTX) : eta_LTD*(-1.0-Q1_CTX);
            for (int i=0; i<N_MF; ++i) W_PF1[i] += lr_cb * err_cb * mf[i];
        } else {
            double err_cb = R_raw - Q2_CB;
            Q2_CTX += (out[t] == 1) ? eta_LTP*(1.0-Q2_CTX) : eta_LTD*(-1.0-Q2_CTX);
            for (int i=0; i<N_MF; ++i) W_PF2[i] += lr_cb * err_cb * mf[i];
        }
        
        for(int j=k-2; j>=0; --j) { mf[j+1] = mf[j]; mf[k+j+1] = mf[k+j]; }
        mf[0] = (ch == 1) ? 1.0 : -1.0; mf[k] = R_raw;
    }
    return calc_pen_ll(D_vec);
}
