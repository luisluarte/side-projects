#include <Rcpp.h>
#include "shared_utils.h"
#include <cmath>

using namespace Rcpp;

// Model 20: Ablated MF-Gated Asymmetric Reversal
// Deletes Granule Cell (GC) and Molecular Layer Interneuron (MLI) layers.
// Direct linear mapping from Mossy Fibers (MF) to Purkinje Cells (PC).

// [[Rcpp::export]]
inline double eval_eccm_mf_rev_ablated(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt, const NumericVector& delta_t) {
    double a = std::exp(phi[0]), t_nd = 1.0 / (1.0 + std::exp(-phi[1])), beta_v = std::exp(phi[2]);
    double eta_LTP = 1.0 / (1.0 + std::exp(-phi[3])), eta_LTD = 1.0 / (1.0 + std::exp(-phi[4])), w_cb = phi[5]; 
    double lambda_shift = std::exp(phi[6]); 
    double gamma_suppress = std::exp(phi[7]); 
    // phi[8] was theta_golgi in Model 19. We omit it here.
    double mf_threshold = phi[8];                 
    double explore_gain_win = std::exp(phi[9]);  
    double explore_gain_loss = std::exp(phi[10]); 
    
    int k = 80; int N_MF = 240; 
    std::vector<double> W_MF_PC1(N_MF, 0.0), W_MF_PC2(N_MF, 0.0), mf(N_MF, 0.0);
    
    std::vector<double> D_vec;
    double Q1_CTX = 0.0, Q2_CTX = 0.0;
    double prev_mf_energy = 0.0;
    int prev_choice = 0;
    int prev_outcome = -1; // -1 = no previous, 0 = loss, 1 = win
    
    for (int t = 0; t < resp.size(); ++t) {
        int ch = resp[t]; double R_raw = (out[t] == 1) ? 1.0 : -1.0;
        double Q1_CB = 0.0, Q2_CB = 0.0;
        
        double mf_energy = 0.0;
        for (int j=0; j<N_MF; ++j) mf_energy += mf[j] * mf[j];
        double l2_mf = std::sqrt(mf_energy);
        
        for (int j=0; j<N_MF; ++j) {
            Q1_CB += W_MF_PC1[j] * mf[j];
            Q2_CB += W_MF_PC2[j] * mf[j];
        }
        
        double delta_cc = (ch == 1) ? std::abs(Q1_CTX - Q1_CB) : std::abs(Q2_CTX - Q2_CB);
        double v_t = beta_v * ( (Q1_CTX * (1.0 + w_cb * Q1_CB)) - (Q2_CTX * (1.0 + w_cb * Q2_CB)) );
        double v_effective = v_t * std::exp(-gamma_suppress * delta_cc);
        
        if (prev_choice != 0) {
            double delta_mf = l2_mf - prev_mf_energy;
            double gate = 1.0 / (1.0 + std::exp(-(delta_mf - mf_threshold)));
            double reversal_sign = (prev_choice == 1) ? -1.0 : 1.0;
            double current_gain = (prev_outcome == 1) ? explore_gain_win : explore_gain_loss;
            v_effective += reversal_sign * current_gain * gate;
        }
        
        prev_mf_energy = l2_mf;
        prev_choice = ch;
        prev_outcome = out[t];
        
        double safe_v = std::abs(v_effective) < 1e-4 ? (v_effective >= 0 ? 1e-4 : -1e-4) : v_effective;
        D_vec.push_back(-2.0 * std::log(wiener_pdf(rt[t], ch, safe_v, a, t_nd)));
        
        double lr = (out[t] == 1) ? eta_LTP : eta_LTD;
        double lr_cb = lr / (double)N_MF;
        double err_cb = (ch == 1) ? (R_raw - Q1_CB) : (R_raw - Q2_CB);
        double err_ctx = (ch == 1) ? (R_raw - Q1_CTX) : (R_raw - Q2_CTX);
        
        if (ch == 1) {
            Q1_CTX += (out[t] == 1) ? eta_LTP*(1.0-Q1_CTX) : eta_LTD*(-1.0-Q1_CTX);
            for (int j=0; j<N_MF; ++j) W_MF_PC1[j] += lr_cb * err_cb * mf[j];
        } else {
            Q2_CTX += (out[t] == 1) ? eta_LTP*(1.0-Q2_CTX) : eta_LTD*(-1.0-Q2_CTX);
            for (int j=0; j<N_MF; ++j) W_MF_PC2[j] += lr_cb * err_cb * mf[j];
        }
        
        int shifts = (delta_t[t] > 2.0 && !std::isnan(delta_t[t])) ? std::min(k, 1 + (int)std::floor(lambda_shift * (delta_t[t] - 2.0))) : 1;
        for (int s = 0; s < shifts; ++s) {
            for(int j=k-2; j>=0; --j) { mf[j+1] = mf[j]; mf[k+j+1] = mf[k+j]; mf[2*k+j+1] = mf[2*k+j]; }
            if (s == 0) { mf[0] = (ch == 1) ? 1.0 : -1.0; mf[k] = R_raw; mf[2*k] = err_ctx; } 
            else { mf[0] = 0.0; mf[k] = 0.0; mf[2*k] = 0.0; }
        }
    }
    return calc_pen_ll(D_vec);
}
