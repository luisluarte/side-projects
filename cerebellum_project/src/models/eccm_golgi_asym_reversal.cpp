#include <Rcpp.h>
#include "shared_utils.h"
#include <cmath>

using namespace Rcpp;

// Model 19: Golgi + MF-Gated Asymmetric Reversal
// 
// Same as Model 18 but splits explore_gain into two parameters:
//   - theta_explore_win:  reversal strength after a win trial (expected to be small)
//   - theta_explore_loss: reversal strength after a loss trial (expected to be large)
//
// Biologically: the LC phasic burst magnitude depends on the valence of the 
// preceding outcome. A loss increases noradrenergic drive → stronger exploration.
// A win should suppress exploration → stay with what works.

// [[Rcpp::export]]
inline double eval_eccm_golgi_asym_reversal(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt, const NumericVector& delta_t) {
    double a = std::exp(phi[0]), t_nd = 1.0 / (1.0 + std::exp(-phi[1])), beta_v = std::exp(phi[2]);
    double eta_LTP = 1.0 / (1.0 + std::exp(-phi[3])), eta_LTD = 1.0 / (1.0 + std::exp(-phi[4])), w_cb = phi[5]; 
    double lambda_shift = std::exp(phi[6]); 
    double gamma_suppress = std::exp(phi[7]); 
    double theta_golgi = std::exp(phi[8]);       // Golgi divisive gain
    double mf_threshold = phi[9];                 // MF energy threshold (unconstrained)
    double explore_gain_win = std::exp(phi[10]);  // Reversal strength after WIN
    double explore_gain_loss = std::exp(phi[11]); // Reversal strength after LOSS
    
    int k = 80; int N_MF = 240, N_GC = 1024, N_MLI = 256; 
    std::vector<double> W_PF1(N_GC, 0.0), W_PF2(N_GC, 0.0), W_MLI1(N_MLI, 0.0), W_MLI2(N_MLI, 0.0), mf(N_MF, 0.0);
    std::vector<std::vector<double>> W_MF_GC(N_GC, std::vector<double>(N_MF, 0.0)), W_GC_MLI(N_MLI, std::vector<double>(N_GC, 0.0));
    
    SimpleRNG rng(42);
    for (int i=0; i<N_GC; ++i) for (int j=0; j<N_MF; ++j) W_MF_GC[i][j] = rng.rnorm() / std::sqrt((double)N_MF);
    for (int i=0; i<N_MLI; ++i) for (int j=0; j<N_GC; ++j) W_GC_MLI[i][j] = rng.rnorm() / std::sqrt((double)N_GC);
    
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
        double gain = 1.0 / (1.0 + theta_golgi * l2_mf);
        
        std::vector<double> gc(N_GC, 0.0), mli(N_MLI, 0.0);
        for (int i=0; i<N_GC; ++i) {
            double act = 0.0; for (int j=0; j<N_MF; ++j) act += W_MF_GC[i][j] * mf[j];
            gc[i] = std::tanh(act * gain);
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
        
        // ASYMMETRIC MF-GATED CHOICE REVERSAL
        if (prev_choice != 0) {
            double delta_mf = l2_mf - prev_mf_energy;
            double gate = 1.0 / (1.0 + std::exp(-(delta_mf - mf_threshold)));
            double reversal_sign = (prev_choice == 1) ? -1.0 : 1.0;
            // Select gain based on previous outcome valence
            double current_gain = (prev_outcome == 1) ? explore_gain_win : explore_gain_loss;
            v_effective += reversal_sign * current_gain * gate;
        }
        
        prev_mf_energy = l2_mf;
        prev_choice = ch;
        prev_outcome = out[t];
        
        double safe_v = std::abs(v_effective) < 1e-4 ? (v_effective >= 0 ? 1e-4 : -1e-4) : v_effective;
        D_vec.push_back(-2.0 * std::log(wiener_pdf(rt[t], ch, safe_v, a, t_nd)));
        
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
    return calc_pen_ll(D_vec);
}
