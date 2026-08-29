#include <Rcpp.h>
#include "../models/shared_utils.h"

using namespace Rcpp;

// Included models
#include "../models/wsls.cpp"
#include "../models/eccm_intact.cpp"
#include "../models/eccm_cortical_rpe.cpp"
#include "../models/eccm_golgi_reversal.cpp"
#include "../models/eccm_golgi_asym_reversal.cpp"
#include "../models/eccm_mf_rev_ablated.cpp"
#include "../models/eccm_smooth_graph.cpp"
#include "../models/eccm_dynamic_boundary.cpp"
#include "../models/eccm_lca.cpp"
#include "../models/eccm_bvk.cpp"

// We need pointwise log likelihoods.
// Since the original functions return a single double, we can just copy them here and change the return type.
// But to keep it DRY, it's better to just implement a dedicated extractor for the required models.
// Let's implement pointwise extractors for models 0, 6, 18, 19, 20.

// Helper to get pointwise ll for Model 0 (WSLS)
NumericVector extract_pointwise_ll_0(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    double a = std::exp(phi[0]), t_nd = 1.0 / (1.0 + std::exp(-phi[1])), beta_v = std::exp(phi[2]);
    int T = resp.size();
    NumericVector ll(T);
    int last_ch = -1, last_out = -1;
    for (int t=0; t<T; ++t) {
        double v = 0.0;
        if (last_ch != -1) {
            int pred_ch = (last_out == 1) ? last_ch : (last_ch == 1 ? 2 : 1);
            v = (pred_ch == 1) ? beta_v : -beta_v;
        }
        double safe_v = std::abs(v) < 1e-4 ? (v >= 0 ? 1e-4 : -1e-4) : v;
        ll[t] = std::log(wiener_pdf(rt[t], resp[t], safe_v, a, t_nd));
        last_ch = resp[t]; last_out = out[t];
    }
    return ll;
}

// Q-Learning pointwise LL will be handled in R directly since it's an R function.

// Helper to get pointwise ll for Model 6 (ECCM Cortical RPE)
NumericVector extract_pointwise_ll_6(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt, const NumericVector& delta_t) {
    double a = std::exp(phi[0]), t_nd = 1.0 / (1.0 + std::exp(-phi[1])), beta_v = std::exp(phi[2]);
    double eta_LTP = 1.0 / (1.0 + std::exp(-phi[3])), eta_LTD = 1.0 / (1.0 + std::exp(-phi[4])), w_cb = phi[5]; 
    double lambda_shift = std::exp(phi[6]); 
    double gamma_suppress = std::exp(phi[7]); 
    int k = 80; int N_MF = 240, N_GC = 1024, N_MLI = 256; 
    std::vector<double> W_PF1(N_GC, 0.0), W_PF2(N_GC, 0.0), W_MLI1(N_MLI, 0.0), W_MLI2(N_MLI, 0.0);
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
        for (int i=0; i<N_GC; ++i) { double act = 0.0; for (int j=0; j<N_MF; ++j) act += W_MF_GC[i][j] * mf[j]; gc[i] = std::tanh(act); }
        for (int i=0; i<N_MLI; ++i) { double act = 0.0; for (int j=0; j<N_GC; ++j) act += W_GC_MLI[i][j] * gc[j]; mli[i] = std::tanh(act); }
        for (int i=0; i<N_GC; ++i) { Q1_CB += W_PF1[i]*gc[i]; Q2_CB += W_PF2[i]*gc[i]; }
        for (int i=0; i<N_MLI; ++i) { Q1_CB -= W_MLI1[i]*mli[i]; Q2_CB -= W_MLI2[i]*mli[i]; }
        double delta_cc = (ch == 1) ? std::abs(Q1_CTX - Q1_CB) : std::abs(Q2_CTX - Q2_CB);
        double v_t = beta_v * ( (Q1_CTX * (1.0 + w_cb * Q1_CB)) - (Q2_CTX * (1.0 + w_cb * Q2_CB)) );
        double v_effective = v_t * std::exp(-gamma_suppress * delta_cc);
        double safe_v = std::abs(v_effective) < 1e-4 ? (v_effective >= 0 ? 1e-4 : -1e-4) : v_effective;
        ll[t] = std::log(wiener_pdf(rt[t], ch, safe_v, a, t_nd));
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
        double dt = delta_t[t]; int shifts = 1;
        if (dt > 2.0 && !std::isnan(dt)) shifts = 1 + std::floor(lambda_shift * (dt - 2.0));
        shifts = std::min(shifts, k);
        for (int s = 0; s < shifts; ++s) {
            for(int j=k-2; j>=0; --j) { mf[j+1] = mf[j]; mf[k+j+1] = mf[k+j]; mf[2*k+j+1] = mf[2*k+j]; }
            if (s == 0) { mf[0] = (ch == 1) ? 1.0 : -1.0; mf[k] = R_raw; mf[2*k] = err_ctx; } 
            else { mf[0] = 0.0; mf[k] = 0.0; mf[2*k] = 0.0; }
        }
    }
    return ll;
}

// Helper to get pointwise ll for Model 18 (MF-Reversal Sym)
NumericVector extract_pointwise_ll_18(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt, const NumericVector& delta_t) {
    double a = std::exp(phi[0]), t_nd = 1.0 / (1.0 + std::exp(-phi[1])), beta_v = std::exp(phi[2]);
    double eta_LTP = 1.0 / (1.0 + std::exp(-phi[3])), eta_LTD = 1.0 / (1.0 + std::exp(-phi[4])), w_cb = phi[5]; 
    double lambda_shift = std::exp(phi[6]); 
    double gamma_suppress = std::exp(phi[7]); 
    double theta_golgi = std::exp(phi[8]);   
    double mf_threshold = phi[9];            
    double explore_gain = std::exp(phi[10]); 
    int k = 80; int N_MF = 240, N_GC = 1024, N_MLI = 256; 
    std::vector<double> W_PF1(N_GC, 0.0), W_PF2(N_GC, 0.0), W_MLI1(N_MLI, 0.0), W_MLI2(N_MLI, 0.0), mf(N_MF, 0.0);
    std::vector<std::vector<double>> W_MF_GC(N_GC, std::vector<double>(N_MF, 0.0)), W_GC_MLI(N_MLI, std::vector<double>(N_GC, 0.0));
    SimpleRNG rng(42);
    for (int i=0; i<N_GC; ++i) for (int j=0; j<N_MF; ++j) W_MF_GC[i][j] = rng.rnorm() / std::sqrt((double)N_MF);
    for (int i=0; i<N_MLI; ++i) for (int j=0; j<N_GC; ++j) W_GC_MLI[i][j] = rng.rnorm() / std::sqrt((double)N_GC);
    int T = resp.size();
    NumericVector ll(T);
    double Q1_CTX = 0.0, Q2_CTX = 0.0;
    double prev_mf_energy = 0.0; int prev_choice = 0; 
    for (int t = 0; t < T; ++t) {
        int ch = resp[t]; double R_raw = (out[t] == 1) ? 1.0 : -1.0;
        double Q1_CB = 0.0, Q2_CB = 0.0;
        double mf_energy = 0.0;
        for (int j=0; j<N_MF; ++j) mf_energy += mf[j] * mf[j];
        double l2_mf = std::sqrt(mf_energy);
        double gain = 1.0 / (1.0 + theta_golgi * l2_mf);
        std::vector<double> gc(N_GC, 0.0), mli(N_MLI, 0.0);
        for (int i=0; i<N_GC; ++i) { double act = 0.0; for (int j=0; j<N_MF; ++j) act += W_MF_GC[i][j] * mf[j]; gc[i] = std::tanh(act * gain); }
        for (int i=0; i<N_MLI; ++i) { double act = 0.0; for (int j=0; j<N_GC; ++j) act += W_GC_MLI[i][j] * gc[j]; mli[i] = std::tanh(act); }
        for (int i=0; i<N_GC; ++i) { Q1_CB += W_PF1[i]*gc[i]; Q2_CB += W_PF2[i]*gc[i]; }
        for (int i=0; i<N_MLI; ++i) { Q1_CB -= W_MLI1[i]*mli[i]; Q2_CB -= W_MLI2[i]*mli[i]; }
        double delta_cc = (ch == 1) ? std::abs(Q1_CTX - Q1_CB) : std::abs(Q2_CTX - Q2_CB);
        double v_t = beta_v * ( (Q1_CTX * (1.0 + w_cb * Q1_CB)) - (Q2_CTX * (1.0 + w_cb * Q2_CB)) );
        double v_effective = v_t * std::exp(-gamma_suppress * delta_cc);
        if (prev_choice != 0) {
            double delta_mf = l2_mf - prev_mf_energy;
            double gate = 1.0 / (1.0 + std::exp(-(delta_mf - mf_threshold)));
            double reversal_sign = (prev_choice == 1) ? -1.0 : 1.0;
            v_effective += reversal_sign * explore_gain * gate;
        }
        prev_mf_energy = l2_mf; prev_choice = ch;
        double safe_v = std::abs(v_effective) < 1e-4 ? (v_effective >= 0 ? 1e-4 : -1e-4) : v_effective;
        ll[t] = std::log(wiener_pdf(rt[t], ch, safe_v, a, t_nd));
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
    return ll;
}

// Helper to get pointwise ll for Model 19 (MF-Reversal Asym)
NumericVector extract_pointwise_ll_19(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt, const NumericVector& delta_t) {
    double a = std::exp(phi[0]), t_nd = 1.0 / (1.0 + std::exp(-phi[1])), beta_v = std::exp(phi[2]);
    double eta_LTP = 1.0 / (1.0 + std::exp(-phi[3])), eta_LTD = 1.0 / (1.0 + std::exp(-phi[4])), w_cb = phi[5]; 
    double lambda_shift = std::exp(phi[6]); 
    double gamma_suppress = std::exp(phi[7]); 
    double theta_golgi = std::exp(phi[8]);       
    double mf_threshold = phi[9];                 
    double explore_gain_win = std::exp(phi[10]);  
    double explore_gain_loss = std::exp(phi[11]); 
    int k = 80; int N_MF = 240, N_GC = 1024, N_MLI = 256; 
    std::vector<double> W_PF1(N_GC, 0.0), W_PF2(N_GC, 0.0), W_MLI1(N_MLI, 0.0), W_MLI2(N_MLI, 0.0), mf(N_MF, 0.0);
    std::vector<std::vector<double>> W_MF_GC(N_GC, std::vector<double>(N_MF, 0.0)), W_GC_MLI(N_MLI, std::vector<double>(N_GC, 0.0));
    SimpleRNG rng(42);
    for (int i=0; i<N_GC; ++i) for (int j=0; j<N_MF; ++j) W_MF_GC[i][j] = rng.rnorm() / std::sqrt((double)N_MF);
    for (int i=0; i<N_MLI; ++i) for (int j=0; j<N_GC; ++j) W_GC_MLI[i][j] = rng.rnorm() / std::sqrt((double)N_GC);
    int T = resp.size();
    NumericVector ll(T);
    double Q1_CTX = 0.0, Q2_CTX = 0.0;
    double prev_mf_energy = 0.0; int prev_choice = 0; int prev_outcome = -1;
    for (int t = 0; t < T; ++t) {
        int ch = resp[t]; double R_raw = (out[t] == 1) ? 1.0 : -1.0;
        double Q1_CB = 0.0, Q2_CB = 0.0;
        double mf_energy = 0.0;
        for (int j=0; j<N_MF; ++j) mf_energy += mf[j] * mf[j];
        double l2_mf = std::sqrt(mf_energy);
        double gain = 1.0 / (1.0 + theta_golgi * l2_mf);
        std::vector<double> gc(N_GC, 0.0), mli(N_MLI, 0.0);
        for (int i=0; i<N_GC; ++i) { double act = 0.0; for (int j=0; j<N_MF; ++j) act += W_MF_GC[i][j] * mf[j]; gc[i] = std::tanh(act * gain); }
        for (int i=0; i<N_MLI; ++i) { double act = 0.0; for (int j=0; j<N_GC; ++j) act += W_GC_MLI[i][j] * gc[j]; mli[i] = std::tanh(act); }
        for (int i=0; i<N_GC; ++i) { Q1_CB += W_PF1[i]*gc[i]; Q2_CB += W_PF2[i]*gc[i]; }
        for (int i=0; i<N_MLI; ++i) { Q1_CB -= W_MLI1[i]*mli[i]; Q2_CB -= W_MLI2[i]*mli[i]; }
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
        prev_mf_energy = l2_mf; prev_choice = ch; prev_outcome = out[t];
        double safe_v = std::abs(v_effective) < 1e-4 ? (v_effective >= 0 ? 1e-4 : -1e-4) : v_effective;
        ll[t] = std::log(wiener_pdf(rt[t], ch, safe_v, a, t_nd));
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
    return ll;
}

// Helper to get pointwise ll for Model 20 (Ablated)
NumericVector extract_pointwise_ll_20(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt, const NumericVector& delta_t) {
    double a = std::exp(phi[0]), t_nd = 1.0 / (1.0 + std::exp(-phi[1])), beta_v = std::exp(phi[2]);
    double eta_LTP = 1.0 / (1.0 + std::exp(-phi[3])), eta_LTD = 1.0 / (1.0 + std::exp(-phi[4])), w_cb = phi[5]; 
    double lambda_shift = std::exp(phi[6]); 
    double gamma_suppress = std::exp(phi[7]); 
    double mf_threshold = phi[8];                 
    double explore_gain_win = std::exp(phi[9]);  
    double explore_gain_loss = std::exp(phi[10]); 
    int k = 80; int N_MF = 240; 
    std::vector<double> W_MF_PC1(N_MF, 0.0), W_MF_PC2(N_MF, 0.0), mf(N_MF, 0.0);
    int T = resp.size();
    NumericVector ll(T);
    double Q1_CTX = 0.0, Q2_CTX = 0.0;
    double prev_mf_energy = 0.0; int prev_choice = 0; int prev_outcome = -1;
    for (int t = 0; t < T; ++t) {
        int ch = resp[t]; double R_raw = (out[t] == 1) ? 1.0 : -1.0;
        double Q1_CB = 0.0, Q2_CB = 0.0;
        double mf_energy = 0.0;
        for (int j=0; j<N_MF; ++j) mf_energy += mf[j] * mf[j];
        double l2_mf = std::sqrt(mf_energy);
        for (int j=0; j<N_MF; ++j) { Q1_CB += W_MF_PC1[j] * mf[j]; Q2_CB += W_MF_PC2[j] * mf[j]; }
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
        prev_mf_energy = l2_mf; prev_choice = ch; prev_outcome = out[t];
        double safe_v = std::abs(v_effective) < 1e-4 ? (v_effective >= 0 ? 1e-4 : -1e-4) : v_effective;
        ll[t] = std::log(wiener_pdf(rt[t], ch, safe_v, a, t_nd));
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
    return ll;
}

// Helper to get pointwise ll for Model 21 (Smooth Graph)
NumericVector extract_pointwise_ll_21(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt, const NumericVector& delta_t) {
    double a = std::exp(phi[0]), t_nd = 1.0 / (1.0 + std::exp(-phi[1])), beta_v = std::exp(phi[2]);
    double eta_win = 1.0 / (1.0 + std::exp(-phi[3])); 
    double eta_loss = 1.0 / (1.0 + std::exp(-phi[4])); 
    double w_cb = phi[5]; 
    double lambda_shift = std::exp(phi[6]); 
    double gamma_suppress = std::exp(phi[7]);
    double alpha_diff = std::exp(phi[8]);

    int k = 80; int N_MF = 240, N_GC = 1024, N_MLI = 256;
    
    std::vector<double> W_PF1(N_GC, 0.0), W_PF2(N_GC, 0.0);
    std::vector<double> W_MLI1(N_MLI, 0.0), W_MLI2(N_MLI, 0.0);
    std::vector<std::vector<double>> W_MF_GC(N_GC, std::vector<double>(N_MF, 0.0));
    std::vector<std::vector<double>> W_GC_MLI(N_MLI, std::vector<double>(N_GC, 0.0));
    std::vector<double> mf(N_MF, 0.0);
    
    uint32_t state = 42;
    auto next_rnd = [&]() { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state; };
    auto runif = [&]() { return (next_rnd() % 1000000) / 1000000.0; };
    auto rnorm = [&]() {
        double u1 = std::max(1e-6, runif()); double u2 = runif();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
    };

    for (int i=0; i<N_GC; ++i) for (int j=0; j<N_MF; ++j) W_MF_GC[i][j] = rnorm() / std::sqrt((double)N_MF);
    for (int i=0; i<N_MLI; ++i) for (int j=0; j<N_GC; ++j) W_GC_MLI[i][j] = rnorm() / std::sqrt((double)N_GC);
    
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
        
        double delta_cc = (ch == 1) ? std::abs(Q1_CTX - Q1_CB) : std::abs(Q2_CTX - Q2_CB);
        double v_t = beta_v * ( (Q1_CTX * (1.0 + w_cb * Q1_CB)) - (Q2_CTX * (1.0 + w_cb * Q2_CB)) );
        double v_effective = v_t * std::exp(-gamma_suppress * delta_cc);
        
        double safe_v = std::abs(v_effective) < 1e-4 ? (v_effective >= 0 ? 1e-4 : -1e-4) : v_effective;
        ll[t] = std::log(wiener_pdf(rt[t], resp[t], safe_v, a, t_nd));
        
        double dt = delta_t[t];
        double dt_safe = std::isnan(dt) ? 2.0 : dt;
        double diffusion_envelope = 1.0 - std::exp(-alpha_diff * dt_safe);
        
        double lr_ctx = (out[t] == 1) ? eta_win : eta_loss;
        double lr_cb = diffusion_envelope * ((out[t] == 1) ? (eta_win / N_GC) : (eta_loss / N_GC));
        double lr_mli = diffusion_envelope * ((out[t] == 1) ? (eta_win / N_MLI) : (eta_loss / N_MLI));
        
        if (ch == 1) {
            double err_cb = R_raw - Q1_CB;
            Q1_CTX += lr_ctx * (R_raw - Q1_CTX);
            for (int i=0; i<N_GC; ++i) W_PF1[i] += lr_cb * err_cb * gc[i];
            for (int i=0; i<N_MLI; ++i) W_MLI1[i] -= lr_mli * err_cb * mli[i];
        } else {
            double err_cb = R_raw - Q2_CB;
            Q2_CTX += lr_ctx * (R_raw - Q2_CTX);
            for (int i=0; i<N_GC; ++i) W_PF2[i] += lr_cb * err_cb * gc[i];
            for (int i=0; i<N_MLI; ++i) W_MLI2[i] -= lr_mli * err_cb * mli[i];
        }
        
        int shifts = 1;
        if (dt_safe > 2.0) {
            shifts = 1 + std::floor(lambda_shift * (dt_safe - 2.0));
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
                mf[2*k] = (ch == 1) ? (R_raw - Q1_CTX) : (R_raw - Q2_CTX);
            } else {
                mf[0] = 0.0; mf[k] = 0.0; mf[2*k] = 0.0;
            }
        }
    }
    return ll;
}

// Helper to get pointwise ll for Model 10 (Dynamic Boundary)
NumericVector extract_pointwise_ll_10(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt, const NumericVector& delta_t) {
    double a = std::exp(phi[0]), t_nd = 1.0 / (1.0 + std::exp(-phi[1])), beta_v = std::exp(phi[2]);
    double eta_LTP = 1.0 / (1.0 + std::exp(-phi[3])), eta_LTD = 1.0 / (1.0 + std::exp(-phi[4])), w_cb = phi[5]; 
    double lambda_shift = std::exp(phi[6]); 
    double gamma_v = std::exp(phi[7]); 
    double gamma_a = std::exp(phi[8]);
    
    int k = 80; int N_MF = 240, N_GC = 1024, N_MLI = 256;
    
    std::vector<double> W_PF1(N_GC, 0.0), W_PF2(N_GC, 0.0);
    std::vector<double> W_MLI1(N_MLI, 0.0), W_MLI2(N_MLI, 0.0);
    std::vector<std::vector<double>> W_MF_GC(N_GC, std::vector<double>(N_MF, 0.0));
    std::vector<std::vector<double>> W_GC_MLI(N_MLI, std::vector<double>(N_GC, 0.0));
    std::vector<double> mf(N_MF, 0.0);
    
    uint32_t state = 42;
    auto next_rnd = [&]() { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state; };
    auto runif = [&]() { return (next_rnd() % 1000000) / 1000000.0; };
    auto rnorm = [&]() {
        double u1 = std::max(1e-6, runif()); double u2 = runif();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
    };

    for (int i=0; i<N_GC; ++i) for (int j=0; j<N_MF; ++j) W_MF_GC[i][j] = rnorm() / std::sqrt((double)N_MF);
    for (int i=0; i<N_MLI; ++i) for (int j=0; j<N_GC; ++j) W_GC_MLI[i][j] = rnorm() / std::sqrt((double)N_GC);
    
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
        
        double delta_cc = (ch == 1) ? std::abs(Q1_CTX - Q1_CB) : std::abs(Q2_CTX - Q2_CB);
        double v_t = beta_v * ( (Q1_CTX * (1.0 + w_cb * Q1_CB)) - (Q2_CTX * (1.0 + w_cb * Q2_CB)) );
        
        // DYNAMIC BOUNDARY
        double v_effective = v_t * std::exp(-gamma_v * delta_cc);
        double a_effective = a * std::max(0.1, std::exp(-gamma_a * delta_cc));
        
        double safe_v = std::abs(v_effective) < 1e-4 ? (v_effective >= 0 ? 1e-4 : -1e-4) : v_effective;
        ll[t] = std::log(wiener_pdf(rt[t], resp[t], safe_v, a_effective, t_nd));
        
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
            if (s == 0) { mf[0] = (ch == 1) ? 1.0 : -1.0; mf[k] = R_raw; mf[2*k] = err_ctx; } 
            else { mf[0] = 0.0; mf[k] = 0.0; mf[2*k] = 0.0; }
        }
    }
    return ll;
}

// Helper to get pointwise ll for Model 22 (LCA)
NumericVector extract_pointwise_ll_22(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt, const NumericVector& delta_t) {
    double A = std::exp(phi[0]), t_nd = 1.0 / (1.0 + std::exp(-phi[1])), beta_v = std::exp(phi[2]);
    double eta_LTP = 1.0 / (1.0 + std::exp(-phi[3])), eta_LTD = 1.0 / (1.0 + std::exp(-phi[4])), w_cb = phi[5]; 
    double lambda_shift = std::exp(phi[6]); 
    double gamma_suppress = std::exp(phi[7]); 
    double leak_lambda = std::exp(phi[8]); 
    double mutual_omega = std::exp(phi[9]); 
    
    int k = 80; int N_MF = 240, N_GC = 1024, N_MLI = 256;
    
    std::vector<double> W_PF1(N_GC, 0.0), W_PF2(N_GC, 0.0);
    std::vector<double> W_MLI1(N_MLI, 0.0), W_MLI2(N_MLI, 0.0);
    std::vector<std::vector<double>> W_MF_GC(N_GC, std::vector<double>(N_MF, 0.0));
    std::vector<std::vector<double>> W_GC_MLI(N_MLI, std::vector<double>(N_GC, 0.0));
    std::vector<double> mf(N_MF, 0.0);
    
    uint32_t state = 42;
    auto next_rnd = [&]() { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state; };
    auto runif = [&]() { return (next_rnd() % 1000000) / 1000000.0; };
    auto rnorm = [&]() {
        double u1 = std::max(1e-6, runif()); double u2 = runif();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
    };

    for (int i=0; i<N_GC; ++i) for (int j=0; j<N_MF; ++j) W_MF_GC[i][j] = rnorm() / std::sqrt((double)N_MF);
    for (int i=0; i<N_MLI; ++i) for (int j=0; j<N_GC; ++j) W_GC_MLI[i][j] = rnorm() / std::sqrt((double)N_GC);
    
    int T = resp.size();
    NumericVector ll(T);
    double Q1_CTX = 0.0, Q2_CTX = 0.0;
    int rnd_idx = 0;
    
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
        
        double delta_cc1 = std::abs(Q1_CTX - Q1_CB);
        double delta_cc2 = std::abs(Q2_CTX - Q2_CB);
        
        double base_v1 = beta_v * (Q1_CTX * (1.0 + w_cb * Q1_CB));
        double base_v2 = beta_v * (Q2_CTX * (1.0 + w_cb * Q2_CB));
        
        double v1 = std::max(0.0, base_v1) * std::exp(-gamma_suppress * delta_cc1);
        double v2 = std::max(0.0, base_v2) * std::exp(-gamma_suppress * delta_cc2);
        
        double pdf = simulate_lca_pdf(rt[t], ch, v1, v2, leak_lambda, mutual_omega, 1.0, A, t_nd, rnd_idx);
        ll[t] = std::log(pdf);
        
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
        
        double dt_val = delta_t[t];
        int shifts = 1;
        if (dt_val > 2.0 && !std::isnan(dt_val)) shifts = 1 + std::floor(lambda_shift * (dt_val - 2.0));
        shifts = std::min(shifts, k);

        for (int s = 0; s < shifts; ++s) {
            for(int j=k-2; j>=0; --j) { 
                mf[j+1] = mf[j]; 
                mf[k+j+1] = mf[k+j]; 
                mf[2*k+j+1] = mf[2*k+j];
            }
            if (s == 0) { mf[0] = (ch == 1) ? 1.0 : -1.0; mf[k] = R_raw; mf[2*k] = err_ctx; } 
            else { mf[0] = 0.0; mf[k] = 0.0; mf[2*k] = 0.0; }
        }
    }
    return ll;
}

// Helper to get pointwise ll for Model 23 (BVK)
NumericVector extract_pointwise_ll_23(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt, const NumericVector& delta_t) {
    double a = std::exp(phi[0]), t_nd = 1.0 / (1.0 + std::exp(-phi[1])), beta_v = std::exp(phi[2]);
    double eta_LTP = 1.0 / (1.0 + std::exp(-phi[3])), eta_LTD = 1.0 / (1.0 + std::exp(-phi[4])), w_cb = phi[5]; 
    double lambda_shift = std::exp(phi[6]); 
    double gamma_suppress = std::exp(phi[7]); 
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
        ll[t] = std::log(wiener_pdf_w(rt[t], ch, safe_v, a, t_nd, w_t));
        
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
        
        double dt_val = delta_t[t];
        int shifts = 1;
        if (dt_val > 2.0 && !std::isnan(dt_val)) shifts = 1 + std::floor(lambda_shift * (dt_val - 2.0));
        shifts = std::min(shifts, k);

        for (int s = 0; s < shifts; ++s) {
            for(int j=k-2; j>=0; --j) { 
                mf[j+1] = mf[j]; 
                mf[k+j+1] = mf[k+j]; 
                mf[2*k+j+1] = mf[2*k+j];
            }
            if (s == 0) { mf[0] = (ch == 1) ? 1.0 : -1.0; mf[k] = R_raw; mf[2*k] = err_ctx; } 
            else { mf[0] = 0.0; mf[k] = 0.0; mf[2*k] = 0.0; }
        }
    }
    return ll;
}

// [[Rcpp::export]]
NumericMatrix extract_all_pointwise_ll(int model_type, NumericMatrix posterior_samples, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt, const NumericVector& delta_t = NumericVector::create()) {
    int iters = posterior_samples.nrow();
    int T = resp.size();
    NumericMatrix all_ll(iters, T);
    for (int i = 0; i < iters; ++i) {
        std::vector<double> phi(posterior_samples.ncol());
        for (int p=0; p<posterior_samples.ncol(); ++p) phi[p] = posterior_samples(i, p);
        
        NumericVector ll;
        if (model_type == 0) ll = extract_pointwise_ll_0(phi, resp, out, rt);
        else if (model_type == 6) ll = extract_pointwise_ll_6(phi, resp, out, rt, delta_t);
        else if (model_type == 10) ll = extract_pointwise_ll_10(phi, resp, out, rt, delta_t);
        else if (model_type == 18) ll = extract_pointwise_ll_18(phi, resp, out, rt, delta_t);
        else if (model_type == 19) ll = extract_pointwise_ll_19(phi, resp, out, rt, delta_t);
        else if (model_type == 20) ll = extract_pointwise_ll_20(phi, resp, out, rt, delta_t);
        else if (model_type == 21) ll = extract_pointwise_ll_21(phi, resp, out, rt, delta_t);
        else if (model_type == 22) ll = extract_pointwise_ll_22(phi, resp, out, rt, delta_t);
        else if (model_type == 23) ll = extract_pointwise_ll_23(phi, resp, out, rt, delta_t);
        
        for (int t=0; t<T; ++t) all_ll(i, t) = ll[t];
    }
    return all_ll;
}
