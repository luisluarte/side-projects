#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace Rcpp;

// [[Rcpp::export]]
List eval_metrics_smooth_graph(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt, const NumericVector& delta_t) {
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
    
    std::vector<double> prob_ch1;
    std::vector<double> exp_rt;
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
        
        double delta_cc = (ch == 1) ? std::abs(Q1_CTX - Q1_CB) : std::abs(Q2_CTX - Q2_CB);
        double v_t = beta_v * ( (Q1_CTX * (1.0 + w_cb * Q1_CB)) - (Q2_CTX * (1.0 + w_cb * Q2_CB)) );
        double v_effective = v_t * std::exp(-gamma_suppress * delta_cc);
        
        double p1 = 1.0 / (1.0 + std::exp(-v_effective * a));
        prob_ch1.push_back(p1);
        
        double ert = t_nd;
        if (std::abs(v_effective) < 1e-4) ert += (a * a) / 4.0;
        else ert += (a / (2.0 * v_effective)) * std::tanh(v_effective * a / 2.0);
        exp_rt.push_back(ert);
        
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
    
    return List::create(Named("prob_ch1") = prob_ch1, Named("exp_rt") = exp_rt);
}
