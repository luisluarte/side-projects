#include <Rcpp.h>
#include "shared_utils.h"

using namespace Rcpp;

static std::vector<double> pc_rnorm;
static bool pc_initialized = false;
static inline double get_rnorm(int& idx) {
    if (!pc_initialized) {
        SimpleRNG rng(12345);
        pc_rnorm.resize(2000000);
        for(int i=0; i<2000000; ++i) pc_rnorm[i] = rng.rnorm();
        pc_initialized = true;
    }
    idx = (idx + 1) % 2000000;
    return pc_rnorm[idx];
}

inline double simulate_lca_pdf(double rt_obs, int ch_obs, double v1, double v2, double lambda, double omega, double sigma, double A, double t_nd, int& rnd_idx) {
    double t = rt_obs - t_nd;
    if (t <= 0.05) return 1e-12; 
    
    int N_sim = 1000; 
    double dt = 0.01; // slightly larger dt for speed
    double sqrt_dt = std::sqrt(dt);
    double sigma_sqdt = sigma * sqrt_dt;
    
    double bandwidth = 0.05;
    double max_T = t + 4.0 * bandwidth;
    int max_steps = (int)(max_T / dt);
    
    double sum_k = 0.0;
    
    for (int s = 0; s < N_sim; ++s) {
        double x1 = 0.0, x2 = 0.0;
        double current_t = 0.0;
        
        for (int step = 0; step < max_steps; ++step) {
            double dx1 = (v1 - lambda * x1 - omega * x2) * dt + sigma_sqdt * get_rnorm(rnd_idx);
            double dx2 = (v2 - lambda * x2 - omega * x1) * dt + sigma_sqdt * get_rnorm(rnd_idx);
            
            x1 += dx1;
            x2 += dx2;
            
            if (x1 < 0.0) x1 = 0.0;
            if (x2 < 0.0) x2 = 0.0;
            
            current_t += dt;
            
            if (x1 >= A || x2 >= A) {
                int ch_sim = (x1 >= x2) ? 1 : 2;
                if (ch_sim == ch_obs) {
                    double diff = current_t - t;
                    double k_val = std::exp(-0.5 * (diff * diff) / (bandwidth * bandwidth)) / (bandwidth * std::sqrt(2.0 * M_PI));
                    sum_k += k_val;
                }
                break;
            }
        }
    }
    
    double pdf = sum_k / (double)N_sim;
    return std::max(pdf, 1e-12);
}

// [[Rcpp::export]]
inline double eval_eccm_lca(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt, const NumericVector& delta_t) {
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
    
    SimpleRNG rng(42);
    for (int i=0; i<N_GC; ++i) for (int j=0; j<N_MF; ++j) W_MF_GC[i][j] = rng.rnorm() / std::sqrt((double)N_MF);
    for (int i=0; i<N_MLI; ++i) for (int j=0; j<N_GC; ++j) W_GC_MLI[i][j] = rng.rnorm() / std::sqrt((double)N_GC);
    
    std::vector<double> D_vec;
    double Q1_CTX = 0.0, Q2_CTX = 0.0;
    int rnd_idx = 0;
    
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
        
        double delta_cc1 = std::abs(Q1_CTX - Q1_CB);
        double delta_cc2 = std::abs(Q2_CTX - Q2_CB);
        
        double base_v1 = beta_v * (Q1_CTX * (1.0 + w_cb * Q1_CB));
        double base_v2 = beta_v * (Q2_CTX * (1.0 + w_cb * Q2_CB));
        
        double v1 = std::max(0.0, base_v1) * std::exp(-gamma_suppress * delta_cc1);
        double v2 = std::max(0.0, base_v2) * std::exp(-gamma_suppress * delta_cc2);
        
        // Pass to LCA simulator
        double pdf = simulate_lca_pdf(rt[t], ch, v1, v2, leak_lambda, mutual_omega, 1.0, A, t_nd, rnd_idx);
        D_vec.push_back(-2.0 * std::log(pdf));
        
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
        if (dt_val > 2.0 && !std::isnan(dt_val)) {
            shifts = 1 + std::floor(lambda_shift * (dt_val - 2.0));
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
