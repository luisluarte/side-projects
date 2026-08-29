#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace Rcpp;

double compute_entropy(const std::vector<double>& x) {
    double max_x = *std::max_element(x.begin(), x.end());
    double sum_exp = 0.0;
    for (double v : x) sum_exp += std::exp(v - max_x);
    double entropy = 0.0;
    for (double v : x) {
        double p = std::exp(v - max_x) / sum_exp;
        if (p > 1e-12) entropy -= p * std::log(p);
    }
    return entropy;
}

double compute_hoyer(const std::vector<double>& x) {
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    for (double v : x) {
        sum_abs += std::abs(v);
        sum_sq += v * v;
    }
    double n = x.size();
    if (sum_sq < 1e-12) return 1.0; // Completely sparse (all zeros)
    double l1 = sum_abs;
    double l2 = std::sqrt(sum_sq);
    return (std::sqrt(n) - (l1 / l2)) / (std::sqrt(n) - 1.0);
}

double compute_l2(const std::vector<double>& x) {
    double sum_sq = 0.0;
    for (double v : x) sum_sq += v * v;
    return std::sqrt(sum_sq);
}

// [[Rcpp::export]]
DataFrame extract_layer_metrics_cpp(int model_type, const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& delta_t) {
    double a = std::exp(phi[0]), t_nd = 1.0 / (1.0 + std::exp(-phi[1])), beta_v = std::exp(phi[2]);
    double eta_LTP = 1.0 / (1.0 + std::exp(-phi[3])), eta_LTD = 1.0 / (1.0 + std::exp(-phi[4])), w_cb = phi[5]; 
    double lambda_shift = std::exp(phi[6]); 
    double gamma_suppress = std::exp(phi[7]); 
    double theta_golgi = 0.0;
    if (model_type == 12 && phi.size() > 8) {
        theta_golgi = std::exp(phi[8]);
    }
    
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
    
    double Q1_CTX = 0.0, Q2_CTX = 0.0;
    
    std::vector<double> mf_ent, mf_spa, mf_l2;
    std::vector<double> gc_ent, gc_spa, gc_l2;
    std::vector<double> mli_ent, mli_spa, mli_l2;
    
    for (int t = 0; t < resp.size(); ++t) {
        int ch = resp[t]; double R_raw = (out[t] == 1) ? 1.0 : -1.0;
        double Q1_CB = 0.0, Q2_CB = 0.0;
        
        double mf_energy = 0.0;
        for (int j=0; j<N_MF; ++j) mf_energy += mf[j]*mf[j];
        double golgi_inhibition = 1.0;
        if (model_type == 12) {
            golgi_inhibition += theta_golgi * std::sqrt(mf_energy);
        }
        
        std::vector<double> gc(N_GC, 0.0), mli(N_MLI, 0.0);
        for (int i=0; i<N_GC; ++i) {
            double act = 0.0; for (int j=0; j<N_MF; ++j) act += W_MF_GC[i][j] * mf[j];
            gc[i] = std::tanh(act / golgi_inhibition);
        }
        for (int i=0; i<N_MLI; ++i) {
            double act = 0.0; for (int j=0; j<N_GC; ++j) act += W_GC_MLI[i][j] * gc[j];
            mli[i] = std::tanh(act);
        }
        for (int i=0; i<N_GC; ++i) { Q1_CB += W_PF1[i]*gc[i]; Q2_CB += W_PF2[i]*gc[i]; }
        for (int i=0; i<N_MLI; ++i) { Q1_CB -= W_MLI1[i]*mli[i]; Q2_CB -= W_MLI2[i]*mli[i]; }
        
        // Extract Metrics BEFORE the update (State of the network AT decision time)
        mf_ent.push_back(compute_entropy(mf));
        mf_spa.push_back(compute_hoyer(mf));
        mf_l2.push_back(compute_l2(mf));
        
        gc_ent.push_back(compute_entropy(gc));
        gc_spa.push_back(compute_hoyer(gc));
        gc_l2.push_back(compute_l2(gc));
        
        mli_ent.push_back(compute_entropy(mli));
        mli_spa.push_back(compute_hoyer(mli));
        mli_l2.push_back(compute_l2(mli));
        
        // Updates
        double lr = (out[t] == 1) ? eta_LTP : eta_LTD;
        double lr_cb = lr / (double)N_GC, lr_mli = lr / (double)N_MLI;
        
        double err_cb = 0.0, err_ctx = 0.0; 
        if (ch == 1) {
            err_cb = R_raw - Q1_CB; err_ctx = R_raw - Q1_CTX;
            Q1_CTX += (out[t] == 1) ? eta_LTP*(1.0-Q1_CTX) : eta_LTD*(-1.0-Q1_CTX);
            for (int i=0; i<N_GC; ++i) W_PF1[i] += lr_cb * err_cb * gc[i];
            for (int i=0; i<N_MLI; ++i) W_MLI1[i] -= lr_mli * err_cb * mli[i];
        } else {
            err_cb = R_raw - Q2_CB; err_ctx = R_raw - Q2_CTX;
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
                mf[j+1] = mf[j]; mf[k+j+1] = mf[k+j]; mf[2*k+j+1] = mf[2*k+j];
            }
            if (s == 0) {
                mf[0] = (ch == 1) ? 1.0 : -1.0; mf[k] = R_raw; mf[2*k] = err_ctx; 
            } else {
                mf[0] = 0.0; mf[k] = 0.0; mf[2*k] = 0.0;
            }
        }
    }
    
    return DataFrame::create(
        Named("MF_Ent") = mf_ent, Named("MF_Spa") = mf_spa, Named("MF_L2") = mf_l2,
        Named("GC_Ent") = gc_ent, Named("GC_Spa") = gc_spa, Named("GC_L2") = gc_l2,
        Named("MLI_Ent") = mli_ent, Named("MLI_Spa") = mli_spa, Named("MLI_L2") = mli_l2
    );
}
