#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace Rcpp;

// Helper to compute area under PR curve given predictions and true labels
double calc_pr_auc(const std::vector<double>& preds, const IntegerVector& true_labels) {
    // A simplified PR-AUC approximation or we can just return it to R.
    // Actually, returning the raw vectors to R is better so we can use the `prROC` package.
    return 0.0;
}

// [[Rcpp::export]]
List eval_metrics_wsls(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    double a = std::exp(phi[0]), t_nd = 1.0 / (1.0 + std::exp(-phi[1])), beta_v = std::exp(phi[2]);
    std::vector<double> prob_ch1;
    std::vector<double> exp_rt;
    int last_ch = -1, last_out = -1;
    for (int t=0; t<resp.size(); ++t) {
        double v = 0.0;
        if (last_ch != -1) {
            int pred_ch = (last_out == 1) ? last_ch : (last_ch == 1 ? 2 : 1);
            v = (pred_ch == 1) ? beta_v : -beta_v;
        }
        double p1 = 1.0 / (1.0 + std::exp(-v * a));
        prob_ch1.push_back(p1);
        
        double ert = t_nd;
        if (std::abs(v) < 1e-4) ert += (a * a) / 4.0;
        else ert += (a / (2.0 * v)) * std::tanh(v * a / 2.0);
        exp_rt.push_back(ert);
        
        last_ch = resp[t]; last_out = out[t];
    }
    return List::create(Named("prob_ch1") = prob_ch1, Named("exp_rt") = exp_rt);
}

// [[Rcpp::export]]
List eval_metrics_eccm(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt, bool lesioned) {
    double a = std::exp(phi[0]), t_nd = 1.0 / (1.0 + std::exp(-phi[1])), beta_v = std::exp(phi[2]);
    double eta_LTP = 1.0 / (1.0 + std::exp(-phi[3])), eta_LTD = 1.0 / (1.0 + std::exp(-phi[4])), w_cb = phi[5]; 
    int k = 80; int N_MF = 160, N_GC = 1024, N_MLI = 256;
    
    std::vector<double> W_PF1(lesioned ? N_MF : N_GC, 0.0), W_PF2(lesioned ? N_MF : N_GC, 0.0);
    std::vector<double> W_MLI1(N_MLI, 0.0), W_MLI2(N_MLI, 0.0);
    std::vector<std::vector<double>> W_MF_GC(N_GC, std::vector<double>(N_MF, 0.0));
    std::vector<std::vector<double>> W_GC_MLI(N_MLI, std::vector<double>(N_GC, 0.0));
    std::vector<double> mf(N_MF, 0.0);
    
    // Seed locally
    uint32_t state = 42;
    auto next_rnd = [&]() { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state; };
    auto runif = [&]() { return (next_rnd() % 1000000) / 1000000.0; };
    auto rnorm = [&]() {
        double u1 = std::max(1e-6, runif()); double u2 = runif();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
    };

    if (!lesioned) {
        for (int i=0; i<N_GC; ++i) for (int j=0; j<N_MF; ++j) W_MF_GC[i][j] = rnorm() / std::sqrt((double)N_MF);
        for (int i=0; i<N_MLI; ++i) for (int j=0; j<N_GC; ++j) W_GC_MLI[i][j] = rnorm() / std::sqrt((double)N_GC);
    }
    
    std::vector<double> prob_ch1;
    std::vector<double> exp_rt;
    double Q1_CTX = 0.0, Q2_CTX = 0.0;
    
    for (int t = 0; t < resp.size(); ++t) {
        int ch = resp[t]; double R_raw = (out[t] == 1) ? 1.0 : -1.0;
        double Q1_CB = 0.0, Q2_CB = 0.0;
        
        std::vector<double> gc(N_GC, 0.0), mli(N_MLI, 0.0);
        if (!lesioned) {
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
        } else {
            for (int i=0; i<N_MF; ++i) { Q1_CB += W_PF1[i]*mf[i]; Q2_CB += W_PF2[i]*mf[i]; }
            Q1_CB = std::max(-1.0, std::min(1.0, Q1_CB));
            Q2_CB = std::max(-1.0, std::min(1.0, Q2_CB));
        }
        
        double v_t = beta_v * ( (Q1_CTX * (1.0 + w_cb * Q1_CB)) - (Q2_CTX * (1.0 + w_cb * Q2_CB)) );
        
        // Compute predicted P(ch=1) and Exp(RT)
        double p1 = 1.0 / (1.0 + std::exp(-v_t * a));
        prob_ch1.push_back(p1);
        
        double ert = t_nd;
        if (std::abs(v_t) < 1e-4) ert += (a * a) / 4.0;
        else ert += (a / (2.0 * v_t)) * std::tanh(v_t * a / 2.0);
        exp_rt.push_back(ert);
        
        // Updates
        double lr = (out[t] == 1) ? eta_LTP : eta_LTD;
        if (!lesioned) {
            double lr_cb = lr / (double)N_GC, lr_mli = lr / (double)N_MLI;
            if (ch == 1) {
                double err_cb = R_raw - Q1_CB;
                Q1_CTX += (out[t] == 1) ? eta_LTP*(1.0-Q1_CTX) : eta_LTD*(-1.0-Q1_CTX);
                for (int i=0; i<N_GC; ++i) W_PF1[i] += lr_cb * err_cb * gc[i];
                for (int i=0; i<N_MLI; ++i) W_MLI1[i] -= lr_mli * err_cb * mli[i];
            } else {
                double err_cb = R_raw - Q2_CB;
                Q2_CTX += (out[t] == 1) ? eta_LTP*(1.0-Q2_CTX) : eta_LTD*(-1.0-Q2_CTX);
                for (int i=0; i<N_GC; ++i) W_PF2[i] += lr_cb * err_cb * gc[i];
                for (int i=0; i<N_MLI; ++i) W_MLI2[i] -= lr_mli * err_cb * mli[i];
            }
        } else {
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
        }
        
        for(int j=k-2; j>=0; --j) { mf[j+1] = mf[j]; mf[k+j+1] = mf[k+j]; }
        mf[0] = (ch == 1) ? 1.0 : -1.0; mf[k] = R_raw;
    }
    return List::create(Named("prob_ch1") = prob_ch1, Named("exp_rt") = exp_rt);
}
