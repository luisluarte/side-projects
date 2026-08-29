#include <Rcpp.h>
#include "shared_utils.h"

using namespace Rcpp;

// [[Rcpp::export]]
inline double eval_eccm_intact(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    // a = decision boundary
    // t_nd = non-decision time
    // beta_v = drift rate
    double a = std::exp(phi[0]), t_nd = 1.0 / (1.0 + std::exp(-phi[1])), beta_v = std::exp(phi[2]);
    // eta_LTP = assymentric update when R^(t) = +1
    // eta_LTD = assymentric update when R^(t) = -1
    double eta_LTP = 1.0 / (1.0 + std::exp(-phi[3])), eta_LTD = 1.0 / (1.0 + std::exp(-phi[4])), w_cb = phi[5];
    // cerebellar dimension
    int k = 80; int N_MF = 160, N_GC = 1024, N_MLI = 256;

    // read out weight vectors for purkinje
    std::vector<double> W_PF1(N_GC, 0.0), W_PF2(N_GC, 0.0);
    // read out weight vectors for mli (molecular layer interneurons)
    std::vector<double> W_MLI1(N_MLI, 0.0), W_MLI2(N_MLI, 0.0);

    // forward projection matrix (mf -> gc) : expansion
    std::vector<std::vector<double>> W_MF_GC(N_GC, std::vector<double>(N_MF, 0.0));
    // forward projection matrix (gc -> mli) : compression
    std::vector<std::vector<double>> W_GC_MLI(N_MLI, std::vector<double>(N_GC, 0.0));
    // init empty mossy fiber shift register
    std::vector<double> mf(N_MF, 0.0);

    SimpleRNG rng(42);
    // variance preserving random projection init
    for (int i=0; i<N_GC; ++i) for (int j=0; j<N_MF; ++j) W_MF_GC[i][j] = rng.rnorm() / std::sqrt((double)N_MF);
    for (int i=0; i<N_MLI; ++i) for (int j=0; j<N_GC; ++j) W_GC_MLI[i][j] = rng.rnorm() / std::sqrt((double)N_GC);

    // store deviance
    std::vector<double> D_vec;
    // CTX value trackers
    double Q1_CTX = 0.0, Q2_CTX = 0.0;

    // hot loop
    for (int t = 0; t < resp.size(); ++t) {
      // map discrete choice and reward signal
        int ch = resp[t]; double R_raw = (out[t] == 1) ? 1.0 : -1.0;
        // start value estimates
        double Q1_CB = 0.0, Q2_CB = 0.0;

        // expansion step
        // start the gc and mli layer
        std::vector<double> gc(N_GC, 0.0), mli(N_MLI, 0.0);
        // non-linearity of gc layer
        for (int i=0; i<N_GC; ++i) {
            double act = 0.0; for (int j=0; j<N_MF; ++j) act += W_MF_GC[i][j] * mf[j];
            gc[i] = std::tanh(act);
        }
        // compression step
        for (int i=0; i<N_MLI; ++i) {
            double act = 0.0; for (int j=0; j<N_GC; ++j) act += W_GC_MLI[i][j] * gc[j];
            mli[i] = std::tanh(act);
        }
        // to CTX output
        for (int i=0; i<N_GC; ++i) { Q1_CB += W_PF1[i]*gc[i]; Q2_CB += W_PF2[i]*gc[i]; }
        // actual compression
        for (int i=0; i<N_MLI; ++i) { Q1_CB -= W_MLI1[i]*mli[i]; Q2_CB -= W_MLI2[i]*mli[i]; }

        // thalamic multiplicative modulation
        double v_t = beta_v * ( (Q1_CTX * (1.0 + w_cb * Q1_CB)) - (Q2_CTX * (1.0 + w_cb * Q2_CB)) );
        // numerical stabilization
        double safe_v = std::abs(v_t) < 1e-4 ? (v_t >= 0 ? 1e-4 : -1e-4) : v_t;
        // pdf for empirical choice + reaction time
        D_vec.push_back(-2.0 * std::log(wiener_pdf(rt[t], ch, safe_v, a, t_nd)));

        // implementation of assymetric learning rates
        // if outcome is 1 -> then we generate LTP
        // if outcome is 0 -> then we generate LTD
        double lr = (out[t] == 1) ? eta_LTP : eta_LTD;
        // normalize by layer size
        double lr_cb = lr / (double)N_GC, lr_mli = lr / (double)N_MLI;

        // apply LTP/LTD
        if (ch == 1) {
            double err_cb = R_raw - Q1_CB;
            Q1_CTX += (out[t] == 1) ? eta_LTP*(1.0-Q1_CTX) : eta_LTD*(-1.0-Q1_CTX);
            // plasticity option 1
            for (int i=0; i<N_GC; ++i) W_PF1[i] += lr_cb * err_cb * gc[i];
            for (int i=0; i<N_MLI; ++i) W_MLI1[i] -= lr_mli * err_cb * mli[i];
        } else {
            double err_cb = R_raw - Q2_CB;
            Q2_CTX += (out[t] == 1) ? eta_LTP*(1.0-Q2_CTX) : eta_LTD*(-1.0-Q2_CTX);
            // plasticity option 2
            for (int i=0; i<N_GC; ++i) W_PF2[i] += lr_cb * err_cb * gc[i];
            for (int i=0; i<N_MLI; ++i) W_MLI2[i] -= lr_mli * err_cb * mli[i];
        }

        // shift register
        for(int j=k-2; j>=0; --j) { mf[j+1] = mf[j]; mf[k+j+1] = mf[k+j]; }
        mf[0] = (ch == 1) ? 1.0 : -1.0; mf[k] = R_raw;
    }
    return calc_pen_ll(D_vec);
}
