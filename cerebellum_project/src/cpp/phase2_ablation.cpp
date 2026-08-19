#include <Rcpp.h>
#include <vector>
#include <cmath>

using namespace Rcpp;

inline double wiener_pdf(double t_raw, int choice, double v, double a, double t_nd, double eps = 1e-9) {
  double t = t_raw - t_nd;
  if (t <= 0.001) return 1e-12;
  double sign = (choice == 1) ? 1.0 : -1.0;
  double w = 0.50; double x0 = (choice == 1) ? (1.0 - w) : w;
  double drift_term = std::exp(sign * v * a * w - 0.5 * v * v * t);
  double tt = t / (a * a);
  double sum = 0.0;
  if (tt >= 0.08) {
    for (int k = 1; k <= 30; ++k) {
      double term = (double)k * std::sin((double)k * M_PI * x0) * std::exp(-0.5 * k * k * M_PI * M_PI * tt);
      sum += term;
      if (std::abs(term) < eps) break;
    }
    sum *= M_PI;
  } else {
    double sqrt_tt = std::sqrt(tt);
    for (int k = -15; k <= 15; ++k) {
      double num = (x0 + 2.0 * k);
      double term = num * std::exp(-0.5 * (num * num) / tt);
      sum += term;
    }
    sum /= (std::sqrt(2.0 * M_PI) * tt * sqrt_tt);
  }
  return std::max(1e-12, (drift_term / (a * a)) * sum);
}

class SimpleRNG {
    uint32_t state;
public:
    SimpleRNG(uint32_t seed) : state(seed) {}
    uint32_t next() { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state; }
    double runif() { return (next() % 1000000) / 1000000.0; }
    double rnorm() {
        double u1 = std::max(1e-6, runif()); double u2 = runif();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
    }
};

// [[Rcpp::export]]
double eval_eccm_k(int k, const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    double a_e = std::exp(phi[0]);
    double t_nd_e = 1.0 / (1.0 + std::exp(-phi[1]));
    double beta_v_e = std::exp(phi[2]);
    double eta_LTP = 1.0 / (1.0 + std::exp(-phi[3]));
    double eta_LTD = 1.0 / (1.0 + std::exp(-phi[4]));
    double w_cb = phi[5]; 
    
    int N_MF = 2 * k, N_GC = 1024, N_MLI = 256;
    std::vector<double> W_PF1(N_GC, 0.0), W_PF2(N_GC, 0.0);
    std::vector<double> W_MLI1(N_MLI, 0.0), W_MLI2(N_MLI, 0.0);
    std::vector<std::vector<double>> W_MF_GC(N_GC, std::vector<double>(N_MF, 0.0));
    std::vector<std::vector<double>> W_GC_MLI(N_MLI, std::vector<double>(N_GC, 0.0));
    std::vector<double> mf(N_MF, 0.0);
    
    SimpleRNG rng(42);
    for (int i=0; i<N_GC; ++i) for (int j=0; j<N_MF; ++j) W_MF_GC[i][j] = rng.rnorm() / std::sqrt((double)N_MF);
    for (int i=0; i<N_MLI; ++i) for (int j=0; j<N_GC; ++j) W_GC_MLI[i][j] = rng.rnorm() / std::sqrt((double)N_GC);
    
    std::vector<double> D_eccm;
    double Q1_CTX = 0.0, Q2_CTX = 0.0;
    
    for (int t = 0; t < resp.size(); ++t) {
        int ch = resp[t]; double R_raw = (out[t] == 1) ? 1.0 : -1.0;
        
        std::vector<double> gc(N_GC, 0.0), mli(N_MLI, 0.0);
        for (int i=0; i<N_GC; ++i) {
            double act = 0.0; for (int j=0; j<N_MF; ++j) act += W_MF_GC[i][j] * mf[j];
            gc[i] = std::tanh(act);
        }
        for (int i=0; i<N_MLI; ++i) {
            double act = 0.0; for (int j=0; j<N_GC; ++j) act += W_GC_MLI[i][j] * gc[j];
            mli[i] = std::tanh(act);
        }
        
        double Q1_CB = 0.0, Q2_CB = 0.0;
        for (int i=0; i<N_GC; ++i) { Q1_CB += W_PF1[i]*gc[i]; Q2_CB += W_PF2[i]*gc[i]; }
        for (int i=0; i<N_MLI; ++i) { Q1_CB -= W_MLI1[i]*mli[i]; Q2_CB -= W_MLI2[i]*mli[i]; }
        
        double v_t = beta_v_e * ( (Q1_CTX * (1.0 + w_cb * Q1_CB)) - (Q2_CTX * (1.0 + w_cb * Q2_CB)) );
        double safe_v = std::abs(v_t) < 1e-4 ? (v_t >= 0 ? 1e-4 : -1e-4) : v_t;
        D_eccm.push_back(-2.0 * std::log(wiener_pdf(rt[t], ch, safe_v, a_e, t_nd_e)));
        
        double lr = (out[t] == 1) ? eta_LTP : eta_LTD;
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
        for(int j=k-2; j>=0; --j) { mf[j+1] = mf[j]; mf[k+j+1] = mf[k+j]; }
        mf[0] = (ch == 1) ? 1.0 : -1.0; mf[k] = R_raw;
    }
    
    double mean_t = (resp.size() + 1) / 2.0;
    double m_eccm = 0.0; double ss_t = 0.0; double mean_D = 0.0;
    for (int t=0; t<resp.size(); ++t) mean_D += D_eccm[t]; mean_D /= resp.size();
    for (int t=0; t<resp.size(); ++t) {
        m_eccm += (t + 1 - mean_t) * (D_eccm[t] - mean_D);
        ss_t += (t + 1 - mean_t) * (t + 1 - mean_t);
    }
    m_eccm /= ss_t;
    
    double total_D = 0.0;
    for (int t=0; t<resp.size(); ++t) total_D += D_eccm[t];
    
    return total_D + std::abs(m_eccm); // Penalized Likelihood
}

// [[Rcpp::export]]
double eval_lesioned(int k, const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    double a_e = std::exp(phi[0]);
    double t_nd_e = 1.0 / (1.0 + std::exp(-phi[1]));
    double beta_v_e = std::exp(phi[2]);
    double eta_LTP = 1.0 / (1.0 + std::exp(-phi[3]));
    double eta_LTD = 1.0 / (1.0 + std::exp(-phi[4]));
    double w_cb = phi[5]; 
    
    int N_MF = 2 * k;
    std::vector<double> W_PF1(N_MF, 0.0), W_PF2(N_MF, 0.0);
    std::vector<double> mf(N_MF, 0.0);
    
    std::vector<double> D_eccm;
    double Q1_CTX = 0.0, Q2_CTX = 0.0;
    
    for (int t = 0; t < resp.size(); ++t) {
        int ch = resp[t]; double R_raw = (out[t] == 1) ? 1.0 : -1.0;
        
        double Q1_CB = 0.0, Q2_CB = 0.0;
        for (int i=0; i<N_MF; ++i) { Q1_CB += W_PF1[i]*mf[i]; Q2_CB += W_PF2[i]*mf[i]; }
        
        // Soft bounds since we lost tanh
        Q1_CB = std::max(-1.0, std::min(1.0, Q1_CB));
        Q2_CB = std::max(-1.0, std::min(1.0, Q2_CB));
        
        double v_t = beta_v_e * ( (Q1_CTX * (1.0 + w_cb * Q1_CB)) - (Q2_CTX * (1.0 + w_cb * Q2_CB)) );
        double safe_v = std::abs(v_t) < 1e-4 ? (v_t >= 0 ? 1e-4 : -1e-4) : v_t;
        D_eccm.push_back(-2.0 * std::log(wiener_pdf(rt[t], ch, safe_v, a_e, t_nd_e)));
        
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
    
    double mean_t = (resp.size() + 1) / 2.0;
    double m_eccm = 0.0; double ss_t = 0.0; double mean_D = 0.0;
    for (int t=0; t<resp.size(); ++t) mean_D += D_eccm[t]; mean_D /= resp.size();
    for (int t=0; t<resp.size(); ++t) {
        m_eccm += (t + 1 - mean_t) * (D_eccm[t] - mean_D);
        ss_t += (t + 1 - mean_t) * (t + 1 - mean_t);
    }
    m_eccm /= ss_t;
    
    double total_D = 0.0;
    for (int t=0; t<resp.size(); ++t) total_D += D_eccm[t];
    
    return total_D + std::abs(m_eccm);
}
