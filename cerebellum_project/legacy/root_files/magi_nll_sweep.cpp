#include <Rcpp.h>
#include <cmath>
#include <vector>
using namespace Rcpp;

inline double signum(double x) { return (x > 0) ? 1.0 : ((x < 0) ? -1.0 : 0.0); }

// ==========================================
// 1. NEGATIVE LOG-LIKELIHOOD EVALUATORS
// ==========================================

// [[Rcpp::export]]
double get_nll_base(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    double a_base = std::exp(phi[0]);
    double mu_tnd = 1.0 / (1.0 + std::exp(-phi[1]));
    double kappa_v = std::exp(phi[2]); 
    double alpha_ctx = 1.0 / (1.0 + std::exp(-phi[3])); 
    double nll = 0.0;
    double Q_ctx[2] = {0.5, 0.5};
    for (int t = 0; t < resp.size(); ++t) {
        int ch = resp[t] - 1; double R = (out[t] == 1) ? 1.0 : 0.0;
        double v0 = kappa_v * std::max(Q_ctx[0], 0.01);
        double v1 = kappa_v * std::max(Q_ctx[1], 0.01);
        double p_ch = (ch == 1) ? (std::exp(v1) / (std::exp(v0) + std::exp(v1))) : (std::exp(v0) / (std::exp(v0) + std::exp(v1)));
        if (p_ch < 1e-5) p_ch = 1e-5;
        double v_t = kappa_v * std::max(Q_ctx[ch], 0.01);
        double mu = a_base / v_t; double lambda = a_base * a_base;
        double t_diff = rt[t] - mu_tnd; if (t_diff < 0.001) t_diff = 0.001;
        double pdf = std::sqrt(lambda / (2.0 * M_PI * std::pow(t_diff, 3))) * std::exp(- (lambda * std::pow(t_diff - mu, 2)) / (2.0 * mu * mu * t_diff));
        if (pdf < 1e-10) pdf = 1e-10;
        nll -= (std::log(p_ch) + std::log(pdf));
        Q_ctx[ch] += alpha_ctx * (R - Q_ctx[ch]);
    }
    return nll;
}

// [[Rcpp::export]]
double get_nll_qperturbed(const std::vector<double>& phi, const std::vector<double>& hyper, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    int N = 100; double lambda_min = hyper[0], lambda_max = hyper[1];
    double a_base = std::exp(phi[0]), t_nd = 1.0 / (1.0 + std::exp(-phi[1])), kappa_v = std::exp(phi[2]); 
    double alpha_ctx = 1.0 / (1.0 + std::exp(-phi[3])), alpha_cb = 1.0 / (1.0 + std::exp(-phi[4])); 
    double delta_cb = phi[9];
    std::vector<double> Tau(N);
    for(int i=0; i<N; ++i) Tau[i] = 1.0 / (lambda_min * std::pow(lambda_max / lambda_min, (double)i / (double)(N - 1)));
    double Q_ctx[2] = {0.5, 0.5}; std::vector<double> W_cb(N, 0.0), Z_trace(N, 0.0);
    double cb_bias = 0.0; double nll = 0.0;
    for (int t = 0; t < resp.size(); ++t) {
        int ch = resp[t] - 1; double R = (out[t] == 1) ? 1.0 : 0.0;
        double v0 = kappa_v * std::max(Q_ctx[0], 0.01) - cb_bias;
        double v1 = kappa_v * std::max(Q_ctx[1], 0.01) + cb_bias;
        if(v0 < 0.01) v0 = 0.01; if(v1 < 0.01) v1 = 0.01;
        double p_ch = (ch == 1) ? (std::exp(v1) / (std::exp(v0) + std::exp(v1))) : (std::exp(v0) / (std::exp(v0) + std::exp(v1)));
        if (p_ch < 1e-5) p_ch = 1e-5;
        double v_t = (ch == 1) ? v1 : v0;
        double mu = a_base / v_t; double lambda = a_base * a_base;
        double t_diff = rt[t] - t_nd; if (t_diff < 0.001) t_diff = 0.001;
        double pdf = std::sqrt(lambda / (2.0 * M_PI * std::pow(t_diff, 3))) * std::exp(- (lambda * std::pow(t_diff - mu, 2)) / (2.0 * mu * mu * t_diff));
        if (pdf < 1e-10) pdf = 1e-10;
        nll -= (std::log(p_ch) + std::log(pdf));
        double Q_active = std::max(Q_ctx[ch], 0.01);
        Q_ctx[ch] += alpha_ctx * (R - Q_ctx[ch]);
        double RPE_cb = R - Q_active; cb_bias = 0.0;
        for(int i=0; i<N; ++i) {
            Z_trace[i] = Z_trace[i] * std::exp(-1.0 / Tau[i]) + 1.0;
            W_cb[i] += alpha_cb * RPE_cb * Z_trace[i];
            cb_bias += W_cb[i] * Z_trace[i] * delta_cb;
        }
    }
    return nll;
}

// [[Rcpp::export]]
double get_nll_hybrid(const std::vector<double>& phi, const std::vector<double>& hyper, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    int N = 100; double lambda_min = hyper[0], lambda_max = hyper[1];
    double a_base = std::exp(phi[0]), t_nd = 1.0 / (1.0 + std::exp(-phi[1])), kappa_v = std::exp(phi[2]); 
    double alpha_ctx = 1.0 / (1.0 + std::exp(-phi[3])), alpha_cb = 1.0 / (1.0 + std::exp(-phi[4])); 
    double lambda_lasso = std::exp(phi[5]), gamma_spectral = std::exp(phi[8]), delta_cb = phi[11];
    std::vector<double> Tau(N);
    for(int i=0; i<N; ++i) Tau[i] = 1.0 / (lambda_min * std::pow(lambda_max / lambda_min, (double)i / (double)(N - 1)));
    double Q_ctx[2] = {0.5, 0.5}; std::vector<double> W_cb(N, 0.0), Z_trace(N, 0.0);
    double cb_bias = 0.0; double nll = 0.0;
    for (int t = 0; t < resp.size(); ++t) {
        int ch = resp[t] - 1; double R = (out[t] == 1) ? 1.0 : 0.0;
        double v0 = kappa_v * std::max(Q_ctx[0], 0.01) - cb_bias;
        double v1 = kappa_v * std::max(Q_ctx[1], 0.01) + cb_bias;
        if(v0 < 0.01) v0 = 0.01; if(v1 < 0.01) v1 = 0.01;
        double p_ch = (ch == 1) ? (std::exp(v1) / (std::exp(v0) + std::exp(v1))) : (std::exp(v0) / (std::exp(v0) + std::exp(v1)));
        if (p_ch < 1e-5) p_ch = 1e-5;
        double v_t = (ch == 1) ? v1 : v0;
        double a_t = a_base * (1.0 + gamma_spectral * std::abs(cb_bias));
        double mu = a_t / v_t; double lambda = a_t * a_t;
        double t_diff = rt[t] - t_nd; if (t_diff < 0.001) t_diff = 0.001;
        double pdf = std::sqrt(lambda / (2.0 * M_PI * std::pow(t_diff, 3))) * std::exp(- (lambda * std::pow(t_diff - mu, 2)) / (2.0 * mu * mu * t_diff));
        if (pdf < 1e-10) pdf = 1e-10;
        nll -= (std::log(p_ch) + std::log(pdf));
        double Q_active = std::max(Q_ctx[ch], 0.01);
        Q_ctx[ch] += alpha_ctx * (R - Q_ctx[ch]);
        double RPE_cb = R - Q_active; cb_bias = 0.0;
        for(int i=0; i<N; ++i) {
            Z_trace[i] = Z_trace[i] * std::exp(-1.0 / Tau[i]) + 1.0;
            W_cb[i] += alpha_cb * RPE_cb * Z_trace[i] - lambda_lasso * signum(W_cb[i]);
            cb_bias += W_cb[i] * Z_trace[i] * delta_cb;
        }
    }
    return nll;
}

// ==========================================
// 2. EXTRACTORS (Returns Matrix [RT_exp, P1])
// ==========================================

// [[Rcpp::export]]
NumericMatrix ext_base(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    double a = std::exp(phi[0]), tnd = 1.0 / (1.0 + std::exp(-phi[1])), kv = std::exp(phi[2]), ac = 1.0 / (1.0 + std::exp(-phi[3]));
    NumericMatrix res(resp.size(), 2); double Q[2] = {0.5, 0.5};
    for (int t = 0; t < resp.size(); ++t) {
        int ch = resp[t] - 1; double R = (out[t] == 1) ? 1.0 : 0.0;
        double v0 = kv * std::max(Q[0], 0.01), v1 = kv * std::max(Q[1], 0.01);
        res(t, 1) = std::exp(v1) / (std::exp(v0) + std::exp(v1));
        double v_t = kv * std::max(Q[ch], 0.01);
        res(t, 0) = tnd + (a / v_t);
        Q[ch] += ac * (R - Q[ch]);
    }
    return res;
}

// [[Rcpp::export]]
NumericMatrix ext_qperturbed(const std::vector<double>& phi, const std::vector<double>& hyper, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    int N = 100; double lmin = hyper[0], lmax = hyper[1];
    double a = std::exp(phi[0]), tnd = 1.0 / (1.0 + std::exp(-phi[1])), kv = std::exp(phi[2]), ac = 1.0 / (1.0 + std::exp(-phi[3])), ab = 1.0 / (1.0 + std::exp(-phi[4])), dcb = phi[9];
    std::vector<double> Tau(N); for(int i=0; i<N; ++i) Tau[i] = 1.0 / (lmin * std::pow(lmax/lmin, (double)i/(N-1)));
    NumericMatrix res(resp.size(), 2); double Q[2] = {0.5, 0.5}; std::vector<double> W(N, 0.0), Z(N, 0.0); double cb = 0.0;
    for (int t = 0; t < resp.size(); ++t) {
        int ch = resp[t] - 1; double R = (out[t] == 1) ? 1.0 : 0.0;
        double v0 = kv * std::max(Q[0], 0.01) - cb; double v1 = kv * std::max(Q[1], 0.01) + cb;
        if(v0 < 0.01) v0 = 0.01; if(v1 < 0.01) v1 = 0.01;
        res(t, 1) = std::exp(v1) / (std::exp(v0) + std::exp(v1));
        res(t, 0) = tnd + (a / ((ch == 1) ? v1 : v0));
        double qa = std::max(Q[ch], 0.01); Q[ch] += ac * (R - Q[ch]);
        cb = 0.0; double rpe = R - qa;
        for(int i=0; i<N; ++i) { Z[i] = Z[i]*std::exp(-1.0/Tau[i]) + 1.0; W[i] += ab * rpe * Z[i]; cb += W[i] * Z[i] * dcb; }
    }
    return res;
}

// [[Rcpp::export]]
NumericMatrix ext_hybrid(const std::vector<double>& phi, const std::vector<double>& hyper, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    int N = 100; double lmin = hyper[0], lmax = hyper[1];
    double a = std::exp(phi[0]), tnd = 1.0 / (1.0 + std::exp(-phi[1])), kv = std::exp(phi[2]), ac = 1.0 / (1.0 + std::exp(-phi[3])), ab = 1.0 / (1.0 + std::exp(-phi[4]));
    double llas = std::exp(phi[5]), gs = std::exp(phi[8]), dcb = phi[11];
    std::vector<double> Tau(N); for(int i=0; i<N; ++i) Tau[i] = 1.0 / (lmin * std::pow(lmax/lmin, (double)i/(N-1)));
    NumericMatrix res(resp.size(), 2); double Q[2] = {0.5, 0.5}; std::vector<double> W(N, 0.0), Z(N, 0.0); double cb = 0.0;
    for (int t = 0; t < resp.size(); ++t) {
        int ch = resp[t] - 1; double R = (out[t] == 1) ? 1.0 : 0.0;
        double v0 = kv * std::max(Q[0], 0.01) - cb; double v1 = kv * std::max(Q[1], 0.01) + cb;
        if(v0 < 0.01) v0 = 0.01; if(v1 < 0.01) v1 = 0.01;
        res(t, 1) = std::exp(v1) / (std::exp(v0) + std::exp(v1));
        double at = a * (1.0 + gs * std::abs(cb));
        res(t, 0) = tnd + (at / ((ch == 1) ? v1 : v0));
        double qa = std::max(Q[ch], 0.01); Q[ch] += ac * (R - Q[ch]);
        cb = 0.0; double rpe = R - qa;
        for(int i=0; i<N; ++i) { Z[i] = Z[i]*std::exp(-1.0/Tau[i]) + 1.0; W[i] += ab * rpe * Z[i] - llas * signum(W[i]); cb += W[i] * Z[i] * dcb; }
    }
    return res;
}
