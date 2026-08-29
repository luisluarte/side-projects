#include <Rcpp.h>
#include <cmath>
#include <vector>
using namespace Rcpp;

inline double signum(double x) { return (x > 0) ? 1.0 : ((x < 0) ? -1.0 : 0.0); }

double wiener_pdf(double t, int resp, double v, double a, double t0, double w) {
    if (t <= t0) return 1e-10;
    double tt = t - t0;
    double k = 0.0;
    double p = 0.0;
    double err = 1e-10;
    if (resp == 2) { v = -v; w = 1.0 - w; }
    
    while (true) {
        k++;
        double term = k * std::sin(k * M_PI * w) * std::exp(-0.5 * (v * v * tt) - 0.5 * (k * k * M_PI * M_PI * tt / (a * a))) * std::exp(v * a * w) * M_PI / (a * a);
        p += term;
        if (std::abs(term) < err) break;
        if (k > 50) break;
    }
    return p > 1e-10 ? p : 1e-10;
}

// 1. BASELINE Q-LEARNING DDM (Wiener)
// [[Rcpp::export]]
double get_nll_ddm(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    double a = std::exp(phi[0]), t_nd = 1.0 / (1.0 + std::exp(-phi[1])), beta_v = std::exp(phi[2]);
    double alpha = 1.0 / (1.0 + std::exp(-phi[3])), alpha_c = 1.0 / (1.0 + std::exp(-phi[4]));
    double Q[2] = {0.5, 0.5}; double nll = 0.0;
    for (int t=0; t<resp.size(); ++t) {
        int ch = resp[t] - 1; double R = (out[t] == 1) ? 1.0 : 0.0;
        double v = beta_v * (Q[1] - Q[0]);
        double safe_v = std::abs(v) < 1e-4 ? (v >= 0 ? 1e-4 : -1e-4) : v;
        int w_choice = (resp[t] == 2) ? 1 : 2;
        double pdf = wiener_pdf(rt[t], w_choice, safe_v, a, t_nd, 0.5);
        nll -= std::log(pdf);
        Q[ch] += alpha * (R - Q[ch]); Q[1-ch] += alpha_c * ((1.0 - R) - Q[1-ch]);
    }
    return nll;
}

// [[Rcpp::export]]
NumericMatrix ext_ddm(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    double a = std::exp(phi[0]), t_nd = 1.0 / (1.0 + std::exp(-phi[1])), beta_v = std::exp(phi[2]), alpha = 1.0 / (1.0 + std::exp(-phi[3])), alpha_c = 1.0 / (1.0 + std::exp(-phi[4]));
    NumericMatrix res(resp.size(), 2); double Q[2] = {0.5, 0.5};
    for (int t=0; t<resp.size(); ++t) {
        int ch = resp[t] - 1; double R = (out[t] == 1) ? 1.0 : 0.0;
        double v = beta_v * (Q[1] - Q[0]);
        double exp_rt = t_nd + (a / (2.0 * v)) * std::tanh(a * v / 2.0); if(std::isnan(exp_rt) || std::abs(v) < 1e-4) exp_rt = t_nd + (a*a)/4.0;
        double p1 = 1.0 / (1.0 + std::exp(-a * v));
        res(t, 0) = exp_rt; res(t, 1) = p1;
        Q[ch] += alpha * (R - Q[ch]); Q[1-ch] += alpha_c * ((1.0 - R) - Q[1-ch]);
    }
    return res;
}

// 2. CONTEXT-GATED DDM (Wiener)
// [[Rcpp::export]]
double get_nll_ddm_ctx(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    double a = std::exp(phi[0]), t_nd = 1.0 / (1.0 + std::exp(-phi[1])), beta_v = std::exp(phi[2]);
    double alpha = 1.0 / (1.0 + std::exp(-phi[3])), alpha_c = 1.0 / (1.0 + std::exp(-phi[4]));
    double theta_ctx = std::exp(phi[5]), beta_a = std::exp(phi[6]);
    double Q[2] = {0.5, 0.5}; double nll = 0.0;
    for (int t=0; t<resp.size(); ++t) {
        int ch = resp[t] - 1; double R = (out[t] == 1) ? 1.0 : 0.0;
        double v = beta_v * (Q[1] - Q[0]); double safe_v = std::abs(v) < 1e-4 ? (v >= 0 ? 1e-4 : -1e-4) : v;
        double prev_rt = (t == 0) ? 0.5 : rt[t-1];
        double w_bias = 0.5 + 0.45 * std::tanh(theta_ctx * prev_rt);
        double a_t = a + beta_a * std::tanh(theta_ctx * prev_rt);
        if (a_t < 0.01) a_t = 0.01; if (a_t > 10.0) a_t = 10.0;
        int w_choice = (resp[t] == 2) ? 1 : 2;
        double pdf = wiener_pdf(rt[t], w_choice, safe_v, a_t, t_nd, w_bias);
        nll -= std::log(pdf);
        Q[ch] += alpha * (R - Q[ch]); Q[1-ch] += alpha_c * ((1.0 - R) - Q[1-ch]);
    }
    return nll;
}

// [[Rcpp::export]]
NumericMatrix ext_ddm_ctx(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    double a = std::exp(phi[0]), t_nd = 1.0 / (1.0 + std::exp(-phi[1])), beta_v = std::exp(phi[2]);
    double alpha = 1.0 / (1.0 + std::exp(-phi[3])), alpha_c = 1.0 / (1.0 + std::exp(-phi[4]));
    double theta_ctx = std::exp(phi[5]), beta_a = std::exp(phi[6]);
    NumericMatrix res(resp.size(), 2); double Q[2] = {0.5, 0.5};
    for (int t=0; t<resp.size(); ++t) {
        int ch = resp[t] - 1; double R = (out[t] == 1) ? 1.0 : 0.0;
        double v = beta_v * (Q[1] - Q[0]);
        double prev_rt = (t == 0) ? 0.5 : rt[t-1];
        double w_bias = 0.5 + 0.45 * std::tanh(theta_ctx * prev_rt);
        double a_t = a + beta_a * std::tanh(theta_ctx * prev_rt);
        if (a_t < 0.01) a_t = 0.01; if (a_t > 10.0) a_t = 10.0;
        double exp_rt = t_nd + (a_t / (2.0 * v)) * std::tanh(a_t * v / 2.0); if(std::isnan(exp_rt) || std::abs(v) < 1e-4) exp_rt = t_nd + (a_t*a_t)/4.0;
        double p1 = 1.0 / (1.0 + std::exp(-a_t * v));
        res(t, 0) = exp_rt; res(t, 1) = p1;
        Q[ch] += alpha * (R - Q[ch]); Q[1-ch] += alpha_c * ((1.0 - R) - Q[1-ch]);
    }
    return res;
}

// 3. BASELINE WALD
// [[Rcpp::export]]
double get_nll_base(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    double a = std::exp(phi[0]), t_nd = 1.0 / (1.0 + std::exp(-phi[1])), kv = std::exp(phi[2]), ac = 1.0 / (1.0 + std::exp(-phi[3])); 
    double nll = 0.0; double Q[2] = {0.5, 0.5};
    for (int t = 0; t < resp.size(); ++t) {
        int ch = resp[t] - 1; double R = (out[t] == 1) ? 1.0 : 0.0;
        double v0 = kv * std::max(Q[0], 0.01), v1 = kv * std::max(Q[1], 0.01);
        double p_ch = (ch == 1) ? (std::exp(v1) / (std::exp(v0) + std::exp(v1))) : (std::exp(v0) / (std::exp(v0) + std::exp(v1)));
        if (p_ch < 1e-5) p_ch = 1e-5;
        double vt = kv * std::max(Q[ch], 0.01); double mu = a / vt; double lam = a * a;
        double td = rt[t] - t_nd; if (td < 0.001) td = 0.001;
        double pdf = std::sqrt(lam / (2.0 * M_PI * std::pow(td, 3))) * std::exp(- (lam * std::pow(td - mu, 2)) / (2.0 * mu * mu * td));
        if (pdf < 1e-10) pdf = 1e-10;
        nll -= (std::log(p_ch) + std::log(pdf)); Q[ch] += ac * (R - Q[ch]);
    }
    return nll;
}

// [[Rcpp::export]]
NumericMatrix ext_base(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    double a = std::exp(phi[0]), tnd = 1.0 / (1.0 + std::exp(-phi[1])), kv = std::exp(phi[2]), ac = 1.0 / (1.0 + std::exp(-phi[3]));
    NumericMatrix res(resp.size(), 2); double Q[2] = {0.5, 0.5};
    for (int t = 0; t < resp.size(); ++t) {
        int ch = resp[t] - 1; double R = (out[t] == 1) ? 1.0 : 0.0;
        double v0 = kv * std::max(Q[0], 0.01), v1 = kv * std::max(Q[1], 0.01);
        res(t, 1) = std::exp(v1) / (std::exp(v0) + std::exp(v1));
        double v_t = kv * std::max(Q[ch], 0.01); res(t, 0) = tnd + (a / v_t); Q[ch] += ac * (R - Q[ch]);
    }
    return res;
}

// 4. ECCM SYMPLECTIC RESERVOIR (Wald) - 6 params
// [[Rcpp::export]]
double get_nll_eccm(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    double a = std::exp(phi[0]), t_nd = 1.0 / (1.0 + std::exp(-phi[1])), kv = std::exp(phi[2]);
    double ac = 1.0 / (1.0 + std::exp(-phi[3])), cb_lr = 1.0 / (1.0 + std::exp(-phi[4])), cb_w = phi[5];
    int N_GC = 256; std::vector<double> W(N_GC, 0.0), GC(N_GC, 0.0);
    uint32_t state = 42; auto runif = [&]() { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return (state % 1000) / 1000.0; };
    double nll = 0.0; double Q[2] = {0.5, 0.5};
    for (int t = 0; t < resp.size(); ++t) {
        int ch = resp[t] - 1; double R = (out[t] == 1) ? 1.0 : 0.0;
        for(int i=0; i<N_GC; ++i) GC[i] = std::sin(rt[t] + runif());
        double cb = 0.0; for(int i=0; i<N_GC; ++i) cb += W[i] * GC[i] * cb_w;
        double v0 = kv * std::max(Q[0], 0.01) - cb, v1 = kv * std::max(Q[1], 0.01) + cb;
        if(v0 < 0.01) v0 = 0.01; if(v1 < 0.01) v1 = 0.01;
        double p_ch = (ch == 1) ? (std::exp(v1) / (std::exp(v0) + std::exp(v1))) : (std::exp(v0) / (std::exp(v0) + std::exp(v1)));
        if (p_ch < 1e-5) p_ch = 1e-5;
        double vt = (ch == 1) ? v1 : v0; double mu = a / vt; double lam = a * a;
        double td = rt[t] - t_nd; if (td < 0.001) td = 0.001;
        double pdf = std::sqrt(lam / (2.0 * M_PI * std::pow(td, 3))) * std::exp(- (lam * std::pow(td - mu, 2)) / (2.0 * mu * mu * td));
        if (pdf < 1e-10) pdf = 1e-10;
        nll -= (std::log(p_ch) + std::log(pdf)); Q[ch] += ac * (R - Q[ch]);
        double rpe = R - std::max(Q[ch], 0.01);
        for(int i=0; i<N_GC; ++i) W[i] += cb_lr * rpe * GC[i];
    }
    return nll;
}

// [[Rcpp::export]]
NumericMatrix ext_eccm(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    double a = std::exp(phi[0]), t_nd = 1.0 / (1.0 + std::exp(-phi[1])), kv = std::exp(phi[2]);
    double ac = 1.0 / (1.0 + std::exp(-phi[3])), cb_lr = 1.0 / (1.0 + std::exp(-phi[4])), cb_w = phi[5];
    int N_GC = 256; std::vector<double> W(N_GC, 0.0), GC(N_GC, 0.0);
    uint32_t state = 42; auto runif = [&]() { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return (state % 1000) / 1000.0; };
    NumericMatrix res(resp.size(), 2); double Q[2] = {0.5, 0.5};
    for (int t = 0; t < resp.size(); ++t) {
        int ch = resp[t] - 1; double R = (out[t] == 1) ? 1.0 : 0.0;
        for(int i=0; i<N_GC; ++i) GC[i] = std::sin(rt[t] + runif());
        double cb = 0.0; for(int i=0; i<N_GC; ++i) cb += W[i] * GC[i] * cb_w;
        double v0 = kv * std::max(Q[0], 0.01) - cb, v1 = kv * std::max(Q[1], 0.01) + cb;
        if(v0 < 0.01) v0 = 0.01; if(v1 < 0.01) v1 = 0.01;
        res(t, 1) = std::exp(v1) / (std::exp(v0) + std::exp(v1));
        double vt = (ch == 1) ? v1 : v0; res(t, 0) = t_nd + (a / vt); Q[ch] += ac * (R - Q[ch]);
        double rpe = R - std::max(Q[ch], 0.01); for(int i=0; i<N_GC; ++i) W[i] += cb_lr * rpe * GC[i];
    }
    return res;
}

// 5. Q-PERTURBED WALD
// [[Rcpp::export]]
double get_nll_qperturbed(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    int N = 100; double lmin = 0.01, lmax = 1.0;
    double a = std::exp(phi[0]), t_nd = 1.0 / (1.0 + std::exp(-phi[1])), kv = std::exp(phi[2]); 
    double ac = 1.0 / (1.0 + std::exp(-phi[3])), ab = 1.0 / (1.0 + std::exp(-phi[4])); double dcb = phi[9];
    std::vector<double> Tau(N); for(int i=0; i<N; ++i) Tau[i] = 1.0 / (lmin * std::pow(lmax/lmin, (double)i/(N-1)));
    double Q[2] = {0.5, 0.5}; std::vector<double> W(N, 0.0), Z(N, 0.0); double cb = 0.0; double nll = 0.0;
    for (int t = 0; t < resp.size(); ++t) {
        int ch = resp[t] - 1; double R = (out[t] == 1) ? 1.0 : 0.0;
        double v0 = kv * std::max(Q[0], 0.01) - cb, v1 = kv * std::max(Q[1], 0.01) + cb;
        if(v0 < 0.01) v0 = 0.01; if(v1 < 0.01) v1 = 0.01;
        double p_ch = (ch == 1) ? (std::exp(v1) / (std::exp(v0) + std::exp(v1))) : (std::exp(v0) / (std::exp(v0) + std::exp(v1)));
        if (p_ch < 1e-5) p_ch = 1e-5;
        double vt = (ch == 1) ? v1 : v0; double mu = a / vt; double lam = a * a;
        double td = rt[t] - t_nd; if (td < 0.001) td = 0.001;
        double pdf = std::sqrt(lam / (2.0 * M_PI * std::pow(td, 3))) * std::exp(- (lam * std::pow(td - mu, 2)) / (2.0 * mu * mu * td));
        if (pdf < 1e-10) pdf = 1e-10; nll -= (std::log(p_ch) + std::log(pdf));
        double qa = std::max(Q[ch], 0.01); Q[ch] += ac * (R - Q[ch]);
        cb = 0.0; double rpe = R - qa;
        for(int i=0; i<N; ++i) { Z[i] = Z[i]*std::exp(-1.0/Tau[i]) + 1.0; W[i] += ab * rpe * Z[i]; cb += W[i] * Z[i] * dcb; }
    }
    return nll;
}

// [[Rcpp::export]]
NumericMatrix ext_qperturbed(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    int N = 100; double lmin = 0.01, lmax = 1.0;
    double a = std::exp(phi[0]), tnd = 1.0 / (1.0 + std::exp(-phi[1])), kv = std::exp(phi[2]); 
    double ac = 1.0 / (1.0 + std::exp(-phi[3])), ab = 1.0 / (1.0 + std::exp(-phi[4])), dcb = phi[9];
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

// 6. TERMINAL HYBRID
// [[Rcpp::export]]
double get_nll_hybrid(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    int N = 100; double lmin = 0.01, lmax = 1.0;
    double a = std::exp(phi[0]), t_nd = 1.0 / (1.0 + std::exp(-phi[1])), kv = std::exp(phi[2]); 
    double ac = 1.0 / (1.0 + std::exp(-phi[3])), ab = 1.0 / (1.0 + std::exp(-phi[4])); 
    double llas = std::exp(phi[5]), gs = std::exp(phi[8]), dcb = phi[11];
    std::vector<double> Tau(N); for(int i=0; i<N; ++i) Tau[i] = 1.0 / (lmin * std::pow(lmax/lmin, (double)i/(N-1)));
    double Q[2] = {0.5, 0.5}; std::vector<double> W(N, 0.0), Z(N, 0.0); double cb = 0.0; double nll = 0.0;
    for (int t = 0; t < resp.size(); ++t) {
        int ch = resp[t] - 1; double R = (out[t] == 1) ? 1.0 : 0.0;
        double v0 = kv * std::max(Q[0], 0.01) - cb, v1 = kv * std::max(Q[1], 0.01) + cb;
        if(v0 < 0.01) v0 = 0.01; if(v1 < 0.01) v1 = 0.01;
        double p_ch = (ch == 1) ? (std::exp(v1) / (std::exp(v0) + std::exp(v1))) : (std::exp(v0) / (std::exp(v0) + std::exp(v1)));
        if (p_ch < 1e-5) p_ch = 1e-5;
        double vt = (ch == 1) ? v1 : v0; double at = a * (1.0 + gs * std::abs(cb));
        double mu = at / vt; double lam = at * at;
        double td = rt[t] - t_nd; if (td < 0.001) td = 0.001;
        double pdf = std::sqrt(lam / (2.0 * M_PI * std::pow(td, 3))) * std::exp(- (lam * std::pow(td - mu, 2)) / (2.0 * mu * mu * td));
        if (pdf < 1e-10) pdf = 1e-10; nll -= (std::log(p_ch) + std::log(pdf));
        double qa = std::max(Q[ch], 0.01); Q[ch] += ac * (R - Q[ch]);
        cb = 0.0; double rpe = R - qa;
        for(int i=0; i<N; ++i) { Z[i] = Z[i]*std::exp(-1.0/Tau[i]) + 1.0; W[i] += ab * rpe * Z[i] - llas * signum(W[i]); cb += W[i] * Z[i] * dcb; }
    }
    return nll;
}

// [[Rcpp::export]]
NumericMatrix ext_hybrid(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    int N = 100; double lmin = 0.01, lmax = 1.0;
    double a = std::exp(phi[0]), tnd = 1.0 / (1.0 + std::exp(-phi[1])), kv = std::exp(phi[2]); 
    double ac = 1.0 / (1.0 + std::exp(-phi[3])), ab = 1.0 / (1.0 + std::exp(-phi[4])); 
    double llas = std::exp(phi[5]), gs = std::exp(phi[8]), dcb = phi[11];
    std::vector<double> Tau(N); for(int i=0; i<N; ++i) Tau[i] = 1.0 / (lmin * std::pow(lmax/lmin, (double)i/(N-1)));
    NumericMatrix res(resp.size(), 2); double Q[2] = {0.5, 0.5}; std::vector<double> W(N, 0.0), Z(N, 0.0); double cb = 0.0;
    for (int t = 0; t < resp.size(); ++t) {
        int ch = resp[t] - 1; double R = (out[t] == 1) ? 1.0 : 0.0;
        double v0 = kv * std::max(Q[0], 0.01) - cb; double v1 = kv * std::max(Q[1], 0.01) + cb;
        if(v0 < 0.01) v0 = 0.01; if(v1 < 0.01) v1 = 0.01;
        res(t, 1) = std::exp(v1) / (std::exp(v0) + std::exp(v1));
        double at = a * (1.0 + gs * std::abs(cb)); res(t, 0) = tnd + (at / ((ch == 1) ? v1 : v0));
        double qa = std::max(Q[ch], 0.01); Q[ch] += ac * (R - Q[ch]);
        cb = 0.0; double rpe = R - qa;
        for(int i=0; i<N; ++i) { Z[i] = Z[i]*std::exp(-1.0/Tau[i]) + 1.0; W[i] += ab * rpe * Z[i] - llas * signum(W[i]); cb += W[i] * Z[i] * dcb; }
    }
    return res;
}
