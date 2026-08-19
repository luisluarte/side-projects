#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

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

double compute_penalized_likelihood(const std::vector<double>& D_t) {
    int T = D_t.size();
    if (T < 2) return 0.0;
    double sum_t = 0.0, sum_D = 0.0, sum_t_sq = 0.0, sum_tD = 0.0;
    for (int t = 0; t < T; ++t) {
        double tt = (double)(t + 1);
        sum_t += tt;
        sum_D += D_t[t];
        sum_t_sq += tt * tt;
        sum_tD += tt * D_t[t];
    }
    double mean_t = sum_t / T;
    double mean_D = sum_D / T;
    double num = 0.0, den = 0.0;
    for (int t = 0; t < T; ++t) {
        double tt = (double)(t + 1);
        num += (tt - mean_t) * (D_t[t] - mean_D);
        den += (tt - mean_t) * (tt - mean_t);
    }
    double m_D = (den == 0.0) ? 0.0 : num / den;
    return sum_D + 1.0 * std::abs(m_D);
}

// Model Evaluators return Penalized Likelihood
double eval_wsls(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    double a = std::exp(phi[0]);
    double t_nd = 1.0 / (1.0 + std::exp(-phi[1]));
    double beta_v = std::exp(phi[2]);
    double p_stay = 1.0 / (1.0 + std::exp(-phi[3])); // prob stay after win
    double p_shift = 1.0 / (1.0 + std::exp(-phi[4])); // prob shift after loss
    
    std::vector<double> D_t;
    int prev_ch = -1; int prev_out = -1;
    for (int t = 0; t < resp.size(); ++t) {
        int ch = resp[t];
        double v_t = 0.0;
        if (t > 0 && prev_ch != -1) {
            bool stay_pred = (prev_out == 1) ? true : false;
            double prob_stay = (prev_out == 1) ? p_stay : (1.0 - p_shift);
            v_t = beta_v * ( (prev_ch == 1) ? (prob_stay - 0.5) : (0.5 - prob_stay) ) * 2.0;
        }
        double safe_v = std::abs(v_t) < 1e-4 ? (v_t >= 0 ? 1e-4 : -1e-4) : v_t;
        double pdf = wiener_pdf(rt[t], ch, safe_v, a, t_nd);
        D_t.push_back(-2.0 * std::log(pdf));
        prev_ch = ch; prev_out = out[t];
    }
    return compute_penalized_likelihood(D_t);
}

double eval_cfmr(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    double a = std::exp(phi[0]);
    double t_nd = 1.0 / (1.0 + std::exp(-phi[1]));
    double beta_v = std::exp(phi[2]);
    double eta_LTP = 1.0 / (1.0 + std::exp(-phi[3]));
    double eta_LTD = 1.0 / (1.0 + std::exp(-phi[4]));
    
    std::vector<double> D_t;
    double Q1 = 0.0, Q2 = 0.0;
    for (int t = 0; t < resp.size(); ++t) {
        int ch = resp[t];
        double v_t = beta_v * (Q1 - Q2);
        double safe_v = std::abs(v_t) < 1e-4 ? (v_t >= 0 ? 1e-4 : -1e-4) : v_t;
        double pdf = wiener_pdf(rt[t], ch, safe_v, a, t_nd);
        D_t.push_back(-2.0 * std::log(pdf));
        
        if (ch == 1) Q1 += (out[t] == 1) ? eta_LTP*(1.0-Q1) : eta_LTD*(-1.0-Q1);
        else         Q2 += (out[t] == 1) ? eta_LTP*(1.0-Q2) : eta_LTD*(-1.0-Q2);
    }
    return compute_penalized_likelihood(D_t);
}

double eval_eccm(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    double a = std::exp(phi[0]);
    double t_nd = 1.0 / (1.0 + std::exp(-phi[1]));
    double beta_v = std::exp(phi[2]);
    double eta_LTP = 1.0 / (1.0 + std::exp(-phi[3]));
    double eta_LTD = 1.0 / (1.0 + std::exp(-phi[4]));
    double w_cb = phi[5]; // No transform needed for real line
    
    int N_MF = 160, N_GC = 1024, N_MLI = 256;
    std::vector<double> W_PF1(N_GC, 0.0), W_PF2(N_GC, 0.0);
    std::vector<double> W_MLI1(N_MLI, 0.0), W_MLI2(N_MLI, 0.0);
    std::vector<std::vector<double>> W_MF_GC(N_GC, std::vector<double>(N_MF, 0.0));
    std::vector<std::vector<double>> W_GC_MLI(N_MLI, std::vector<double>(N_GC, 0.0));
    std::vector<double> mf(N_MF, 0.0);
    
    SimpleRNG rng(42);
    for (int i=0; i<N_GC; ++i) for (int j=0; j<N_MF; ++j) W_MF_GC[i][j] = rng.rnorm() / std::sqrt((double)N_MF);
    for (int i=0; i<N_MLI; ++i) for (int j=0; j<N_GC; ++j) W_GC_MLI[i][j] = rng.rnorm() / std::sqrt((double)N_GC);
    
    std::vector<double> D_t;
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
        
        double v_t = beta_v * ( (Q1_CTX * (1.0 + w_cb * Q1_CB)) - (Q2_CTX * (1.0 + w_cb * Q2_CB)) );
        double safe_v = std::abs(v_t) < 1e-4 ? (v_t >= 0 ? 1e-4 : -1e-4) : v_t;
        double pdf = wiener_pdf(rt[t], ch, safe_v, a, t_nd);
        D_t.push_back(-2.0 * std::log(pdf));
        
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
        for(int k=78; k>=0; --k) { mf[k+1] = mf[k]; mf[80+k+1] = mf[80+k]; }
        mf[0] = (ch == 1) ? 1.0 : -1.0; mf[80] = R_raw;
    }
    return compute_penalized_likelihood(D_t);
}

// [[Rcpp::export]]
List run_hierarchical_mcmc(int model_idx, const List& resp_list, const List& out_list, const List& rt_list, int iters, int warmup) {
    int N_subjs = resp_list.size();
    int N_params = (model_idx == 3) ? 6 : 5; // 1: WSLS, 2: CFMR, 3: ECCM
    
    std::vector<double> mu(N_params, 0.0);
    std::vector<double> sigma(N_params, 1.0);
    std::vector<std::vector<double>> theta(N_subjs, std::vector<double>(N_params, 0.0));
    
    NumericMatrix mu_trace(iters, N_params);
    NumericMatrix sigma_trace(iters, N_params);
    NumericVector deviance_trace(iters);
    
    SimpleRNG rng(12345);
    
    for (int iter = 0; iter < iters + warmup; ++iter) {
        double total_dev = 0.0;
        
        // Sample Subject Parameters
        for (int s = 0; s < N_subjs; ++s) {
            IntegerVector resp = resp_list[s];
            IntegerVector out = out_list[s];
            NumericVector rt = rt_list[s];
            
            double curr_pen_lik = 0.0;
            if (model_idx == 1) curr_pen_lik = eval_wsls(theta[s], resp, out, rt);
            else if (model_idx == 2) curr_pen_lik = eval_cfmr(theta[s], resp, out, rt);
            else curr_pen_lik = eval_eccm(theta[s], resp, out, rt);
            
            for (int p = 0; p < N_params; ++p) {
                double prop_theta = theta[s][p] + rng.rnorm() * 0.1;
                std::vector<double> prop_vec = theta[s]; prop_vec[p] = prop_theta;
                
                double prop_pen_lik = 0.0;
                if (model_idx == 1) prop_pen_lik = eval_wsls(prop_vec, resp, out, rt);
                else if (model_idx == 2) prop_pen_lik = eval_cfmr(prop_vec, resp, out, rt);
                else prop_pen_lik = eval_eccm(prop_vec, resp, out, rt);
                
                double prior_curr = -0.5 * std::pow((theta[s][p] - mu[p]) / sigma[p], 2) - std::log(sigma[p]);
                double prior_prop = -0.5 * std::pow((prop_theta - mu[p]) / sigma[p], 2) - std::log(sigma[p]);
                
                double log_accept = (-0.5 * prop_pen_lik + prior_prop) - (-0.5 * curr_pen_lik + prior_curr);
                
                if (std::log(std::max(1e-12, rng.runif())) < log_accept) {
                    theta[s][p] = prop_theta;
                    curr_pen_lik = prop_pen_lik;
                }
            }
            total_dev += curr_pen_lik;
        }
        
        // Sample Group Parameters
        for (int p = 0; p < N_params; ++p) {
            double prop_mu = mu[p] + rng.rnorm() * 0.1;
            double log_prior_curr = -0.5 * std::pow(mu[p] / 10.0, 2);
            double log_prior_prop = -0.5 * std::pow(prop_mu / 10.0, 2);
            
            double log_lik_curr = 0.0, log_lik_prop = 0.0;
            for (int s = 0; s < N_subjs; ++s) {
                log_lik_curr += -0.5 * std::pow((theta[s][p] - mu[p]) / sigma[p], 2);
                log_lik_prop += -0.5 * std::pow((theta[s][p] - prop_mu) / sigma[p], 2);
            }
            if (std::log(std::max(1e-12, rng.runif())) < (log_lik_prop + log_prior_prop) - (log_lik_curr + log_prior_curr)) {
                mu[p] = prop_mu;
            }
            
            double prop_sigma = std::max(0.01, sigma[p] + rng.rnorm() * 0.1);
            log_lik_curr = 0.0; log_lik_prop = 0.0;
            for (int s = 0; s < N_subjs; ++s) {
                log_lik_curr += -0.5 * std::pow((theta[s][p] - mu[p]) / sigma[p], 2) - std::log(sigma[p]);
                log_lik_prop += -0.5 * std::pow((theta[s][p] - mu[p]) / prop_sigma, 2) - std::log(prop_sigma);
            }
            if (std::log(std::max(1e-12, rng.runif())) < log_lik_prop - log_lik_curr) {
                sigma[p] = prop_sigma;
            }
        }
        
        if (iter >= warmup) {
            int i = iter - warmup;
            for (int p = 0; p < N_params; ++p) {
                mu_trace(i, p) = mu[p];
                sigma_trace(i, p) = sigma[p];
            }
            deviance_trace[i] = total_dev;
        }
    }
    
    return List::create(
        Named("mu") = mu_trace,
        Named("sigma") = sigma_trace,
        Named("deviance") = deviance_trace
    );
}
