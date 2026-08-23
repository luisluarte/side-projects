#include <Rcpp.h>
#include "shared_utils.h"

using namespace Rcpp;

// [[Rcpp::export]]
inline double eval_ql_ddm(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    double a = std::exp(phi[0]), t_nd = 1.0 / (1.0 + std::exp(-phi[1])), beta_v = std::exp(phi[2]);
    double alpha = 1.0 / (1.0 + std::exp(-phi[3]));
    double alpha_c = 1.0 / (1.0 + std::exp(-phi[4]));
    
    std::vector<double> D_vec;
    double Q[2] = {0.0, 0.0};
    
    for (int t=0; t<resp.size(); ++t) {
        int ch = resp[t] - 1; // 0 or 1
        double R = (out[t] == 1) ? 1.0 : 0.0;
        
        double v = beta_v * (Q[0] - Q[1]);
        double safe_v = std::abs(v) < 1e-4 ? (v >= 0 ? 1e-4 : -1e-4) : v;
        D_vec.push_back(-2.0 * std::log(wiener_pdf(rt[t], resp[t], safe_v, a, t_nd)));
        
        Q[ch] = Q[ch] + alpha * (R - Q[ch]);
        int unch = (ch == 1) ? 0 : 1;
        Q[unch] = Q[unch] + alpha_c * ((1.0 - R) - Q[unch]);
    }
    return calc_pen_ll(D_vec);
}

// Extract Pointwise LL for WSLS and QL
// [[Rcpp::export]]
NumericMatrix extract_ll_wsls_ql(const NumericMatrix& chain_wsls, const NumericMatrix& chain_ql, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    int iters = chain_wsls.nrow();
    int T = resp.size();
    NumericMatrix ll_wsls(iters, T);
    
    for (int iter=0; iter<iters; ++iter) {
        double a = std::exp(chain_wsls(iter, 0)), t_nd = 1.0 / (1.0 + std::exp(-chain_wsls(iter, 1))), beta_v = std::exp(chain_wsls(iter, 2));
        int last_ch = -1, last_out = -1;
        for (int t=0; t<T; ++t) {
            double v = 0.0;
            if (last_ch != -1) {
                int pred_ch = (last_out == 1) ? last_ch : (last_ch == 1 ? 2 : 1);
                v = (pred_ch == 1) ? beta_v : -beta_v;
            }
            double safe_v = std::abs(v) < 1e-4 ? (v >= 0 ? 1e-4 : -1e-4) : v;
            ll_wsls(iter, t) = std::log(wiener_pdf(rt[t], resp[t], safe_v, a, t_nd));
            last_ch = resp[t]; last_out = out[t];
        }
    }
    return ll_wsls;
}

// [[Rcpp::export]]
NumericMatrix extract_ll_ql(const NumericMatrix& chain_ql, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    int iters = chain_ql.nrow();
    int T = resp.size();
    NumericMatrix ll_ql(iters, T);
    
    for (int iter=0; iter<iters; ++iter) {
        double a = std::exp(chain_ql(iter, 0)), t_nd = 1.0 / (1.0 + std::exp(-chain_ql(iter, 1))), beta_v = std::exp(chain_ql(iter, 2));
        double alpha = 1.0 / (1.0 + std::exp(-chain_ql(iter, 3)));
        double alpha_c = 1.0 / (1.0 + std::exp(-chain_ql(iter, 4)));
        
        double Q[2] = {0.0, 0.0};
        for (int t=0; t<T; ++t) {
            int ch = resp[t] - 1; 
            double R = (out[t] == 1) ? 1.0 : 0.0;
            double v = beta_v * (Q[0] - Q[1]);
            double safe_v = std::abs(v) < 1e-4 ? (v >= 0 ? 1e-4 : -1e-4) : v;
            ll_ql(iter, t) = std::log(wiener_pdf(rt[t], resp[t], safe_v, a, t_nd));
            
            Q[ch] = Q[ch] + alpha * (R - Q[ch]);
            int unch = (ch == 1) ? 0 : 1;
            Q[unch] = Q[unch] + alpha_c * ((1.0 - R) - Q[unch]);
        }
    }
    return ll_ql;
}
