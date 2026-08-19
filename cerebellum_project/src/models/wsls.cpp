#include <Rcpp.h>
#include "shared_utils.h"

using namespace Rcpp;

// [[Rcpp::export]]
double eval_wsls(const std::vector<double>& phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    double a = std::exp(phi[0]), t_nd = 1.0 / (1.0 + std::exp(-phi[1])), beta_v = std::exp(phi[2]);
    std::vector<double> D_vec;
    int last_ch = -1, last_out = -1;
    for (int t=0; t<resp.size(); ++t) {
        double v = 0.0;
        if (last_ch != -1) {
            int pred_ch = (last_out == 1) ? last_ch : (last_ch == 1 ? 2 : 1);
            v = (pred_ch == 1) ? beta_v : -beta_v;
        }
        double safe_v = std::abs(v) < 1e-4 ? (v >= 0 ? 1e-4 : -1e-4) : v;
        D_vec.push_back(-2.0 * std::log(wiener_pdf(rt[t], resp[t], safe_v, a, t_nd)));
        last_ch = resp[t]; last_out = out[t];
    }
    return calc_pen_ll(D_vec);
}
