#ifndef SHARED_UTILS_H
#define SHARED_UTILS_H

#include <vector>
#include <cmath>
#include <algorithm>

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

inline double calc_pen_ll(const std::vector<double>& D_vec) {
    int N = D_vec.size();
    if (N == 0) return 1e9;
    double mean_t = (N + 1) / 2.0;
    double m = 0.0; double ss_t = 0.0; double mean_D = 0.0;
    for (int t=0; t<N; ++t) mean_D += D_vec[t]; mean_D /= N;
    for (int t=0; t<N; ++t) {
        m += (t + 1 - mean_t) * (D_vec[t] - mean_D);
        ss_t += (t + 1 - mean_t) * (t + 1 - mean_t);
    }
    m /= ss_t;
    double total_D = 0.0;
    for (int t=0; t<N; ++t) total_D += D_vec[t];
    return total_D + std::abs(m);
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

#endif
