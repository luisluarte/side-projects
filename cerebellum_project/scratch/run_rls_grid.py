import os
import subprocess

r_script_content = """
library(cmaes)
library(Rcpp)
library(parallel)

# Load data
source("src/R/utils.R")
dat_all <- load_and_preprocess_data("data/raw/trials.csv")
participants <- unique(dat_all$participant_id)
set.seed(42)
sample_participants <- sample(participants, 10)

# We use Rcpp to define the exact RLS batch function for blistering speed
cpp_code <- "
#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace Rcpp;

inline double wiener_pdf(double t_raw, int choice, double v, double a, double t_nd, double eps = 1e-9) {
  double t = t_raw - t_nd;
  if (t <= 0.001) return 1e-12;
  double sign = (choice == 1) ? 1.0 : -1.0;
  double w = 0.50; 
  double x0 = (choice == 1) ? (1.0 - w) : w;
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
  double pdf_val = (drift_term / (a * a)) * sum;
  return std::max(1e-12, pdf_val);
}

class SimpleRNG {
    uint32_t state;
public:
    SimpleRNG(uint32_t seed) : state(seed) {}
    uint32_t next() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    }
    double runif() { return (next() % 1000000) / 1000000.0; }
    double rnorm() {
        double u1 = std::max(1e-6, runif());
        double u2 = runif();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
    }
};

// [[Rcpp::export]]
double evaluate_rls_cmaes(
    const NumericVector& phi_15d,
    const IntegerVector& resp_R,
    const IntegerVector& out_R,
    const NumericVector& rt_R,
    const NumericVector& ttp_R,
    const IntegerVector& subj_idx_R,
    int N_comp
) {
  int N_t = resp_R.size();
  
  double b_v         = phi_15d[0];
  double a_0         = phi_15d[1];
  double t_nd        = phi_15d[2];
  double k_mod       = phi_15d[3]; 
  double mu_beta     = phi_15d[4];
  double sigma_beta  = phi_15d[5];
  double lambda_d    = phi_15d[6];
  double mu_tau      = phi_15d[7];
  double sigma_tau   = phi_15d[8];
  double rho_base    = phi_15d[9];
  double gamma_rls   = phi_15d[10]; // Forgetting factor [0.5, 1.0]
  double reg_lambda  = phi_15d[11]; // Ridge prior variance
  
  int N_GC = 100;
  int N_MF = 10;
  
  std::vector<int> mf_c(N_MF);
  std::vector<double> mf_beta(N_MF);
  std::vector<int> mf_d(N_MF);
  
  SimpleRNG rng(42);
  for(int j = 0; j < N_MF; ++j) {
      mf_c[j] = rng.next() % 6; 
      double beta_raw = mu_beta + sigma_beta * rng.rnorm();
      mf_beta[j] = std::exp(beta_raw);
      double d_raw = lambda_d + std::sqrt(lambda_d) * rng.rnorm();
      mf_d[j] = std::max(0, std::min(10, (int)std::round(d_raw)));
  }

  std::vector<std::vector<int>> gc_mossy_map(N_GC, std::vector<int>(4));
  std::vector<std::vector<double>> gc_mossy_weights(N_GC, std::vector<double>(4, 0.0));
  for (int i = 0; i < N_GC; ++i) {
    for (int k = 0; k < 4; ++k) {
      gc_mossy_map[i][k] = rng.next() % N_MF;
      gc_mossy_weights[i][k] = (rng.rnorm() > 0) ? 1.0 : -1.0; 
    }
  }
  
  // Phase 2: Compression Matrix Initialization
  std::vector<std::vector<double>> W_comp(N_comp, std::vector<double>(N_GC, 0.0));
  double comp_scale = 1.0 / std::sqrt((double)N_GC);
  for (int j = 0; j < N_comp; ++j) {
      for (int i = 0; i < N_GC; ++i) {
          W_comp[j][i] = rng.rnorm() * comp_scale;
      }
  }

  std::vector<double> tau_vec(N_GC, 1.0);
  for (int i = 0; i < N_GC; ++i) {
      double tau_raw = mu_tau + sigma_tau * rng.rnorm();
      tau_vec[i] = std::exp(tau_raw);
  }

  std::vector<double> z_GC_curr(N_GC, 0.0);
  std::vector<double> z_GC_prev(N_GC, 0.0);
  std::vector<std::vector<double>> state_hist(15, std::vector<double>(6, 0.0));
  
  // RLS State Matrices
  std::vector<std::vector<double>> P1(N_comp, std::vector<double>(N_comp, 0.0));
  std::vector<std::vector<double>> P2(N_comp, std::vector<double>(N_comp, 0.0));
  std::vector<double> w1(N_comp, 0.0);
  std::vector<double> w2(N_comp, 0.0);
  std::vector<double> z_comp(N_comp, 0.0);
  
  auto reset_rls = [&]() {
      for(int r=0; r<N_comp; ++r) {
          w1[r] = 0.0;
          w2[r] = 0.0;
          for(int c=0; c<N_comp; ++c) {
              P1[r][c] = (r == c) ? (1.0 / reg_lambda) : 0.0;
              P2[r][c] = (r == c) ? (1.0 / reg_lambda) : 0.0;
          }
      }
  };
  
  double total_nll = 0.0;
  double rpe_abs_prev = 0.0;
  
  for (int t = 0; t < N_t; ++t) {
    if (t == 0 || subj_idx_R[t] != subj_idx_R[t - 1]) {
      std::fill(z_GC_prev.begin(), z_GC_prev.end(), 0.0);
      std::fill(z_GC_curr.begin(), z_GC_curr.end(), 0.0);
      for (int d = 0; d < 15; ++d) std::fill(state_hist[d].begin(), state_hist[d].end(), 0.0);
      rpe_abs_prev = 0.0;
      reset_rls();
    }

    int ch = resp_R[t];
    int out = out_R[t];
    
    int prev_ch  = (t > 0 && subj_idx_R[t] == subj_idx_R[t - 1]) ? resp_R[t - 1] : 1;
    int prev_out = (t > 0 && subj_idx_R[t] == subj_idx_R[t - 1]) ? out_R[t - 1] : 1;
    double prev_rt = (t > 0 && subj_idx_R[t] == subj_idx_R[t - 1]) ? rt_R[t - 1] : 0.75;
    double delta_t_val = (t == 0 || (t > 0 && subj_idx_R[t] != subj_idx_R[t - 1])) ? 1.5 : std::max(0.1, (double)(ttp_R[t] - ttp_R[t - 1]));

    double r_in_1_prev = (prev_ch == 1) ? prev_out : ((prev_out == 1) ? 0.0 : 1.0);
    double r_in_2_prev = (prev_ch == 2) ? prev_out : ((prev_out == 1) ? 0.0 : 1.0);
    int prev_not_choice = (prev_ch == 1) ? 2 : 1;

    for (int d = 14; d > 0; --d) state_hist[d] = state_hist[d-1];
    state_hist[0][0] = (double)prev_ch;
    state_hist[0][1] = (double)prev_out;
    state_hist[0][2] = (prev_rt - 0.75) / 0.50;
    state_hist[0][3] = r_in_1_prev;
    state_hist[0][4] = r_in_2_prev;
    state_hist[0][5] = (double)prev_not_choice;

    std::vector<double> u_MF(N_MF, 0.0);
    for(int j = 0; j < N_MF; ++j) {
        if (mf_d[j] == 0) {
            u_MF[j] = 1.0 / (1.0 + std::exp(-mf_beta[j] * state_hist[0][mf_c[j]])); 
        } else {
            int d_idx = std::min(mf_d[j], 14);
            u_MF[j] = 1.0 / (1.0 + std::exp(-mf_beta[j] * (state_hist[0][mf_c[j]] - state_hist[d_idx][mf_c[j]]))); 
        }
    }

    for (int i = 0; i < N_GC; ++i) {
        double in_sum = 0.0;
        for (int k = 0; k < 4; ++k) in_sum += gc_mossy_weights[i][k] * u_MF[gc_mossy_map[i][k]];
        double gamma_decay = rho_base + (1.0 - rho_base) * std::exp(-delta_t_val / tau_vec[i]);
        z_GC_curr[i] = in_sum + gamma_decay * z_GC_prev[i];
    }
    
    // Compression & ReLU
    for (int j = 0; j < N_comp; ++j) {
        double c_val = 0.0;
        for (int i = 0; i < N_GC; ++i) {
            c_val += W_comp[j][i] * z_GC_curr[i];
        }
        z_comp[j] = std::max(0.0, c_val); // ReLU
    }

    double y_PC1 = 0.0, y_PC2 = 0.0;
    for (int j = 0; j < N_comp; ++j) {
        y_PC1 += w1[j] * z_comp[j];
        y_PC2 += w2[j] * z_comp[j];
    }

    double v_t_ddm = b_v * (y_PC1 - y_PC2);
    double a_t = std::max(0.30, a_0 + k_mod * rpe_abs_prev);
    
    double rt_emp = rt_R[t];
    double dens = wiener_pdf(rt_emp, ch, v_t_ddm, a_t, t_nd);
    total_nll -= std::log(dens);

    double IO_error = ((double)out - 0.5) * 2.0 - ((ch == 1) ? y_PC1 : y_PC2);
    rpe_abs_prev = std::abs(IO_error);

    // RLS Update
    std::vector<std::vector<double>>& P = (ch == 1) ? P1 : P2;
    std::vector<double>& w = (ch == 1) ? w1 : w2;
    double target = ((double)out - 0.5) * 2.0;
    
    std::vector<double> Pz(N_comp, 0.0);
    double zPz = 0.0;
    for(int r=0; r<N_comp; ++r) {
        for(int c=0; c<N_comp; ++c) {
            Pz[r] += P[r][c] * z_comp[c];
        }
        zPz += z_comp[r] * Pz[r];
    }
    
    double denom = gamma_rls + zPz;
    std::vector<double> K(N_comp, 0.0);
    for(int r=0; r<N_comp; ++r) K[r] = Pz[r] / denom;
    
    double err = target - ((ch == 1) ? y_PC1 : y_PC2);
    for(int r=0; r<N_comp; ++r) w[r] += K[r] * err;
    
    for(int r=0; r<N_comp; ++r) {
        for(int c=0; c<N_comp; ++c) {
            P[r][c] = (P[r][c] - K[r] * Pz[c]) / gamma_rls;
        }
    }

    z_GC_prev = z_GC_curr;
  }

  if (std::isnan(total_nll) || std::isinf(total_nll)) return 1e9;
  return total_nll;
}
"
sourceCpp(code = cpp_code)

# Bounds
lower_bounds <- c(b_v = 0.0, a_0 = 0.30, t_nd = 0.10, k_mod = 0.0, mu_beta = -2.0, sigma_beta = 0.01, lambda_d = 0.0, mu_tau = -2.0, sigma_tau = 0.01, rho_base = 0.0, gamma_rls = 0.50, reg_lambda = 0.01)
upper_bounds <- c(b_v = 3.0, a_0 = 2.50, t_nd = 0.90, k_mod = 2.0, mu_beta =  2.0, sigma_beta = 2.00, lambda_d = 5.0, mu_tau =  2.0, sigma_tau = 2.00, rho_base = 0.95, gamma_rls = 1.00, reg_lambda = 5.00)
initial_phi <- lower_bounds + (upper_bounds - lower_bounds) / 2

# Grid loop
n_comp_list <- c(10, 20, 40)
results_df <- data.frame()

for (n_comp in n_comp_list) {
    cat(sprintf("\\n=== Starting Grid: N_comp = %d ===\\n", n_comp))
    
    loocv_nlls <- numeric(10)
    for (fold in 1:10) {
        ts_id <- sample_participants[fold]
        tr_ids <- sample_participants[-fold]
        
        tr_dat <- dat_all[dat_all$participant_id %in% tr_ids, ]
        ts_dat <- dat_all[dat_all$participant_id == ts_id, ]
        
        obj_fun <- function(phi) {
            evaluate_rls_cmaes(phi, tr_dat$choice, tr_dat$outcome, tr_dat$rt, tr_dat$trial_time_pos, tr_dat$trial_time_fix, as.integer(as.factor(tr_dat$participant_id)), n_comp)
        }
        
        res <- cma_es(initial_phi, obj_fun, lower = lower_bounds, upper = upper_bounds, control = list(maxit = 150, stopfitness = -Inf))
        
        ts_nll <- evaluate_rls_cmaes(res$par, ts_dat$choice, ts_dat$outcome, ts_dat$rt, ts_dat$trial_time_pos, ts_dat$trial_time_fix, rep(1, nrow(ts_dat)), n_comp)
        loocv_nlls[fold] <- ts_nll
        cat(sprintf(" Fold %d/10: NLL=%.2f\\n", fold, ts_nll))
    }
    
    mean_nll <- mean(loocv_nlls)
    results_df <- rbind(results_df, data.frame(N_comp=n_comp, Mean_NLL=mean_nll))
    cat(sprintf(">>> N_comp=%d completed. Mean NLL: %.2f\\n", n_comp, mean_nll))
}

write.csv(results_df, "results/tables/rls_compression_grid.csv", row.names=FALSE)
cat("\\nDone!\\n")
"""

with open("scratch/rls_compression_grid.R", "w") as f:
    f.write(r_script_content)

print("R script for RLS created.")
subprocess.run(["Rscript", "scratch/rls_compression_grid.R"], check=True)
