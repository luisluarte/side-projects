import re

file_path = "src/cpp/reservoir_loocv_cmaes.cpp"
with open(file_path, "r") as f:
    content = f.read()

parts = content.split("// [[Rcpp::export]]")

new_eval = """
double evaluate_loocv_cmaes_objective_cpp(
    const NumericVector& phi_15d,
    const IntegerVector& resp_R,
    const IntegerVector& out_R,
    const NumericVector& m1_R,
    const NumericVector& m2_R,
    const NumericVector& rt_R,
    const NumericVector& ttp_R,
    const NumericVector& ttf_R,
    const IntegerVector& subj_idx_R
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
  double alpha_LTD   = phi_15d[10];
  double alpha_LTP   = phi_15d[11];
  
  int N_GC = 500;
  int N_MF = 100;
  
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

  std::vector<double> tau_vec(N_GC, 1.0);
  for (int i = 0; i < N_GC; ++i) {
      double tau_raw = mu_tau + sigma_tau * rng.rnorm();
      tau_vec[i] = std::exp(tau_raw);
  }

  std::vector<double> z_GC_curr(N_GC, 0.0);
  std::vector<double> z_GC_prev(N_GC, 0.0);
  
  std::vector<double> w_PF1(N_GC, 0.0);
  std::vector<double> w_PF2(N_GC, 0.0);

  std::vector<std::vector<double>> state_hist(15, std::vector<double>(6, 0.0));
  
  double total_nll = 0.0;
  double rpe_abs_prev = 0.0;
  
  for (int t = 0; t < N_t; ++t) {
    if (t > 0 && subj_idx_R[t] != subj_idx_R[t - 1]) {
      std::fill(z_GC_prev.begin(), z_GC_prev.end(), 0.0);
      std::fill(z_GC_curr.begin(), z_GC_curr.end(), 0.0);
      std::fill(w_PF1.begin(), w_PF1.end(), 0.0);
      std::fill(w_PF2.begin(), w_PF2.end(), 0.0);
      for (int d = 0; d < 15; ++d) std::fill(state_hist[d].begin(), state_hist[d].end(), 0.0);
      rpe_abs_prev = 0.0;
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

    double y_PC1 = 0.0, y_PC2 = 0.0;
    for (int i = 0; i < N_GC; ++i) {
        y_PC1 += w_PF1[i] * z_GC_curr[i];
        y_PC2 += w_PF2[i] * z_GC_curr[i];
    }

    double v_t_ddm = b_v * (y_PC1 - y_PC2);
    double a_t = std::max(0.30, a_0 + k_mod * rpe_abs_prev);
    
    double rt_emp = rt_R[t];
    double dens = wiener_pdf(rt_emp, ch, v_t_ddm, a_t, t_nd);
    total_nll -= std::log(dens);

    double IO_error = (double)out - ((ch == 1) ? y_PC1 : y_PC2);
    rpe_abs_prev = std::abs(IO_error);

    double L1_norm1 = 0.0, L1_norm2 = 0.0;
    for (int i = 0; i < N_GC; ++i) {
        L1_norm1 += std::abs(w_PF1[i]);
        L1_norm2 += std::abs(w_PF2[i]);
    }
    double eps = 1e-4;
    
    for (int i = 0; i < N_GC; ++i) {
        if (ch == 1) {
            double P_i = std::abs(w_PF1[i]) / (L1_norm1 + eps);
            double delta_w = -alpha_LTD * IO_error * z_GC_prev[i] * P_i + alpha_LTP * z_GC_prev[i];
            w_PF1[i] += delta_w;
        } else {
            double P_i = std::abs(w_PF2[i]) / (L1_norm2 + eps);
            double delta_w = -alpha_LTD * IO_error * z_GC_prev[i] * P_i + alpha_LTP * z_GC_prev[i];
            w_PF2[i] += delta_w;
        }
        z_GC_prev[i] = z_GC_curr[i];
    }
  }

  if (std::isnan(total_nll) || std::isinf(total_nll)) return 1e9;
  return total_nll;
}
"""

new_switch = """
NumericVector get_loocv_cmaes_switch_prob_cpp(
    const NumericVector& phi_15d,
    const IntegerVector& resp_R,
    const IntegerVector& out_R,
    const NumericVector& m1_R,
    const NumericVector& m2_R,
    const NumericVector& rt_R,
    const NumericVector& ttp_R,
    const NumericVector& ttf_R,
    const IntegerVector& subj_idx_R
) {
  int N_t = resp_R.size();
  NumericVector p_switch(N_t);
  
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
  double alpha_LTD   = phi_15d[10];
  double alpha_LTP   = phi_15d[11];
  
  int N_GC = 500;
  int N_MF = 100;
  
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

  std::vector<double> tau_vec(N_GC, 1.0);
  for (int i = 0; i < N_GC; ++i) {
      double tau_raw = mu_tau + sigma_tau * rng.rnorm();
      tau_vec[i] = std::exp(tau_raw);
  }

  std::vector<double> z_GC_curr(N_GC, 0.0);
  std::vector<double> z_GC_prev(N_GC, 0.0);
  
  std::vector<double> w_PF1(N_GC, 0.0);
  std::vector<double> w_PF2(N_GC, 0.0);

  std::vector<std::vector<double>> state_hist(15, std::vector<double>(6, 0.0));
  double rpe_abs_prev = 0.0;
  
  for (int t = 0; t < N_t; ++t) {
    if (t > 0 && subj_idx_R[t] != subj_idx_R[t - 1]) {
      std::fill(z_GC_prev.begin(), z_GC_prev.end(), 0.0);
      std::fill(z_GC_curr.begin(), z_GC_curr.end(), 0.0);
      std::fill(w_PF1.begin(), w_PF1.end(), 0.0);
      std::fill(w_PF2.begin(), w_PF2.end(), 0.0);
      for (int d = 0; d < 15; ++d) std::fill(state_hist[d].begin(), state_hist[d].end(), 0.0);
      rpe_abs_prev = 0.0;
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

    double y_PC1 = 0.0, y_PC2 = 0.0;
    for (int i = 0; i < N_GC; ++i) {
        y_PC1 += w_PF1[i] * z_GC_curr[i];
        y_PC2 += w_PF2[i] * z_GC_curr[i];
    }

    double v_t_ddm = b_v * (y_PC1 - y_PC2);
    double a_t = std::max(0.30, a_0 + k_mod * rpe_abs_prev);
    
    double p_ch1 = 1.0 / (1.0 + std::exp(-v_t_ddm * a_t));
    p_switch[t] = (prev_ch == 1) ? (1.0 - p_ch1) : p_ch1;

    double IO_error = (double)out - ((ch == 1) ? y_PC1 : y_PC2);
    rpe_abs_prev = std::abs(IO_error);

    double L1_norm1 = 0.0, L1_norm2 = 0.0;
    for (int i = 0; i < N_GC; ++i) {
        L1_norm1 += std::abs(w_PF1[i]);
        L1_norm2 += std::abs(w_PF2[i]);
    }
    double eps = 1e-4;
    
    for (int i = 0; i < N_GC; ++i) {
        if (ch == 1) {
            double P_i = std::abs(w_PF1[i]) / (L1_norm1 + eps);
            double delta_w = -alpha_LTD * IO_error * z_GC_prev[i] * P_i + alpha_LTP * z_GC_prev[i];
            w_PF1[i] += delta_w;
        } else {
            double P_i = std::abs(w_PF2[i]) / (L1_norm2 + eps);
            double delta_w = -alpha_LTD * IO_error * z_GC_prev[i] * P_i + alpha_LTP * z_GC_prev[i];
            w_PF2[i] += delta_w;
        }
        z_GC_prev[i] = z_GC_curr[i];
    }
  }

  return p_switch;
}
"""

for i, part in enumerate(parts):
    if "double evaluate_loocv_cmaes_objective_cpp" in part:
        parts[i] = new_eval
    elif "NumericVector get_loocv_cmaes_switch_prob_cpp" in part:
        parts[i] = new_switch

with open(file_path, "w") as f:
    f.write("// [[Rcpp::export]]".join(parts))
print("Replaced functions successfully!")
