#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>

using namespace Rcpp;

// Wiener first passage time LPDF approximation for MAP
double wiener_lpdf(double t, double a, double tnd, double w, double v) {
    if (t <= tnd || a <= 0) return -1e10; // invalid state
    double tau = t - tnd;
    double v_sq = v * v;
    
    // Series approximation bounds
    double K = 5.0; // number of terms
    double term = 0.0;
    
    // Large time vs small time approximation (Navarro & Fuss 2009)
    if (tau > 0.15 * a * a) { // Large time
        for (int k = 1; k <= K; ++k) {
            double num = k * M_PI * sin(k * M_PI * w);
            double exp_term = exp(-0.5 * (v_sq * tau + (k * k * M_PI * M_PI * tau) / (a * a)));
            term += num * exp_term;
        }
        term /= (a * a);
    } else { // Small time
        for (int k = -std::ceil(K); k <= std::ceil(K); ++k) {
            double u = w + 2.0 * k;
            double exp_term = u * exp(- (a * a * u * u) / (2.0 * tau) - 0.5 * v_sq * tau);
            term += exp_term;
        }
        term /= (sqrt(2.0 * M_PI * pow(tau, 3.0)));
    }
    
    term *= exp(v * a * w);
    if (term <= 0) return -1e10;
    return log(term);
}

// [[Rcpp::export]]
double get_hierarchical_map_006(
    NumericVector group_mu,     // 9 means
    NumericVector group_sigma,  // 9 sigmas
    NumericMatrix z_scores,     // 9 x N_subj
    IntegerVector subj,
    IntegerVector resp,
    NumericVector reward,
    NumericVector rt,
    NumericVector iti,
    NumericVector min_rt,
    NumericMatrix W_exp         // N_subj x 32
) {
    int N_trials = rt.length();
    int N_subj = z_scores.ncol();
    
    double log_prior = 0.0;
    // Priors on group_mu ~ N(0, 1) and group_sigma ~ N+(0, 0.5)
    for (int i = 0; i < 9; ++i) {
        log_prior += R::dnorm(group_mu[i], 0.0, 1.0, 1);
        if (group_sigma[i] < 0) return 1e15; // invalid
        log_prior += R::dnorm(group_sigma[i], 0.0, 0.5, 1);
    }
    
    // Priors on z_scores ~ N(0, 1)
    for (int s = 0; s < N_subj; ++s) {
        for (int i = 0; i < 9; ++i) {
            log_prior += R::dnorm(z_scores(i, s), 0.0, 1.0, 1);
        }
    }
    
    // Subject specific parameters
    std::vector<double> a_base(N_subj), tnd(N_subj), v_ctx(N_subj);
    std::vector<double> alpha_ctx(N_subj), alpha_pc(N_subj), gamma(N_subj);
    std::vector<double> golgi_scale(N_subj), tau_decay(N_subj), w_u(N_subj);
    
    for (int s = 0; s < N_subj; ++s) {
        double t1 = group_mu[0] + group_sigma[0] * z_scores(0, s);
        double t2 = group_mu[1] + group_sigma[1] * z_scores(1, s);
        double t3 = group_mu[2] + group_sigma[2] * z_scores(2, s);
        double t4 = group_mu[3] + group_sigma[3] * z_scores(3, s);
        double t5 = group_mu[4] + group_sigma[4] * z_scores(4, s);
        double t6 = group_mu[5] + group_sigma[5] * z_scores(5, s);
        double t7 = group_mu[6] + group_sigma[6] * z_scores(6, s);
        double t8 = group_mu[7] + group_sigma[7] * z_scores(7, s);
        double t9 = group_mu[8] + group_sigma[8] * z_scores(8, s);
        
        a_base[s] = t1; // unconstrained raw for now, mapped in trials
        tnd[s] = min_rt[s] * (1.0 / (1.0 + exp(-t2)));
        v_ctx[s] = exp(t3);
        alpha_ctx[s] = 1.0 / (1.0 + exp(-t4));
        alpha_pc[s] = 1.0 / (1.0 + exp(-t5));
        gamma[s] = exp(t6);
        golgi_scale[s] = exp(t7);
        tau_decay[s] = exp(t8);
        w_u[s] = exp(t9);
    }
    
    double log_lik = 0.0;
    
    // Arrays for bases
    std::vector<double> frac_alpha(32), kappa_vec(32);
    for (int i = 0; i < 32; ++i) {
        frac_alpha[i] = 0.1 + 0.8 * (i / 31.0);
        kappa_vec[i] = 0.1 + 0.89 * (i / 31.0);
    }
    
    std::vector<std::vector<double>> Q(N_subj, std::vector<double>(2, 0.5));
    std::vector<std::vector<double>> frac_mem(N_subj, std::vector<double>(32, 0.0));
    std::vector<std::vector<double>> Z(N_subj, std::vector<double>(32, 0.0));
    std::vector<std::vector<double>> U_PC(N_subj, std::vector<double>(32, 0.0));
    std::vector<int> prev_ch(N_subj, 0); // 0-indexed
    std::vector<double> prev_E(N_subj, 0.0);
    
    for (int t = 0; t < N_trials; ++t) {
        int s = subj[t] - 1; // 0-indexed
        int ch = resp[t] - 1;
        double R = reward[t];
        double current_iti = (iti[t] < 0) ? 1.0 : iti[t];
        double phys_decay = exp(-current_iti / tau_decay[s]);
        
        double cb0 = 0.0;
        double cb1 = 0.0;
        
        for (int i = 0; i < 32; ++i) {
            double w_eff = 3.0 * tanh(U_PC[s][i] / 3.0);
            frac_mem[s][i] = frac_alpha[i] * frac_mem[s][i] + (1.0 - frac_alpha[i]) * W_exp(s, i) * Q[s][ch];
            Z[s][i] = phys_decay * kappa_vec[i] * Z[s][i] + tanh(frac_mem[s][i]);
            
            double S_mask = tanh(golgi_scale[s] * std::abs(w_eff * Z[s][i]));
            
            if (i < 16) cb0 += S_mask * w_eff * Z[s][i];
            else        cb1 += S_mask * w_eff * Z[s][i];
        }
        
        double veff = v_ctx[s] * (Q[s][1] - Q[s][0]) + gamma[s] * (cb1 - cb0);
        double safe_veff = (veff >= 0) ? (veff + 1e-4) : (veff - 1e-4);
        
        double raw_a_dyn = a_base[s] + w_u[s] * std::abs(cb0 * cb1);
        double a_dyn = 0.1 + 4.9 * (1.0 / (1.0 + exp(-raw_a_dyn)));
        
        double trial_ll = wiener_lpdf(rt[t], a_dyn, tnd[s], 0.5, safe_veff);
        if (trial_ll <= -1e9) {
            return 1e15; // Fatal bound hit
        }
        log_lik += trial_ll;
        
        prev_E[s] = R - Q[s][ch];
        Q[s][ch] += alpha_ctx[s] * prev_E[s];
        prev_ch[s] = ch;
        
        for (int i = 0; i < 32; ++i) {
            double err_sig = 0.0;
            if (prev_ch[s] == 0 && i < 16) err_sig = prev_E[s];
            if (prev_ch[s] == 1 && i >= 16) err_sig = prev_E[s];
            U_PC[s][i] += alpha_pc[s] * Z[s][i] * err_sig;
        }
    }
    
    // We are returning NLL (Negative Log Posterior)
    return -(log_lik + log_prior);
}
