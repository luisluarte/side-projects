library(dplyr)
library(ggplot2)
library(lme4)
library(zoo)
library(cmaes)
library(Rcpp)

cat("Loading dataset...\n")
dataset_path <- "data/raw/behavioral_compilate.csv"
if (!file.exists(dataset_path)) {
  dataset_path <- "c:/Users/DCCS5/Documents/one_for_all/cerebellum_project/datasets/behavioral_compilate.csv"
}

dat_all <- read.csv(dataset_path)
dat_all[['RT']] <- (as.numeric(dat_all[['ttr']]) - as.numeric(dat_all[['ttp']])) / 1000.0
dat_all <- dat_all[!is.na(dat_all[['RT']]) & dat_all[['RT']] >= 0.1 & dat_all[['RT']] <= 3.0 & dat_all[['Resp']] %in% c(1, 2), ]

set.seed(42)
participants <- unique(dat_all[['participant_id']])
num_participants <- length(participants)
dat_all <- dat_all[dat_all[['participant_id']] %in% participants, ]
dat_all$participant_factor <- as.integer(as.factor(dat_all$participant_id))

dat_all <- dat_all %>%
  group_by(participant_id) %>%
  arrange(ttp) %>%
  mutate(
    trial_idx = row_number(),
    n_trials = n(),
    is_test = ifelse(trial_idx > 0.7 * n_trials, 1, 0)
  ) %>%
  ungroup() %>%
  arrange(participant_factor, ttp)

cpp_code <- '
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

// CFMR
// [[Rcpp::export]]
List evaluate_cfmr_cv(const NumericVector& phi, const IntegerVector& resp_R, const IntegerVector& out_R, const IntegerVector& subj_idx_R, const NumericVector& rt_R, const IntegerVector& is_test_R, int num_participants) {
  double a = phi[0]; double t_nd = phi[1]; double beta_v = phi[2]; double eta_LTP = phi[3]; double eta_LTD = phi[4];
  double train_nll = 0.0; std::vector<double> out_prob1, out_pred_rt; std::vector<int> out_true_ch, out_true_out, out_subj;
  double Q1 = 0.0; double Q2 = 0.0;
  for (int t=0; t<resp_R.size(); ++t) {
    if (t > 0 && subj_idx_R[t] != subj_idx_R[t-1]) { Q1 = 0.0; Q2 = 0.0; }
    int ch = resp_R[t]; int out = out_R[t];
    double v_t_ddm = beta_v * (Q1 - Q2);
    double safe_v_t = std::abs(v_t_ddm) < 1e-4 ? (v_t_ddm >= 0 ? 1e-4 : -1e-4) : v_t_ddm;
    double dens = wiener_pdf(rt_R[t], ch, safe_v_t, a, t_nd);
    if (is_test_R[t] == 0) train_nll -= std::log(dens);
    else {
      out_prob1.push_back(1.0 / (1.0 + std::exp(-a * safe_v_t)));
      out_pred_rt.push_back(t_nd + (a / (2.0 * safe_v_t)) * std::tanh(a * safe_v_t / 2.0));
      out_true_ch.push_back(ch); out_true_out.push_back(out); out_subj.push_back(subj_idx_R[t]);
    }
    if (ch == 1) { Q1 += (out == 1) ? eta_LTP * (1.0 - Q1) : eta_LTD * (-1.0 - Q1); } 
    else         { Q2 += (out == 1) ? eta_LTP * (1.0 - Q2) : eta_LTD * (-1.0 - Q2); }
  }
  return List::create(Named("train_nll") = train_nll, Named("prob1") = out_prob1, Named("pred_rt") = out_pred_rt, Named("true_ch") = out_true_ch, Named("true_out") = out_true_out, Named("subj") = out_subj);
}
'
sourceCpp(code = cpp_code)

cat("Optimizing CFMR (5 parameters) to get predictions...\n")
obj_cfmr <- function(phi) {
    if (any(phi < c(0.1, 0.01, 0.0, 0.0, 0.0)) || any(phi > c(5.0, 1.0, 10.0, 1.0, 1.0))) return(1e9)
    res <- evaluate_cfmr_cv(as.numeric(phi), as.integer(dat_all$Resp), as.integer(dat_all$F), as.integer(dat_all$participant_factor), as.numeric(dat_all$RT), as.integer(dat_all$is_test), num_participants)
    return(res$train_nll)
}
opt_cfmr <- cma_es(c(1.0, 0.3, 2.0, 0.5, 0.5), obj_cfmr, lower=c(0.1, 0.01, 0.0, 0.0, 0.0), upper=c(5.0, 1.0, 10.0, 1.0, 1.0), control=list(maxit=50, trace=FALSE))
res_cfmr <- evaluate_cfmr_cv(opt_cfmr$par, as.integer(dat_all$Resp), as.integer(dat_all$F), as.integer(dat_all$participant_factor), as.numeric(dat_all$RT), as.integer(dat_all$is_test), num_participants)


cat("Computing CFMR adherence...\n")
# We use only the test trials since res_cfmr outputs test trial probabilities
df_test <- dat_all %>% filter(is_test == 1) %>% mutate(
    cfmr_prob1 = res_cfmr$prob1,
    cfmr_pred = ifelse(cfmr_prob1 >= 0.5, 1, 2),
    cfmr_adherence = as.integer(Resp == cfmr_pred)
)

# -------------------------------------------------------------------
# Option 1: Rolling Window CFMR Adherence
# -------------------------------------------------------------------
cat("Computing Option 1: Rolling Window CFMR Adherence...\n")

window_size <- 20

# Compute rolling average per participant
df_rolling <- df_test %>%
  group_by(participant_id) %>%
  arrange(trial_idx) %>%
  mutate(
    rolling_adherence = rollapply(cfmr_adherence, width = window_size, FUN = mean, align = "right", fill = NA)
  ) %>%
  ungroup()

# Aggregate across participants by trial
agg_rolling <- df_rolling %>%
  filter(!is.na(rolling_adherence)) %>%
  group_by(trial_idx) %>%
  summarize(
    mean_adherence = mean(rolling_adherence, na.rm = TRUE),
    se_adherence = sd(rolling_adherence, na.rm = TRUE) / sqrt(n())
  )

p <- ggplot(agg_rolling, aes(x = trial_idx, y = mean_adherence)) +
  geom_line(color = "#8e44ad", linewidth = 1.2) +
  geom_ribbon(aes(ymin = mean_adherence - se_adherence, ymax = mean_adherence + se_adherence), alpha = 0.2, fill = "#8e44ad") +
  geom_smooth(method = "lm", color = "#e74c3c", linetype = "dashed", se = FALSE) +
  theme_minimal() +
  labs(
    title = "CFMR Adherence over Time",
    subtitle = "Rolling 20-trial average of CFMR model adherence across 128 participants",
    x = "Trial Number",
    y = "CFMR Adherence Probability"
  )

plot_path <- "C:/Users/DCCS5/.gemini/antigravity/brain/1d8f9958-fd49-4502-b57b-97a7887eb7ad/cfmr_adherence_over_time.png"
ggsave(plot_path, p, width = 8, height = 5)
cat(sprintf("Saved plot to %s\n", plot_path))


# -------------------------------------------------------------------
# Option 2: Time-Varying Logistic Regression
# -------------------------------------------------------------------
cat("\nComputing Option 2: GLMM for CFMR Adherence vs Trial...\n")

# Scale trial index to prevent convergence warnings
df_test$scaled_trial <- scale(df_test$trial_idx)

model <- glmer(cfmr_adherence ~ scaled_trial + (1 | participant_id), 
               family = binomial(link = "logit"), 
               data = df_test,
               control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=100000)))

cat("\n--- GLMM Summary ---\n")
print(summary(model))

coefs <- summary(model)$coefficients
cat(sprintf("\nEffect of Time on CFMR Adherence: Z = %.3f, p-value = %.3e\n", 
            coefs["scaled_trial", "z value"], 
            coefs["scaled_trial", "Pr(>|z|)"]))
