// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include <cmath>
#include <algorithm>
#include <iostream>

using namespace Rcpp;
using namespace Eigen;

class ExactRModel {
private:
  // non mutable matrices
  SparseMatrix<double, RowMajor> W_in;
  SparseMatrix<double, RowMajor> W_fb;
  SparseMatrix<double, RowMajor> W_inh;
  SparseMatrix<double, RowMajor> W_collateral;

  // mutable matrices

  // critic
  VectorXd w_v;
  // actor
  MatrixXd W_pi;
  // bias term
  double b_v;

  // hyperparameters
  VectorXd rho_base;   // baseline retention per GC
  VectorXd tau_vector; // log-normal time constants per GC
  double eta_v;        // critic learning rate
  double eta_pi;       // actor learning rate
  double k_entropy;    // entropy sensibility
  double epsilon;      // L1-normalization constant
  int N_GC;            // reservoir dimension (GC cells)
  int N_GoC;           // golgi dimensions
  int N_actions;       // action space dimension

  // internal cerebelum state memory
  VectorXd z_GC_prev;
  // decision layer projection
  VectorXd y_prev;

  // cache
  VectorXd z_GC_curr;
  VectorXd pi_curr;
  double V_curr;

  // sigmoid activation function
  inline double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
  }

public:
  // constructor
  ExactRModel(MappedSparseMatrix<double> W_in_R,
              MappedSparseMatrix<double> W_fb_R,
              MappedSparseMatrix<double> W_inh_R,
              MappedSparseMatrix<double> W_collateral_R,
              NumericVector rho_base_R,
              NumericVector tau_vector_R,
              int n_actions, double lr_v, double lr_pi, double k_ent,
              double b_v_init) {

    W_in = W_in_R;
    W_fb = W_fb_R;
    W_inh = W_inh_R;
    W_collateral = W_collateral_R;

    rho_base = as<Map<VectorXd>>(rho_base_R);
    tau_vector = as<Map<VectorXd>>(tau_vector_R);

    N_GC = W_in.rows();
    N_GoC = W_fb.rows();
    N_actions = n_actions;

    eta_v = lr_v;
    eta_pi = lr_pi;
    k_entropy = k_ent;
    epsilon = 1e-12;

    // init vector state to zero
    z_GC_prev = VectorXd::Zero(N_GC);
    y_prev = VectorXd::Zero(1 + N_actions); // [V_t, pi_t]^T

    // init mutable matrices
    // as this are gradient-dependent value starting point
    // does not matter much
    w_v = VectorXd::Zero(N_GC);
    W_pi = MatrixXd::Zero(N_actions, N_GC);
    b_v = b_v_init;
  }

  // forward pass
  List forward_pass(NumericVector u_t_R, double delta_t) {

    // mossy fiber input
    Map<VectorXd> u_t(as<Map<VectorXd>>(u_t_R));

    // time decay function
    VectorXd gamma_dt(N_GC);
    for(int i = 0; i < N_GC; ++i) {
      gamma_dt[i] = rho_base[i] + (1.0 - rho_base[i]) *
        std::exp(-delta_t / tau_vector[i]);
    }

    // feedforward expansion W_in * u_t + \Gamma * z_{GC, t - 1}
    VectorXd h_pre = W_in * u_t;
    VectorXd fading_memory = gamma_dt.cwiseProduct(z_GC_prev);

    for(int i = 0; i < N_GC; ++i) {
      h_pre[i] = std::tanh(h_pre[i] + fading_memory[i]);
    }

    // golgi cell integration
    VectorXd GoC_excitation = W_fb * h_pre + W_collateral * y_prev;
    VectorXd z_GoC = GoC_excitation.cwiseMax(0.0); // ReLU

    // substract the inhibition
    VectorXd I_inh = W_inh * z_GoC;
    z_GC_curr = (h_pre - I_inh).cwiseMax(0.0); // max(0, h_pre - I_inh)

    // (purkinje) critic value estimation
    V_curr = w_v.dot(z_GC_curr) + b_v;

    // actor softmax
    VectorXd logits = W_pi * z_GC_curr;
    VectorXd exp_logits = logits.array().exp();
    pi_curr = exp_logits / exp_logits.sum();

    // update buffers for t + 1
    z_GC_prev = z_GC_curr;
    y_prev.segment(0, 1) << V_curr;
    y_prev.segment(1, N_actions) = pi_curr;

    return List::create(Named("Value") = V_curr,
                        Named("Policy") = pi_curr);
  }

  // backward pass
  List backward_pass(int action_taken, double reward, double ttf_t) {
    // adjust for indexing in R-related variables
    int a_t = action_taken - 1;

    // eligibility trace
    VectorXd gamma_trace(N_GC);
    for(int i = 0; i < N_GC; ++i) {
      gamma_trace[i] = rho_base[i] + (1.0 - rho_base[i]) * std::exp(-ttf_t / tau_vector[i]);
    }
    VectorXd e_t = gamma_trace.cwiseProduct(z_GC_curr);

    // entropy of GC as modulatory gate
    double l1_norm = e_t.cwiseAbs().sum() + epsilon;
    VectorXd p_t = e_t.cwiseAbs() / l1_norm; // spatial probability mass

    // actual entropy computation
    double S_t = 0.0;
    for(int i = 0; i < N_GC; ++i) {
      if(p_t[i] > 0) {
        S_t -= p_t[i] * std::log(p_t[i] + epsilon);
      }
    }
    double Omega_t = std::exp(-k_entropy * S_t);

    // actual reward prediction error
    double delta_t = reward - V_curr;

    // analytic w_v update, gradient descent
    // w_v = max(0, w_v + \eta_v * \Omega_t * \delta_t * e_t)
    VectorXd delta_w_v = eta_v * Omega_t * delta_t * e_t;
    w_v = (w_v + delta_w_v).cwiseMax(0.0);

    // critic intrinsic update
    // w_v = max(0, w_v + \eta_v * \Omega_t * \delta_t * e_t)
    double delta_b_v = eta_v * Omega_t * delta_t;
    b_v = std::max(0.0, b_v + delta_b_v);

    // actor update (this is the one that goes into the softmax)
    // W_{\pi, a_t} = max(0, W_{\pi, a_t} + \eta_\pi * \Omega_t * \delta_t * (1 - \pi_{t, a_t}) * e_t)
    VectorXd delta_W_pi = eta_pi * Omega_t * delta_t * (1.0 - pi_curr[a_t]) * e_t;
    W_pi.row(a_t) = (W_pi.row(a_t).transpose() + delta_W_pi).cwiseMax(0.0).transpose();

    return List::create(Named("RPE") = delta_t,
                        Named("Omega_t") = Omega_t,
                        Named("S_t") = S_t);
  }

  // state inspection & management
  NumericVector get_z_GC() {
    return wrap(z_GC_curr);
  }

  NumericVector get_z_GC_prev() {
    return wrap(z_GC_prev);
  }

  void set_z_GC_prev(NumericVector z_R) {
    Map<VectorXd> z_mapped(as<Map<VectorXd>>(z_R));
    z_GC_prev = z_mapped;
  }

  void reset_state() {
    z_GC_prev = VectorXd::Zero(N_GC);
    y_prev = VectorXd::Zero(1 + N_actions);
    z_GC_curr = VectorXd::Zero(N_GC);
    V_curr = 0.0;
    pi_curr = VectorXd::Zero(N_actions);
  }

  void scale_W_fb(double scale_factor) {
    W_fb = W_fb * scale_factor;
  }
};

// rcpp module exportation
RCPP_MODULE(exact_r_module) {
  class_<ExactRModel>("ExactRModel")
  .constructor<MappedSparseMatrix<double>, MappedSparseMatrix<double>,
  MappedSparseMatrix<double>, MappedSparseMatrix<double>,
  NumericVector, NumericVector, int, double, double, double, double>()
  .method("forward_pass", &ExactRModel::forward_pass)
  .method("backward_pass", &ExactRModel::backward_pass)
  .method("get_z_GC", &ExactRModel::get_z_GC)
  .method("get_z_GC_prev", &ExactRModel::get_z_GC_prev)
  .method("set_z_GC_prev", &ExactRModel::set_z_GC_prev)
  .method("reset_state", &ExactRModel::reset_state)
  .method("scale_W_fb", &ExactRModel::scale_W_fb);
}