#ifndef MODELS_H
#define MODELS_H

#include <Rcpp.h>
#include <vector>

double eval_wsls(const std::vector<double>& phi, const Rcpp::IntegerVector& resp, const Rcpp::IntegerVector& out, const Rcpp::NumericVector& rt);
double eval_eccm_intact(const std::vector<double>& phi, const Rcpp::IntegerVector& resp, const Rcpp::IntegerVector& out, const Rcpp::NumericVector& rt);
double eval_eccm_lesioned(const std::vector<double>& phi, const Rcpp::IntegerVector& resp, const Rcpp::IntegerVector& out, const Rcpp::NumericVector& rt);
double eval_eccm_temporal_decay(const std::vector<double>& phi, const Rcpp::IntegerVector& resp, const Rcpp::IntegerVector& out, const Rcpp::NumericVector& rt, const Rcpp::NumericVector& delta_t);
double eval_eccm_switch_mechanisms(const std::vector<double>& phi, const Rcpp::IntegerVector& resp, const Rcpp::IntegerVector& out, const Rcpp::NumericVector& rt, const Rcpp::NumericVector& delta_t);
double eval_eccm_disagreement_only(const std::vector<double>& phi, const Rcpp::IntegerVector& resp, const Rcpp::IntegerVector& out, const Rcpp::NumericVector& rt, const Rcpp::NumericVector& delta_t);
double eval_eccm_cortical_rpe(const std::vector<double>& phi, const Rcpp::IntegerVector& resp, const Rcpp::IntegerVector& out, const Rcpp::NumericVector& rt, const Rcpp::NumericVector& delta_t);
double eval_eccm_weighted_fit(const std::vector<double>& phi, const Rcpp::IntegerVector& resp, const Rcpp::IntegerVector& out, const Rcpp::NumericVector& rt, const Rcpp::NumericVector& delta_t);
double eval_eccm_multiplexed(const std::vector<double>& phi, const Rcpp::IntegerVector& resp, const Rcpp::IntegerVector& out, const Rcpp::NumericVector& rt, const Rcpp::NumericVector& delta_t);
double eval_eccm_hill(const std::vector<double>& phi, const Rcpp::IntegerVector& resp, const Rcpp::IntegerVector& out, const Rcpp::NumericVector& rt, const Rcpp::NumericVector& delta_t);
double eval_eccm_dynamic_boundary(const std::vector<double>& phi, const Rcpp::IntegerVector& resp, const Rcpp::IntegerVector& out, const Rcpp::NumericVector& rt, const Rcpp::NumericVector& delta_t);
double eval_eccm_dynamic_boundary(const std::vector<double>& phi, const Rcpp::IntegerVector& resp, const Rcpp::IntegerVector& out, const Rcpp::NumericVector& rt, const Rcpp::NumericVector& delta_t);
double eval_eccm_uncertainty_reservoir(const std::vector<double>& phi, const Rcpp::IntegerVector& resp, const Rcpp::IntegerVector& out, const Rcpp::NumericVector& rt, const Rcpp::NumericVector& delta_t);
double eval_eccm_uncertainty_reservoir(const std::vector<double>& phi, const Rcpp::IntegerVector& resp, const Rcpp::IntegerVector& out, const Rcpp::NumericVector& rt, const Rcpp::NumericVector& delta_t);
double eval_eccm_golgi(const std::vector<double>& phi, const Rcpp::IntegerVector& resp, const Rcpp::IntegerVector& out, const Rcpp::NumericVector& rt, const Rcpp::NumericVector& delta_t);
double eval_eccm_golgi_relu(const std::vector<double>& phi, const Rcpp::IntegerVector& resp, const Rcpp::IntegerVector& out, const Rcpp::NumericVector& rt, const Rcpp::NumericVector& delta_t);
double eval_eccm_golgi_ceiling(const std::vector<double>& phi, const Rcpp::IntegerVector& resp, const Rcpp::IntegerVector& out, const Rcpp::NumericVector& rt, const Rcpp::NumericVector& delta_t);
double eval_eccm_golgi_softmax(const std::vector<double>& phi, const Rcpp::IntegerVector& resp, const Rcpp::IntegerVector& out, const Rcpp::NumericVector& rt, const Rcpp::NumericVector& delta_t);
double eval_eccm_entropy_reversal(const std::vector<double>& phi, const Rcpp::IntegerVector& resp, const Rcpp::IntegerVector& out, const Rcpp::NumericVector& rt, const Rcpp::NumericVector& delta_t);

#endif
