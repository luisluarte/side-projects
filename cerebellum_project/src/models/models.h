#ifndef MODELS_H
#define MODELS_H

#include <Rcpp.h>
#include <vector>

double eval_wsls(const std::vector<double>& phi, const Rcpp::IntegerVector& resp, const Rcpp::IntegerVector& out, const Rcpp::NumericVector& rt);
double eval_eccm_intact(const std::vector<double>& phi, const Rcpp::IntegerVector& resp, const Rcpp::IntegerVector& out, const Rcpp::NumericVector& rt);
double eval_eccm_lesioned(const std::vector<double>& phi, const Rcpp::IntegerVector& resp, const Rcpp::IntegerVector& out, const Rcpp::NumericVector& rt);
double eval_eccm_temporal_decay(const std::vector<double>& phi, const Rcpp::IntegerVector& resp, const Rcpp::IntegerVector& out, const Rcpp::NumericVector& rt, const Rcpp::NumericVector& delta_t);

#endif
