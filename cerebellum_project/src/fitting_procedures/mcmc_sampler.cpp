#include <Rcpp.h>
#include <vector>

// Include models directly to compile as a single translation unit in sourceCpp
#include "../models/shared_utils.h"
#include "../models/wsls.cpp"
#include "../models/eccm_intact.cpp"
#include "../models/eccm_lesioned.cpp"

using namespace Rcpp;

// [[Rcpp::export]]
NumericMatrix run_mcmc_subject(int model_type, int iters, std::vector<double> init_phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt) {
    // model_type: 0=WSLS, 1=ECCM_Intact, 2=ECCM_Lesioned
    int n_params = init_phi.size();
    NumericMatrix chain(iters, n_params);
    std::vector<double> current_phi = init_phi;
    
    double current_ll = 1e9;
    if (model_type == 0) current_ll = eval_wsls(current_phi, resp, out, rt);
    else if (model_type == 1) current_ll = eval_eccm_intact(current_phi, resp, out, rt);
    else if (model_type == 2) current_ll = eval_eccm_lesioned(current_phi, resp, out, rt);
    
    SimpleRNG rng(123);
    
    for (int iter=0; iter<iters; ++iter) {
        for (int p=0; p<n_params; ++p) {
            std::vector<double> prop_phi = current_phi;
            prop_phi[p] += rng.rnorm() * 0.05; 
            
            double prop_ll = 1e9;
            if (model_type == 0) prop_ll = eval_wsls(prop_phi, resp, out, rt);
            else if (model_type == 1) prop_ll = eval_eccm_intact(prop_phi, resp, out, rt);
            else if (model_type == 2) prop_ll = eval_eccm_lesioned(prop_phi, resp, out, rt);
            
            // Acceptance
            if (prop_ll < current_ll || std::log(rng.runif()) < (current_ll - prop_ll)) {
                current_phi = prop_phi;
                current_ll = prop_ll;
            }
        }
        for (int p=0; p<n_params; ++p) chain(iter, p) = current_phi[p];
    }
    return chain;
}
