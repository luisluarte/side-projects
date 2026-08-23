#include <Rcpp.h>
#include <vector>

// Include models directly to compile as a single translation unit in sourceCpp
#include "../models/shared_utils.h"
#include "../models/wsls.cpp"
#include "../models/eccm_intact.cpp"
#include "../models/eccm_lesioned.cpp"
#include "../models/eccm_temporal_decay.cpp"
#include "../models/eccm_switch_mechanisms.cpp"
#include "../models/eccm_disagreement_only.cpp"
#include "../models/eccm_cortical_rpe.cpp"
#include "../models/eccm_weighted_fit.cpp"
#include "../models/eccm_multiplexed.cpp"
#include "../models/eccm_hill.cpp"
#include "../models/eccm_dynamic_boundary.cpp"
#include "../models/eccm_uncertainty_reservoir.cpp"
#include "../models/eccm_golgi.cpp"
#include "../models/eccm_golgi_relu.cpp"
#include "../models/eccm_golgi_variants.cpp"
#include "../models/eccm_entropy_reversal.cpp"
#include "../models/eccm_golgi_explore.cpp"
#include "../models/eccm_golgi_reversal.cpp"
#include "../models/eccm_golgi_asym_reversal.cpp"
#include "../models/eccm_mf_rev_ablated.cpp"
#include "../models/eccm_lca.cpp"
#include "../models/eccm_bvk.cpp"

using namespace Rcpp;

// [[Rcpp::export]]
NumericMatrix run_mcmc_subject(int model_type, int iters, std::vector<double> init_phi, const IntegerVector& resp, const IntegerVector& out, const NumericVector& rt, const NumericVector& delta_t = NumericVector::create()) {
    // model_type: 0-19 as before, 20=ECCM_MF_Rev_Ablated
    int n_params = init_phi.size();
    NumericMatrix chain(iters, n_params);
    std::vector<double> current_phi = init_phi;
    
    double current_ll = 1e9;
    if (model_type == 0) current_ll = eval_wsls(current_phi, resp, out, rt);
    else if (model_type == 1) current_ll = eval_eccm_intact(current_phi, resp, out, rt);
    else if (model_type == 2) current_ll = eval_eccm_lesioned(current_phi, resp, out, rt);
    else if (model_type == 3) current_ll = eval_eccm_temporal_decay(current_phi, resp, out, rt, delta_t);
    else if (model_type == 4) current_ll = eval_eccm_switch_mechanisms(current_phi, resp, out, rt, delta_t);
    else if (model_type == 5) current_ll = eval_eccm_disagreement_only(current_phi, resp, out, rt, delta_t);
    else if (model_type == 6) current_ll = eval_eccm_cortical_rpe(current_phi, resp, out, rt, delta_t);
    else if (model_type == 7) current_ll = eval_eccm_weighted_fit(current_phi, resp, out, rt, delta_t);
    else if (model_type == 8) current_ll = eval_eccm_multiplexed(current_phi, resp, out, rt, delta_t);
    else if (model_type == 9) current_ll = eval_eccm_hill(current_phi, resp, out, rt, delta_t);
    else if (model_type == 10) current_ll = eval_eccm_dynamic_boundary(current_phi, resp, out, rt, delta_t);
    else if (model_type == 11) current_ll = eval_eccm_uncertainty_reservoir(current_phi, resp, out, rt, delta_t);
    else if (model_type == 12) current_ll = eval_eccm_golgi(current_phi, resp, out, rt, delta_t);
    else if (model_type == 13) current_ll = eval_eccm_golgi_relu(current_phi, resp, out, rt, delta_t);
    else if (model_type == 14) current_ll = eval_eccm_golgi_ceiling(current_phi, resp, out, rt, delta_t);
    else if (model_type == 15) current_ll = eval_eccm_golgi_softmax(current_phi, resp, out, rt, delta_t);
    else if (model_type == 16) current_ll = eval_eccm_entropy_reversal(current_phi, resp, out, rt, delta_t);
    else if (model_type == 17) current_ll = eval_eccm_golgi_explore(current_phi, resp, out, rt, delta_t);
    else if (model_type == 18) current_ll = eval_eccm_golgi_reversal(current_phi, resp, out, rt, delta_t);
    else if (model_type == 19) current_ll = eval_eccm_golgi_asym_reversal(current_phi, resp, out, rt, delta_t);
    else if (model_type == 20) current_ll = eval_eccm_mf_rev_ablated(current_phi, resp, out, rt, delta_t);
    else if (model_type == 22) current_ll = eval_eccm_lca(current_phi, resp, out, rt, delta_t);
    else if (model_type == 23) current_ll = eval_eccm_bvk(current_phi, resp, out, rt, delta_t);
    
    SimpleRNG rng(123);
    
    for (int iter=0; iter<iters; ++iter) {
        for (int p=0; p<n_params; ++p) {
            std::vector<double> prop_phi = current_phi;
            prop_phi[p] += rng.rnorm() * 0.05; 
            
            double prop_ll = 1e9;
            if (model_type == 0) prop_ll = eval_wsls(prop_phi, resp, out, rt);
            else if (model_type == 1) prop_ll = eval_eccm_intact(prop_phi, resp, out, rt);
            else if (model_type == 2) prop_ll = eval_eccm_lesioned(prop_phi, resp, out, rt);
            else if (model_type == 3) prop_ll = eval_eccm_temporal_decay(prop_phi, resp, out, rt, delta_t);
            else if (model_type == 4) prop_ll = eval_eccm_switch_mechanisms(prop_phi, resp, out, rt, delta_t);
            else if (model_type == 5) prop_ll = eval_eccm_disagreement_only(prop_phi, resp, out, rt, delta_t);
            else if (model_type == 6) prop_ll = eval_eccm_cortical_rpe(prop_phi, resp, out, rt, delta_t);
            else if (model_type == 7) prop_ll = eval_eccm_weighted_fit(prop_phi, resp, out, rt, delta_t);
            else if (model_type == 8) prop_ll = eval_eccm_multiplexed(prop_phi, resp, out, rt, delta_t);
            else if (model_type == 9) prop_ll = eval_eccm_hill(prop_phi, resp, out, rt, delta_t);
            else if (model_type == 10) prop_ll = eval_eccm_dynamic_boundary(prop_phi, resp, out, rt, delta_t);
            else if (model_type == 11) prop_ll = eval_eccm_uncertainty_reservoir(prop_phi, resp, out, rt, delta_t);
            else if (model_type == 12) prop_ll = eval_eccm_golgi(prop_phi, resp, out, rt, delta_t);
            else if (model_type == 13) prop_ll = eval_eccm_golgi_relu(prop_phi, resp, out, rt, delta_t);
            else if (model_type == 14) prop_ll = eval_eccm_golgi_ceiling(prop_phi, resp, out, rt, delta_t);
            else if (model_type == 15) prop_ll = eval_eccm_golgi_softmax(prop_phi, resp, out, rt, delta_t);
            else if (model_type == 16) prop_ll = eval_eccm_entropy_reversal(prop_phi, resp, out, rt, delta_t);
            else if (model_type == 17) prop_ll = eval_eccm_golgi_explore(prop_phi, resp, out, rt, delta_t);
            else if (model_type == 18) prop_ll = eval_eccm_golgi_reversal(prop_phi, resp, out, rt, delta_t);
            else if (model_type == 19) prop_ll = eval_eccm_golgi_asym_reversal(prop_phi, resp, out, rt, delta_t);
            else if (model_type == 20) prop_ll = eval_eccm_mf_rev_ablated(prop_phi, resp, out, rt, delta_t);
            else if (model_type == 22) prop_ll = eval_eccm_lca(prop_phi, resp, out, rt, delta_t);
            else if (model_type == 23) prop_ll = eval_eccm_bvk(prop_phi, resp, out, rt, delta_t);
            
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
